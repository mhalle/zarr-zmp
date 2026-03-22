from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterable
from pathlib import Path
from typing import Any, Self

from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype

from zmanifest._types import Addressing
from zmanifest.builder import Builder, canonical_json, git_blob_hash
from zmanifest.manifest import Manifest, ManifestEntry
from zmanifest.path import ZPath
from zmanifest.resolve import (
    Resolver,
    build_base_chain,
    get_file_base_resolve,
    resolve_entry,
)
from zmanifest.resolver import HttpResolver, GitResolver

# Type alias for mount opener callbacks (zarr Store-typed version)
ZarrMountOpener = Callable[[ManifestEntry], Store]
MountOpener = ZarrMountOpener

_ZARR_METADATA_NAMES = frozenset(
    ("zarr.json", ".zarray", ".zgroup", ".zattrs", ".zmetadata")
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_byte_range(data: bytes, byte_range: ByteRequest | None) -> bytes:
    if byte_range is None:
        return data
    if isinstance(byte_range, RangeByteRequest):
        return data[byte_range.start : byte_range.end]
    if isinstance(byte_range, OffsetByteRequest):
        return data[byte_range.offset :]
    if isinstance(byte_range, SuffixByteRequest):
        return data[-byte_range.suffix :]
    return data


def _is_zarr_metadata(path: ZPath) -> bool:
    """Check if a path is a zarr metadata file."""
    return path.name in _ZARR_METADATA_NAMES


# ---------------------------------------------------------------------------
# Writable store
# ---------------------------------------------------------------------------


class ZMPWritableStore(Store):
    """Writable zarr store that buffers writes and flushes to a ZMP file.

    Two modes controlled by ``chunk_dir``:

    - **Embedded** (``chunk_dir=None``): all content is inlined in the
      parquet file — metadata as ``text``, chunks as ``data``.
    - **External** (``chunk_dir="path/to/dir"``): chunks are written as
      files to the directory named by their git-sha1 hash; the manifest
      references them via a resolve dict with a git oid.

    In both modes, JSON content (zarr.json etc.) is canonicalized via
    RFC 8785 before hashing so the git-sha1 is deterministic regardless
    of key ordering.

    Args:
        output: Path for the output ``.zmp`` parquet file.
        chunk_dir: If set, write chunk blobs to this directory instead
            of inlining them. Directory is created if needed.
        max_rows_per_group: Override adaptive row group sizing.
        zarr_format: Zarr format version for file-level metadata.
        metadata: Additional key-value pairs for file-level metadata.
        base_resolve: File-level base resolution params.
    """

    supports_writes = True
    supports_deletes = True
    supports_partial_writes = False
    supports_listing = True

    def __init__(
        self,
        output: str | Path,
        *,
        chunk_dir: str | Path | None = None,
        max_rows_per_group: int | None = None,
        zarr_format: str = "3",
        metadata: dict[str, object] | None = None,
        base_resolve: dict | None = None,
    ) -> None:
        super().__init__(read_only=False)
        self._output = Path(output)
        self._chunk_dir = Path(chunk_dir) if chunk_dir is not None else None
        self._max_rows_per_group = max_rows_per_group
        self._zarr_format = zarr_format
        self._metadata = metadata or {}
        self._base_resolve = base_resolve
        self._entries: dict[str, bytes] = {}  # keyed by zarr bare path
        self._is_open = True

    @classmethod
    def create(
        cls,
        output: str | Path,
        *,
        chunk_dir: str | Path | None = None,
        max_rows_per_group: int | None = None,
        zarr_format: str = "3",
        metadata: dict[str, object] | None = None,
        base_resolve: dict | None = None,
    ) -> ZMPWritableStore:
        return cls(
            output,
            chunk_dir=chunk_dir,
            max_rows_per_group=max_rows_per_group,
            zarr_format=zarr_format,
            metadata=metadata,
            base_resolve=base_resolve,
        )

    def __enter__(self) -> ZMPWritableStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.commit()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ZMPWritableStore):
            return NotImplemented
        return self._output == other._output

    async def open(self, *args: Any, **kwargs: Any) -> Self:
        self._is_open = True
        return self

    async def close(self) -> None:
        self.commit()
        self._is_open = False

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        raw = self._entries.get(key)
        if raw is None:
            return None
        raw = _apply_byte_range(raw, byte_range)
        return prototype.buffer.from_bytes(raw)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        return [
            await self.get(key, prototype, byte_range)
            for key, byte_range in key_ranges
        ]

    async def exists(self, key: str) -> bool:
        return key in self._entries

    async def set(self, key: str, value: Buffer) -> None:
        self._entries[key] = value.to_bytes()

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        if key not in self._entries:
            self._entries[key] = value.to_bytes()

    async def delete(self, key: str) -> None:
        self._entries.pop(key, None)

    async def list(self) -> AsyncIterator[str]:
        for p in sorted(self._entries):
            yield p

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        # zarr passes bare prefix strings
        zprefix = ZPath.from_zarr(prefix)
        for key in sorted(self._entries):
            zkey = ZPath.from_zarr(key)
            if zkey.is_equal_or_child_of(zprefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        zprefix = ZPath.from_zarr(prefix)
        seen: set[str] = set()
        for key in self._entries:
            zkey = ZPath.from_zarr(key)
            child = zkey.child_name_under(zprefix)
            if child is None:
                continue
            # If this child has deeper entries, it's a directory
            is_dir = (zprefix / child) != zkey
            entry = child + "/" if is_dir else child
            if entry not in seen:
                seen.add(entry)
                yield entry

    def commit(self) -> Path:
        """Flush buffered writes to the ZMP parquet file.

        For external mode, also writes chunk blobs to ``chunk_dir``.

        Returns:
            Path to the written ``.zmp`` file.
        """
        external = self._chunk_dir is not None
        if external:
            self._chunk_dir.mkdir(parents=True, exist_ok=True)

        builder = Builder(
            zarr_format=self._zarr_format,
            max_rows_per_group=self._max_rows_per_group,
            metadata=self._metadata,
            base_resolve=self._base_resolve,
        )

        for key in sorted(self._entries):
            raw = self._entries[key]
            zpath = ZPath.from_zarr(key)
            is_meta = _is_zarr_metadata(zpath)

            if is_meta:
                text = canonical_json(raw.decode("utf-8"))
                builder.add(key, text=text)
            elif external:
                rk = git_blob_hash(raw)
                blob_path = self._chunk_dir / rk
                if not blob_path.exists():
                    blob_path.write_bytes(raw)
                builder.add(
                    key,
                    resolve={"git": {"oid": rk}},
                    checksum=rk,
                    size=len(raw),
                )
            else:
                builder.add(key, data=raw)

        return builder.write(self._output)


# ---------------------------------------------------------------------------
# Default mount opener
# ---------------------------------------------------------------------------


def default_zmp_mount_opener(
    entry: ManifestEntry,
    *,
    resolvers: dict[str, Resolver] | None = None,
    mount_opener: ZarrMountOpener | None = None,
    base_resolve: list[dict] | None = None,
    manifest: Manifest | None = None,
) -> Store:
    """Default mount opener: opens .zmp mount targets.

    Supports external mounts (via resolve) and embedded mounts (via data column).
    """
    # Try embedded mount first
    if manifest is not None and (Addressing.DATA in entry.addressing or Addressing.DATA_Z in entry.addressing):
        embedded_bytes = manifest.get_data(entry.path)
        if embedded_bytes is not None:
            import tempfile

            is_zip = (
                entry.content_type in ("application/zip", "application/x-zip-compressed")
                or (embedded_bytes[:4] == b"PK\x03\x04")
            )

            if is_zip:
                from zarr.storage import ZipStore
                tmp = tempfile.NamedTemporaryFile(suffix=".zarr.zip", delete=False)
                tmp.write(embedded_bytes)
                tmp.close()
                zs = ZipStore(tmp.name, mode="r")
                zs._sync_open()
                return zs
            else:
                child_manifest = Manifest(embedded_bytes)
                return ZMPStore(
                    manifest=child_manifest,
                    resolvers=resolvers,
                    mount_opener=mount_opener,
                    base_resolve=base_resolve,
                )

    # External mount via resolve
    if entry.resolve is None:
        raise ValueError(f"Mount entry {entry.path!r} has no resolve or embedded data")

    resolve_dict = json.loads(entry.resolve) if isinstance(entry.resolve, str) else entry.resolve

    http_params = resolve_dict.get("http")
    if http_params and "url" in http_params:
        url = http_params["url"]
        if base_resolve:
            for base in reversed(base_resolve):
                http_base = base.get("http", {})
                if "url" in http_base and "://" not in url and not url.startswith("/"):
                    from urllib.parse import urljoin
                    import os
                    base_url = http_base["url"]
                    if base_url.startswith(("http://", "https://")):
                        url = urljoin(base_url, url)
                    else:
                        url = os.path.normpath(os.path.join(base_url, url))
                    break

        if url.endswith(".zarr.zip"):
            from zarr.storage import ZipStore
            zs = ZipStore(url, mode="r")
            zs._sync_open()
            return zs

        return ZMPStore.from_url(
            url,
            resolvers=resolvers,
            mount_opener=mount_opener,
        )

    raise ValueError(
        f"Cannot resolve mount target for {entry.path!r} — "
        f"resolve={entry.resolve}"
    )


# ---------------------------------------------------------------------------
# Read-only store
# ---------------------------------------------------------------------------


class ZMPStore(Store):
    """Read-only Zarr v3 store backed by a ZMP (Zarr Manifest Parquet) file."""

    supports_writes = False
    supports_deletes = False
    supports_partial_writes = False
    supports_listing = True

    def __init__(
        self,
        manifest: Manifest,
        resolvers: dict[str, Resolver] | None = None,
        mount_opener: ZarrMountOpener | None = None,
        base_resolve: list[dict] | None = None,
    ) -> None:
        super().__init__(read_only=True)
        self._manifest = manifest
        self._resolvers = resolvers if resolvers is not None else {"http": HttpResolver()}
        self._mount_opener = mount_opener or self._default_mount_opener
        file_base = get_file_base_resolve(manifest)
        chain = list(base_resolve or [])
        if file_base is not None:
            chain.append(file_base)
        self._base_resolve = chain or None
        self._mounts: dict[ZPath, Store] = {}
        self._mount_prefixes: list[ZPath] = []
        self._link_prefixes: dict[ZPath, ZPath] = {}
        self._init_mounts()

    def _init_mounts(self) -> None:
        """Discover mount points and directory links from the manifest."""
        mount_prefixes: list[ZPath] = []
        link_prefixes: dict[ZPath, ZPath] = {}
        for path in self._manifest.list_paths():
            entry = self._manifest.get_entry(path)
            if entry is None:
                continue
            if Addressing.FOLDER not in entry.addressing:
                continue
            if path == "":
                continue
            zpath = ZPath(path)
            if Addressing.MOUNT in entry.addressing:
                mount_prefixes.append(zpath)
            elif Addressing.LINK in entry.addressing and entry.resolve:
                resolve_dict = json.loads(entry.resolve) if isinstance(entry.resolve, str) else entry.resolve
                path_params = resolve_dict.get("_path", {})
                target = path_params.get("target")
                if target is not None:
                    link_prefixes[zpath] = ZPath(target)
        # Sort longest-first for proper prefix matching
        self._mount_prefixes = sorted(mount_prefixes, key=lambda p: p.depth, reverse=True)
        self._link_prefixes = dict(
            sorted(link_prefixes.items(), key=lambda x: x[0].depth, reverse=True)
        )

    def _find_mount(self, key: ZPath, *, include_self: bool = False) -> tuple[ZPath, str] | None:
        """Find mount for key. Returns (mount_path, sub_key_zarr) or None.

        If ``include_self`` is True, also matches when key equals the mount path
        (used by list_dir to delegate listing the mount root).
        """
        for mount in self._mount_prefixes:
            if key.is_child_of(mount):
                return mount, key.relative_to(mount)
            if include_self and key == mount:
                return mount, ""
        return None

    def _find_dir_link(self, key: ZPath) -> ZPath | None:
        """Rewrite a key through directory links. Returns the rewritten ZPath or None."""
        for prefix, target in self._link_prefixes.items():
            if key.is_child_of(prefix):
                rel = key.relative_to(prefix)
                return target / rel
        return None

    def _default_mount_opener(self, entry: ManifestEntry) -> Store:
        chain = list(self._base_resolve or [])
        if entry.base_resolve:
            br = json.loads(entry.base_resolve) if isinstance(entry.base_resolve, str) else entry.base_resolve
            chain.append(br)
        return default_zmp_mount_opener(
            entry,
            resolvers=self._resolvers,
            mount_opener=self._mount_opener,
            base_resolve=chain or None,
            manifest=self._manifest,
        )

    def _get_mount_store(self, mount: ZPath) -> Store:
        if mount in self._mounts:
            return self._mounts[mount]
        entry = self._manifest.get_entry(mount.to_zarr())
        if entry is None:
            raise KeyError(f"Mount point {mount!r} not found")
        store = self._mount_opener(entry)
        self._mounts[mount] = store
        return store

    @classmethod
    def from_file(
        cls,
        path: str,
        resolvers: dict[str, Resolver] | None = None,
        mount_opener: ZarrMountOpener | None = None,
    ) -> ZMPStore:
        manifest = Manifest(path)
        location_base = [{"http": {"url": str(Path(path).resolve().parent) + "/"}}]
        return cls(
            manifest=manifest,
            resolvers=resolvers,
            mount_opener=mount_opener,
            base_resolve=location_base,
        )

    @classmethod
    def from_url(
        cls,
        manifest_url: str,
        *,
        resolvers: dict[str, Resolver] | None = None,
        mount_opener: ZarrMountOpener | None = None,
    ) -> ZMPStore:
        """Open a ZMP store from a manifest path/URL."""
        if manifest_url.startswith("http://") or manifest_url.startswith("https://"):
            import httpx
            import tempfile

            location_base_url = manifest_url.rsplit("/", 1)[0] + "/"
            resp = httpx.get(manifest_url)
            resp.raise_for_status()
            tmp = tempfile.NamedTemporaryFile(suffix=".zmp", delete=False)
            tmp.write(resp.content)
            tmp.close()
            manifest = Manifest(tmp.name)
        else:
            location_base_url = str(Path(manifest_url).resolve().parent) + "/"
            manifest = Manifest(manifest_url)

        location_base = [{"http": {"url": location_base_url}}]

        store = cls(
            manifest=manifest,
            resolvers=resolvers,
            mount_opener=mount_opener,
            base_resolve=location_base,
        )
        store._is_open = True
        return store

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ZMPStore):
            return NotImplemented
        return self._manifest is other._manifest

    async def open(self, *args: Any, **kwargs: Any) -> Self:
        self._is_open = True
        return self

    async def close(self) -> None:
        self._is_open = False

    def _is_annotation(self, key: ZPath) -> bool:
        if key.is_root:
            return True
        entry = self._manifest.get_entry(key.to_zarr())
        if entry is not None:
            return Addressing.FOLDER in entry.addressing
        return False

    # -- Zarr Store interface --------------------------------------------------
    # All methods receive bare zarr strings, convert to ZPath internally,
    # and yield bare strings back to zarr.

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        zkey = ZPath.from_zarr(key)

        if self._is_annotation(zkey):
            return None

        # Check mounts
        mount = self._find_mount(zkey)
        if mount is not None:
            mount_path, sub_key = mount
            child = self._get_mount_store(mount_path)
            return await child.get(sub_key, prototype, byte_range)

        # Check directory links
        rewritten = self._find_dir_link(zkey)
        if rewritten is not None:
            return await self.get(rewritten.to_zarr(), prototype, byte_range)

        entry = self._manifest.get_entry(key)
        if entry is None:
            return None

        chain = list(self._base_resolve or [])
        if entry.base_resolve:
            br = json.loads(entry.base_resolve) if isinstance(entry.base_resolve, str) else entry.base_resolve
            chain.append(br)

        raw = await resolve_entry(
            entry, self._manifest, self._resolvers, chain or None,
        )
        if raw is None:
            return None

        raw = _apply_byte_range(raw, byte_range)
        return prototype.buffer.from_bytes(raw)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        import asyncio
        coros = [
            self.get(key, prototype, byte_range)
            for key, byte_range in key_ranges
        ]
        return list(await asyncio.gather(*coros))

    async def exists(self, key: str) -> bool:
        zkey = ZPath.from_zarr(key)
        if self._is_annotation(zkey):
            return False
        mount = self._find_mount(zkey)
        if mount is not None:
            mount_path, sub_key = mount
            child = self._get_mount_store(mount_path)
            return await child.exists(sub_key)
        rewritten = self._find_dir_link(zkey)
        if rewritten is not None:
            return await self.exists(rewritten.to_zarr())
        return self._manifest.has(key)

    async def set(self, key: str, value: Buffer) -> None:
        raise NotImplementedError("ZMPStore is read-only")

    async def delete(self, key: str) -> None:
        raise NotImplementedError("ZMPStore is read-only")

    async def list(self) -> AsyncIterator[str]:
        for p in self._manifest.list_paths():
            if p == "":
                continue
            zp = ZPath(p)
            if self._is_annotation(zp):
                continue
            if self._find_mount(zp) is not None:
                continue
            if self._find_dir_link(zp) is not None:
                continue
            yield zp.to_zarr()
        # Entries from mounts
        for mount in self._mount_prefixes:
            child = self._get_mount_store(mount)
            async for p in child.list():
                yield (mount / p).to_zarr()
        # Entries through directory links
        for link_prefix, target in self._link_prefixes.items():
            for p in self._manifest.list_prefix(str(target)):
                if p == "":
                    continue
                zp = ZPath(p)
                if self._is_annotation(zp):
                    continue
                rel = zp.relative_to(target)
                yield (link_prefix / rel).to_zarr()

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        zprefix = ZPath.from_zarr(prefix)

        # Check mounts
        mount = self._find_mount(zprefix)
        if mount is not None:
            mount_path, sub_prefix = mount
            child = self._get_mount_store(mount_path)
            async for p in child.list_prefix(sub_prefix):
                yield (mount_path / p).to_zarr()
            return

        # Check directory links
        rewritten = self._find_dir_link(zprefix)
        if rewritten is not None:
            async for p in self.list_prefix(rewritten.to_zarr()):
                zp = ZPath.from_zarr(p)
                # Re-map back through link
                for lp, target in self._link_prefixes.items():
                    if zp.is_equal_or_child_of(target):
                        rel = zp.relative_to(target)
                        yield (lp / rel).to_zarr()
                        break
            return

        # Local entries
        for p in self._manifest.list_prefix(str(zprefix)):
            if p == "":
                continue
            zp = ZPath(p)
            if self._is_annotation(zp) or self._find_mount(zp) is not None:
                continue
            yield zp.to_zarr()

        # Mounts under this prefix
        for mount in self._mount_prefixes:
            if mount.is_equal_or_child_of(zprefix):
                child = self._get_mount_store(mount)
                async for p in child.list():
                    yield (mount / p).to_zarr()

        # Directory links under this prefix
        for link_prefix, target in self._link_prefixes.items():
            if link_prefix.is_equal_or_child_of(zprefix):
                for p in self._manifest.list_prefix(str(target)):
                    if p == "":
                        continue
                    zp = ZPath(p)
                    if self._is_annotation(zp):
                        continue
                    rel = zp.relative_to(target)
                    yield (link_prefix / rel).to_zarr()

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        zprefix = ZPath.from_zarr(prefix)

        # If inside or at a mount, delegate
        mount = self._find_mount(zprefix, include_self=True)
        if mount is not None:
            mount_path, sub_prefix = mount
            child = self._get_mount_store(mount_path)
            async for p in child.list_dir(sub_prefix):
                yield p
            return

        # If inside a directory link, rewrite and list
        rewritten = self._find_dir_link(zprefix)
        if rewritten is not None:
            async for p in self.list_dir(rewritten.to_zarr()):
                yield p
            return

        # Local entries
        seen: set[str] = set()
        for p in self._manifest.list_dir(str(zprefix)):
            if p != "":
                seen.add(p)
                yield p

        # Mount points and directory links as directory entries
        for vdir in (*self._mount_prefixes, *self._link_prefixes):
            child_name = vdir.child_name_under(zprefix)
            if child_name is not None:
                # zarr list_dir convention: directories have trailing /
                dir_entry = child_name + "/"
                if dir_entry not in seen:
                    seen.add(dir_entry)
                    yield dir_entry
