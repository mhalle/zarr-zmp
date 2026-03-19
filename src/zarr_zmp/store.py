from __future__ import annotations

import os
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
from zmanifest.builder import Builder, git_blob_hash
from zmanifest.manifest import Manifest, ManifestEntry
from zmanifest.resolve import fetch_uri, is_relative_uri, resolve_entry, resolve_uri
from zmanifest.resolver import BlobResolver, GitResolver, TemplateResolver

# Type alias for mount opener callbacks (zarr Store-typed version)
ZarrMountOpener = Callable[[ManifestEntry], Store]

# Back-compat alias
MountOpener = ZarrMountOpener


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


def _is_zarr_metadata(path: str) -> bool:
    """Check if a path is a zarr metadata file."""
    basename = path.rsplit("/", 1)[-1] if "/" in path else path
    return basename in ("zarr.json", ".zarray", ".zgroup", ".zattrs", ".zmetadata")


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
      references them via ``retrieval_key``.

    In both modes, JSON content (zarr.json etc.) is canonicalized via
    RFC 8785 before hashing so the git-sha1 is deterministic regardless
    of key ordering.

    Use as a context manager or call :meth:`commit` explicitly::

        with ZMPWritableStore.create("output.zmp") as store:
            root = zarr.open_group(store=store, mode="w")
            root.create_array("data", data=arr)
        # .zmp file written on exit

    Args:
        output: Path for the output ``.zmp`` parquet file.
        chunk_dir: If set, write chunk blobs to this directory instead
            of inlining them. Directory is created if needed.
        max_rows_per_group: Override adaptive row group sizing.
        data_compression: Parquet compression for the ``data`` column.
            Default ``"none"`` (chunks are already compressed by zarr).
            Use ``"zstd"`` or ``"snappy"`` for uncompressed zarr data.
        zarr_format: Zarr format version for file-level metadata.
        metadata: Additional key-value pairs for file-level metadata.
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
        data_compression: str = "none",
        data_compression_level: int | None = None,
        zarr_format: str = "3",
        metadata: dict[str, object] | None = None,
    ) -> None:
        super().__init__(read_only=False)
        self._output = Path(output)
        self._chunk_dir = Path(chunk_dir) if chunk_dir is not None else None
        self._max_rows_per_group = max_rows_per_group
        self._data_compression = data_compression
        self._data_compression_level = data_compression_level
        self._zarr_format = zarr_format
        self._metadata = metadata or {}
        self._entries: dict[str, bytes] = {}
        self._is_open = True

    @classmethod
    def create(
        cls,
        output: str | Path,
        *,
        chunk_dir: str | Path | None = None,
        max_rows_per_group: int | None = None,
        data_compression: str = "none",
        data_compression_level: int | None = None,
        zarr_format: str = "3",
        metadata: dict[str, object] | None = None,
    ) -> ZMPWritableStore:
        return cls(
            output,
            chunk_dir=chunk_dir,
            max_rows_per_group=max_rows_per_group,
            data_compression=data_compression,
            data_compression_level=data_compression_level,
            zarr_format=zarr_format,
            metadata=metadata,
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
        for p in sorted(self._entries):
            if p.startswith(prefix):
                yield p

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"
        result: set[str] = set()
        for p in self._entries:
            if not p.startswith(prefix):
                continue
            rest = p[len(prefix) :]
            slash_idx = rest.find("/")
            if slash_idx == -1:
                result.add(rest)
            else:
                result.add(rest[: slash_idx + 1])
        for item in sorted(result):
            yield item

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
            data_compression=self._data_compression,
            data_compression_level=self._data_compression_level,
            max_rows_per_group=self._max_rows_per_group,
            metadata=self._metadata,
        )

        for path in sorted(self._entries):
            raw = self._entries[path]
            is_meta = _is_zarr_metadata(path)

            if is_meta:
                # Zarr metadata: inline as text, Builder handles hashing
                builder.add(path, text=raw.decode("utf-8"))
            elif external:
                # External mode: write blob to chunk_dir, store retrieval_key only
                rk = git_blob_hash(raw)
                blob_path = self._chunk_dir / rk
                if not blob_path.exists():
                    blob_path.write_bytes(raw)
                builder.add(path, retrieval_key=rk, size=len(raw))
            else:
                # Embedded mode: inline binary data
                builder.add(path, data=raw)

        return builder.write(self._output)


# ---------------------------------------------------------------------------
# Default mount opener (standalone, so users can call it as a fallback)
# ---------------------------------------------------------------------------


def default_zmp_mount_opener(
    entry: ManifestEntry,
    *,
    resolver: BlobResolver | None = None,
    mount_opener: ZarrMountOpener | None = None,
    base_uri: str | None = None,
) -> Store:
    """Default mount opener: opens .zmp and .zarr.zip targets.

    Users can call this as a fallback inside a custom mount_opener to
    handle standard mount types while adding their own logic for others.

    Args:
        entry: The manifest entry for the mount point.
        resolver: BlobResolver to pass to child .zmp stores.
        mount_opener: MountOpener to pass to child .zmp stores.
        base_uri: Base URI for resolving relative uri values.
    """
    uri = entry.uri
    if uri is None:
        raise ValueError(f"Mount entry {entry.path!r} has no uri")

    # Resolve relative URIs against the parent's base
    uri = resolve_uri(uri, base_uri)

    if uri.endswith(".zmp"):
        return ZMPStore.from_url(
            uri, resolver=resolver, mount_opener=mount_opener,
        )
    elif uri.endswith(".zarr.zip"):
        from zarr.storage import ZipStore

        store = ZipStore(uri, mode="r")
        store._sync_open()
        return store
    else:
        raise ValueError(
            f"Unsupported mount target {uri!r} — "
            "only .zmp and .zarr.zip are supported"
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
        resolver: BlobResolver | None = None,
        mount_opener: ZarrMountOpener | None = None,
        base_uri: str | None = None,
    ) -> None:
        super().__init__(read_only=True)
        self._manifest = manifest
        self._resolver = resolver
        self._mount_opener = mount_opener or self._default_mount_opener
        # base_uri precedence: API override > file metadata > None
        if base_uri is not None:
            self._base_uri: str | None = base_uri
        else:
            extra = self._manifest.metadata.get("extra", {})
            self._base_uri = extra.get("base_uri") if extra else None
        self._mounts: dict[str, Store] = {}  # prefix -> lazily opened store
        self._mount_prefixes: list[str] = []  # sorted longest-first
        self._init_mounts()

    def _init_mounts(self) -> None:
        """Discover mount points from the manifest."""
        prefixes = []
        for path in self._manifest.list_paths():
            if not path.endswith("/"):
                continue
            entry = self._manifest.get_entry(path)
            if entry is not None and Addressing.MOUNT in entry.addressing:
                prefixes.append(path)
        # Sort longest first so deeper mounts match before shallower ones
        self._mount_prefixes = sorted(prefixes, key=len, reverse=True)

    def _find_mount(self, key: str) -> tuple[str, str] | None:
        """Find the mount prefix for a key, if any.

        Returns (mount_prefix, sub_key) or None.
        """
        for prefix in self._mount_prefixes:
            if key.startswith(prefix):
                return prefix, key[len(prefix) :]
        return None

    def _default_mount_opener(self, entry: ManifestEntry) -> Store:
        """Default mount opener that inherits resolver, mount_opener, and base_uri."""
        return default_zmp_mount_opener(
            entry,
            resolver=self._resolver,
            mount_opener=self._mount_opener,
            base_uri=entry.base_uri or self._base_uri,
        )

    def _get_mount_store(self, prefix: str) -> Store:
        """Open (or return cached) the store for a mount point."""
        if prefix in self._mounts:
            return self._mounts[prefix]

        entry = self._manifest.get_entry(prefix)
        if entry is None:
            raise KeyError(f"Mount point {prefix!r} not found")

        store = self._mount_opener(entry)
        self._mounts[prefix] = store
        return store

    @classmethod
    def from_file(
        cls,
        path: str,
        resolver: BlobResolver | None = None,
        mount_opener: ZarrMountOpener | None = None,
        base_uri: str | None = None,
    ) -> ZMPStore:
        from zmanifest.resolve import base_uri_from_source

        manifest = Manifest(path)
        if base_uri is None:
            base_uri = base_uri_from_source(path)
        return cls(
            manifest=manifest, resolver=resolver,
            mount_opener=mount_opener, base_uri=base_uri,
        )

    @classmethod
    def from_url(
        cls,
        manifest_url: str,
        blobs: str | None = None,
        *,
        resolver: BlobResolver | None = None,
        mount_opener: ZarrMountOpener | None = None,
        base_uri: str | None = None,
    ) -> ZMPStore:
        """Open a ZMP store from a manifest path/URL and optional blob location.

        The manifest is fetched if it's an HTTP(S) URL, otherwise read from
        a local path.

        The ``blobs`` argument is a URL template with ``{hash}`` as the
        placeholder for the retrieval key. Supports slice syntax for
        fanout conventions like ``{hash[:2]}/{hash[2:]}``.

        - Ends with ``.git``: uses ``GitResolver`` (vost/dulwich)
        - Contains ``{hash}``: uses ``TemplateResolver`` (local or HTTP)
        - Plain path/URL without ``{hash}``: appends ``/{hash}`` automatically
        - ``None``: no resolver (inline-only manifest)

        If ``resolver`` is provided directly, it takes precedence over
        ``blobs`` (the ``blobs`` argument is ignored).

        Relative ``uri`` values in the manifest are resolved
        against ``base_uri``. If not provided, ``base_uri`` falls back
        to the ``base_uri`` file-level metadata key, then to the parent
        directory of ``manifest_url``.

        Examples::

            ZMPStore.from_url("manifest.zmp")
            ZMPStore.from_url("manifest.zmp", blobs="/data/repo.git")
            ZMPStore.from_url("manifest.zmp", blobs="/data/blobs/{hash}")
            ZMPStore.from_url("manifest.zmp", blobs="https://cdn.example.com/{hash[:2]}/{hash[2:]}")
            ZMPStore.from_url("https://server.com/ds.zmp", blobs="https://blobs.server.com/{hash}")

        Args:
            manifest_url: Local path or HTTP(S) URL to the ``.zmp`` file.
            blobs: URL template, git repo path, or base path for blob resolution.
            resolver: Pre-built BlobResolver instance (overrides ``blobs``).
            mount_opener: Custom callable to open child stores for mount entries.
            base_uri: Base URI for resolving relative uri values.
                Overrides file-level metadata and manifest URL derivation.
        """
        from zmanifest.resolve import base_uri_from_source

        # Fetch manifest if remote
        if manifest_url.startswith("http://") or manifest_url.startswith("https://"):
            import httpx
            import tempfile

            resp = httpx.get(manifest_url)
            resp.raise_for_status()
            tmp = tempfile.NamedTemporaryFile(suffix=".zmp", delete=False)
            tmp.write(resp.content)
            tmp.close()
            manifest = Manifest(tmp.name)
        else:
            manifest = Manifest(manifest_url)

        # Use provided resolver, or auto-create from blobs location
        if resolver is None and blobs is not None:
            if blobs.rstrip("/").endswith(".git"):
                resolver = GitResolver(blobs)
            else:
                # If no {hash} placeholder, append /{hash} as default
                if "{hash" not in blobs:
                    blobs = blobs.rstrip("/") + "/{hash}"
                resolver = TemplateResolver(blobs)

        # Derive base_uri: API override > file metadata > manifest URL
        if base_uri is None:
            base_uri = base_uri_from_source(manifest_url)

        store = cls(
            manifest=manifest, resolver=resolver,
            mount_opener=mount_opener, base_uri=base_uri,
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

    @staticmethod
    def _is_annotation(path: str) -> bool:
        """Annotation rows: root ("") and path metadata ("group/")."""
        return path == "" or path.endswith("/")

    def _is_under_mount(self, path: str) -> bool:
        """Check if a path falls under any mount prefix."""
        return self._find_mount(path) is not None

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        if self._is_annotation(key):
            return None

        # Check mounts first
        mount = self._find_mount(key)
        if mount is not None:
            prefix, sub_key = mount
            child = self._get_mount_store(prefix)
            return await child.get(sub_key, prototype, byte_range)

        entry = self._manifest.get_entry(key)
        if entry is None:
            return None

        base = entry.base_uri or self._base_uri
        raw = await resolve_entry(entry, self._manifest, self._resolver, base)
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
        if self._is_annotation(key):
            return False

        mount = self._find_mount(key)
        if mount is not None:
            prefix, sub_key = mount
            child = self._get_mount_store(prefix)
            return await child.exists(sub_key)

        return self._manifest.has(key)

    async def set(self, key: str, value: Buffer) -> None:
        raise NotImplementedError("ZMPStore is read-only")

    async def delete(self, key: str) -> None:
        raise NotImplementedError("ZMPStore is read-only")

    async def list(self) -> AsyncIterator[str]:
        # Yield local entries (skip annotations and anything under mounts)
        for p in self._manifest.list_paths():
            if self._is_annotation(p) or self._is_under_mount(p):
                continue
            yield p
        # Yield entries from each mount, prepending the prefix
        for prefix in self._mount_prefixes:
            child = self._get_mount_store(prefix)
            async for p in child.list():
                yield prefix + p

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        # Check if the prefix falls entirely within a mount
        mount = self._find_mount(prefix)
        if mount is not None:
            mount_prefix, sub_prefix = mount
            child = self._get_mount_store(mount_prefix)
            async for p in child.list_prefix(sub_prefix):
                yield mount_prefix + p
            return

        # Yield local entries under this prefix
        for p in self._manifest.list_prefix(prefix):
            if self._is_annotation(p) or self._is_under_mount(p):
                continue
            yield p

        # Yield from any mounts that are under this prefix
        for mount_prefix in self._mount_prefixes:
            if mount_prefix.startswith(prefix):
                child = self._get_mount_store(mount_prefix)
                async for p in child.list():
                    yield mount_prefix + p

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        # Normalize prefix
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"

        # If prefix is inside a mount, delegate entirely
        mount = self._find_mount(prefix)
        if mount is not None:
            mount_prefix, sub_prefix = mount
            child = self._get_mount_store(mount_prefix)
            async for p in child.list_dir(sub_prefix):
                yield p
            return

        # Yield local directory entries
        seen: set[str] = set()
        for p in self._manifest.list_dir(prefix):
            if p != "":
                seen.add(p)
                yield p

        # Add mount points as directory entries if they're direct
        # children of this prefix
        for mp in self._mount_prefixes:
            if mp.startswith(prefix) and mp != prefix:
                rest = mp[len(prefix) :]
                # Direct child mount: "sub/" under prefix ""
                slash_idx = rest.find("/")
                if slash_idx == len(rest) - 1:
                    # This mount is a direct child directory
                    if rest not in seen:
                        yield rest
