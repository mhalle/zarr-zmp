"""Build ZMP parquet manifests from zarr stores."""

from __future__ import annotations

import asyncio
from pathlib import Path

from zarr.abc.store import Store
from zarr.core.buffer import default_buffer_prototype

from zmanifest.builder import Builder, canonical_json, git_blob_hash


def _parse_array_path_and_chunk_key(
    path: str,
) -> tuple[str | None, str | None]:
    """Extract array_path and chunk_key from a zarr v3 path with a ``c/`` separator."""
    parts = path.split("/")
    try:
        c_idx = parts.index("c")
        array_path = "/".join(parts[:c_idx]) or None
        chunk_key = "/".join(parts[c_idx + 1 :]) or None
        return array_path, chunk_key
    except ValueError:
        return None, None


async def _read_store(store: Store) -> dict[str, bytes]:
    """Read all keys and values from a zarr store."""
    proto = default_buffer_prototype()
    entries: dict[str, bytes] = {}
    async for key in store.list():
        buf = await store.get(key, proto)
        if buf is not None:
            entries[key] = buf.to_bytes()
    return entries


def _is_zarr_metadata(path: str) -> bool:
    """Check if a path is a zarr metadata file."""
    basename = path.rsplit("/", 1)[-1] if "/" in path else path
    return basename in ("zarr.json", ".zarray", ".zgroup", ".zattrs", ".zmetadata")


def build_zmp(
    store: Store,
    output: str | Path,
    *,
    zarr_format: str = "3",
    metadata: dict[str, object] | None = None,
) -> Path:
    """Build a self-contained ZMP parquet file from a zarr store.

    All content is inlined: metadata as ``text``, chunks as ``data``
    (uncompressed parquet column, since zarr chunks are pre-compressed).

    Args:
        store: Any zarr store to read from (MemoryStore, LocalStore, etc.).
        output: Path for the output ``.zmp`` parquet file.
        zarr_format: Zarr format version (``"2"`` or ``"3"``).
        metadata: Additional key-value pairs for file-level metadata.

    Returns:
        Path to the written ZMP file.
    """
    entries = asyncio.run(_read_store(store))
    return _build_zmp_from_entries(
        entries, output, zarr_format=zarr_format, metadata=metadata,
    )


async def async_build_zmp(
    store: Store,
    output: str | Path,
    *,
    zarr_format: str = "3",
    metadata: dict[str, object] | None = None,
) -> Path:
    """Async version of :func:`build_zmp`."""
    entries = await _read_store(store)
    return _build_zmp_from_entries(
        entries, output, zarr_format=zarr_format, metadata=metadata,
    )


def _build_zmp_from_entries(
    entries: dict[str, bytes],
    output: str | Path,
    *,
    zarr_format: str,
    metadata: dict[str, object] | None,
) -> Path:
    builder = Builder(
        zarr_format=zarr_format,
        metadata=metadata,
    )

    for path in sorted(entries):
        raw = entries[path]
        if _is_zarr_metadata(path):
            text = canonical_json(raw.decode("utf-8"))
            builder.add(path, text=text)
        else:
            # Zarr chunks are pre-compressed -> data column (no parquet compression)
            builder.add(path, data=raw)

    return builder.write(output)
