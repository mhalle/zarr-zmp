"""Build ZMP parquet manifests from zarr stores."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from zarr.abc.store import Store
from zarr.core.buffer import default_buffer_prototype

from zmanifest._types import compute_addressing
from zmanifest.builder import _canonical_hash, _git_blob_hash, _parse_array_path_and_chunk_key


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


def _classify_entry(path: str, data: bytes) -> tuple[str | None, bytes | None]:
    """Classify a store entry and return (text, binary_data).

    Only zarr metadata files are auto-inlined as text.
    Everything else is binary data.
    """
    if _is_zarr_metadata(path):
        return data.decode("utf-8"), None
    return None, data


def build_zmp(
    store: Store,
    output: str | Path,
    *,
    zarr_format: str = "3",
    retrieval_scheme: str = "git-sha1",
    data_compression: str = "none",
    data_compression_level: int | None = None,
    metadata: dict[str, object] | None = None,
) -> Path:
    """Build a self-contained ZMP parquet file from a zarr store.

    All content is inlined: metadata as ``text``, chunks as ``data``.

    Args:
        store: Any zarr store to read from (MemoryStore, LocalStore, etc.).
        output: Path for the output ``.zmp`` parquet file.
        zarr_format: Zarr format version (``"2"`` or ``"3"``).
        retrieval_scheme: Retrieval scheme for the manifest metadata.
        data_compression: Parquet compression for the ``data`` column.
            Default ``"none"`` (chunks are already compressed by zarr).
        data_compression_level: Compression level for the ``data`` column.
        metadata: Additional key-value pairs for file-level metadata.

    Returns:
        Path to the written ZMP file.
    """
    entries = asyncio.run(_read_store(store))
    return _build_zmp_from_entries(
        entries,
        output,
        zarr_format=zarr_format,
        retrieval_scheme=retrieval_scheme,
        data_compression=data_compression,
        data_compression_level=data_compression_level,
        metadata=metadata,
    )


async def async_build_zmp(
    store: Store,
    output: str | Path,
    *,
    zarr_format: str = "3",
    retrieval_scheme: str = "git-sha1",
    data_compression: str = "none",
    data_compression_level: int | None = None,
    metadata: dict[str, object] | None = None,
) -> Path:
    """Async version of :func:`build_zmp`."""
    entries = await _read_store(store)
    return _build_zmp_from_entries(
        entries,
        output,
        zarr_format=zarr_format,
        retrieval_scheme=retrieval_scheme,
        data_compression=data_compression,
        data_compression_level=data_compression_level,
        metadata=metadata,
    )


def _build_zmp_from_entries(
    entries: dict[str, bytes],
    output: str | Path,
    *,
    zarr_format: str,
    retrieval_scheme: str,
    data_compression: str,
    data_compression_level: int | None,
    metadata: dict[str, object] | None,
) -> Path:
    output = Path(output)

    paths: list[str] = []
    sizes: list[int] = []
    texts: list[str | None] = []
    data_blobs: list[bytes | None] = []
    retrieval_keys: list[str | None] = []
    array_paths: list[str | None] = []
    chunk_keys: list[str | None] = []
    addressing_lists: list[list[str]] = []

    for path in sorted(entries):
        raw = entries[path]
        text, binary = _classify_entry(path, raw)
        array_path, chunk_key = _parse_array_path_and_chunk_key(path)
        rk = _git_blob_hash(raw)

        paths.append(path)
        sizes.append(len(raw))
        texts.append(text)
        data_blobs.append(binary)
        retrieval_keys.append(rk)
        array_paths.append(array_path)
        chunk_keys.append(chunk_key)

        addressing_lists.append(compute_addressing(
            text=text, data=binary, retrieval_key=rk,
        ))

    table = pa.table(
        {
            "path": pa.array(paths, type=pa.string()),
            "size": pa.array(sizes, type=pa.int64()),
            "retrieval_key": pa.array(retrieval_keys, type=pa.string()),
            "text": pa.array(texts, type=pa.string()),
            "data": pa.array(data_blobs, type=pa.binary()),
            "array_path": pa.array(array_paths, type=pa.string()),
            "chunk_key": pa.array(chunk_keys, type=pa.string()),
            "addressing": pa.array(addressing_lists, type=pa.list_(pa.string())),
        }
    )

    # File-level metadata
    file_meta: dict[bytes, bytes] = {
        b"zmp_version": json.dumps("0.1.0").encode(),
        b"zarr_format": json.dumps(zarr_format).encode(),
        b"retrieval_scheme": json.dumps(retrieval_scheme).encode(),
    }
    if metadata:
        for k, v in metadata.items():
            file_meta[k.encode()] = v.encode() if isinstance(v, str) else json.dumps(v).encode()

    schema = table.schema.with_metadata(file_meta)
    table = table.cast(schema)

    # Per ZMP spec: data column SHOULD be uncompressed (already compressed
    # by zarr's codec pipeline), all other columns use zstd.
    compression = {col: "zstd" for col in table.schema.names}
    compression["data"] = data_compression
    use_dictionary = {col: True for col in table.schema.names}
    use_dictionary["data"] = False

    # Adaptive row group sizing:
    # - With inline data: 2 rows/group (data dominates overhead, small
    #   groups enable efficient lazy access)
    # - Reference-only (no inline data): ~16 groups total (keeps DuckDB
    #   analytics fast, minimal per-group overhead)
    has_inline_data = any(v is not None for v in data_blobs)
    if has_inline_data:
        max_rows_per_group = 2
    else:
        max_rows_per_group = max(1, math.ceil(len(table) / 16))

    compression_level = None
    if data_compression_level is not None:
        compression_level = {"data": data_compression_level}

    writer = pq.ParquetWriter(
        str(output),
        schema,
        compression=compression,
        compression_level=compression_level,
        use_dictionary=use_dictionary,
    )
    try:
        n = len(table)
        i = 0
        while i < n:
            end = min(i + max_rows_per_group, n)
            writer.write_table(table.slice(i, end - i))
            i = end
    finally:
        writer.close()

    return output
