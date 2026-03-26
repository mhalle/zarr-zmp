"""Read-only Zarr v3 store that wraps a single NRRD file.

For raw (uncompressed) NRRD files, the data is served as byte-range
chunks along the slowest-varying axis. For compressed NRRD files
(gzip, bzip2, zlib), the entire data section is one chunk and zarr's
codec pipeline handles decompression.

Usage::

    from zarr_zmp.nrrd_store import NRRDStore
    store = NRRDStore("scan.nrrd")
    arr = zarr.open_array(store=store, mode="r")
    slice = arr[64, :, :]
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import Any, Self

import numpy as np
from zarr.abc.store import ByteRequest, OffsetByteRequest, RangeByteRequest, Store, SuffixByteRequest
from zarr.core.buffer import Buffer, BufferPrototype


# ---------------------------------------------------------------------------
# NRRD header parsing
# ---------------------------------------------------------------------------

# NRRD type → numpy dtype
_NRRD_DTYPES: dict[str, str] = {
    "signed char": "int8",
    "int8": "int8",
    "int8_t": "int8",
    "uchar": "uint8",
    "unsigned char": "uint8",
    "uint8": "uint8",
    "uint8_t": "uint8",
    "short": "int16",
    "short int": "int16",
    "signed short": "int16",
    "signed short int": "int16",
    "int16": "int16",
    "int16_t": "int16",
    "ushort": "uint16",
    "unsigned short": "uint16",
    "unsigned short int": "uint16",
    "uint16": "uint16",
    "uint16_t": "uint16",
    "int": "int32",
    "signed int": "int32",
    "int32": "int32",
    "int32_t": "int32",
    "uint": "uint32",
    "unsigned int": "uint32",
    "uint32": "uint32",
    "uint32_t": "uint32",
    "longlong": "int64",
    "long long": "int64",
    "long long int": "int64",
    "signed long long": "int64",
    "signed long long int": "int64",
    "int64": "int64",
    "int64_t": "int64",
    "ulonglong": "uint64",
    "unsigned long long": "uint64",
    "unsigned long long int": "uint64",
    "uint64": "uint64",
    "uint64_t": "uint64",
    "float": "float32",
    "float32": "float32",
    "double": "float64",
    "float64": "float64",
    "block": "uint8",
}

# NRRD encoding → zarr codec name (or None for raw)
_ENCODING_CODECS: dict[str, str | None] = {
    "raw": None,
    "gzip": "gzip",
    "gz": "gzip",
    "bzip2": "numcodecs.bz2",
    "bz2": "numcodecs.bz2",
    "zlib": "numcodecs.zlib",
}

# NRRD endian → numpy byte order
_ENDIAN_MAP = {
    "little": "<",
    "big": ">",
}


def _parse_nrrd_header(path: str | Path) -> dict[str, Any]:
    """Parse an NRRD header and return metadata + data offset."""
    path = Path(path)

    with open(path, "rb") as f:
        # First line must be "NRRD000X"
        magic = f.readline().decode("ascii", errors="replace").strip()
        if not magic.startswith("NRRD"):
            raise ValueError(f"Not an NRRD file: {magic!r}")

        fields: dict[str, str] = {}
        while True:
            line = f.readline().decode("ascii", errors="replace")
            if not line or line.strip() == "":
                break  # blank line = end of header
            line = line.strip()
            if line.startswith("#"):
                continue  # comment
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip()
            # Key-value pair (with :=) is field info, with : is data
            if _:
                fields[key] = value

        data_offset = f.tell()

    # Parse required fields
    if "type" not in fields:
        raise ValueError("NRRD missing 'type' field")

    nrrd_type = fields["type"].lower()
    dtype_name = _NRRD_DTYPES.get(nrrd_type)
    if dtype_name is None:
        raise ValueError(f"Unsupported NRRD type: {nrrd_type}")

    if "sizes" not in fields:
        raise ValueError("NRRD missing 'sizes' field")
    sizes = [int(s) for s in fields["sizes"].split()]

    encoding = fields.get("encoding", "raw").lower()
    if encoding not in _ENCODING_CODECS:
        raise ValueError(f"Unsupported NRRD encoding: {encoding}")

    endian_str = fields.get("endian", "little").lower()
    byte_order = _ENDIAN_MAP.get(endian_str, "<")

    dtype = np.dtype(dtype_name).newbyteorder(byte_order)

    # Parse optional spatial fields for nrrd attributes
    nrrd_attrs: dict[str, Any] = {}
    for key in fields:
        nrrd_attrs[key] = fields[key]

    # Parse space directions if present
    if "space directions" in fields:
        try:
            dirs_str = fields["space directions"]
            dirs = []
            for part in dirs_str.split(")"):
                part = part.strip().lstrip(",").strip()
                if not part:
                    continue
                if part.lower() == "none":
                    dirs.append(None)
                    continue
                part = part.lstrip("(").rstrip(")")
                vals = [float(v) for v in part.split(",")]
                dirs.append(vals)
            nrrd_attrs["space directions"] = dirs
        except (ValueError, IndexError):
            pass

    # Parse space origin if present
    if "space origin" in fields:
        try:
            origin_str = fields["space origin"].strip("()")
            nrrd_attrs["space origin"] = [float(v) for v in origin_str.split(",")]
        except ValueError:
            pass

    # Parse kinds if present
    if "kinds" in fields:
        nrrd_attrs["kinds"] = fields["kinds"].split()

    return {
        "sizes": sizes,
        "dtype": dtype,
        "dtype_name": dtype_name,
        "encoding": encoding,
        "data_offset": data_offset,
        "byte_order": byte_order,
        "nrrd_attrs": nrrd_attrs,
        "file_path": str(path.resolve()),
    }


# ---------------------------------------------------------------------------
# NRRDStore
# ---------------------------------------------------------------------------


def _apply_byte_range(data: bytes, byte_range: ByteRequest | None) -> bytes:
    if byte_range is None:
        return data
    if isinstance(byte_range, RangeByteRequest):
        return data[byte_range.start:byte_range.end]
    if isinstance(byte_range, OffsetByteRequest):
        return data[byte_range.offset:]
    if isinstance(byte_range, SuffixByteRequest):
        return data[-byte_range.suffix:]
    return data


class NRRDStore(Store):
    """Read-only Zarr v3 store wrapping a single NRRD file.

    For raw encoding, chunks along the slowest axis enable efficient
    random access. For compressed encoding, one chunk covers the
    entire array and zarr decompresses via its codec pipeline.

    Args:
        path: Path to the NRRD file.
        chunks_per_axis: For raw encoding, number of chunks along the
            slowest axis. Default: one slice per chunk.
    """

    supports_writes = False
    supports_deletes = False
    supports_partial_writes = False
    supports_listing = True

    def __init__(
        self,
        path: str | Path,
        *,
        chunks_per_axis: int | None = None,
    ) -> None:
        super().__init__(read_only=True)
        self._header = _parse_nrrd_header(path)
        self._file_path = self._header["file_path"]
        self._is_raw = self._header["encoding"] == "raw"
        self._is_open = True

        sizes = self._header["sizes"]
        dtype = self._header["dtype"]

        # NRRD sizes are fastest-varying first. The raw bytes on disk
        # are Fortran-ordered. Present to zarr as C-order by reversing
        # the axes — same bytes, no reordering needed.
        self._shape = list(reversed(sizes))

        if self._is_raw:
            # Chunk along the slowest axis (first in C order)
            if chunks_per_axis is not None:
                slices = chunks_per_axis
            else:
                slices = self._shape[0]  # one slice per chunk

            chunk_dim0 = max(1, math.ceil(self._shape[0] / slices))
            self._chunk_shape = [chunk_dim0] + self._shape[1:]
        else:
            # Compressed: one chunk = entire array
            self._chunk_shape = list(self._shape)

        self._dtype = dtype
        self._zarr_json = self._build_zarr_json()
        self._keys = self._build_key_set()

    def _build_zarr_json(self) -> str:
        dtype_name = self._header["dtype_name"]
        endian = "little" if self._header["byte_order"] == "<" else "big"

        # No transpose needed — we reversed the shape so the C-order
        # interpretation of the raw bytes matches the NRRD layout.
        codecs: list[dict[str, Any]] = [
            {"name": "bytes", "configuration": {"endian": endian}},
        ]

        encoding = self._header["encoding"]
        codec_name = _ENCODING_CODECS.get(encoding)
        if codec_name is not None:
            if codec_name == "gzip":
                codecs.append({"name": "gzip", "configuration": {"level": 6}})
            else:
                codecs.append({"name": codec_name, "configuration": {}})

        meta: dict[str, Any] = {
            "zarr_format": 3,
            "node_type": "array",
            "shape": self._shape,
            "data_type": dtype_name,
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": self._chunk_shape},
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "/"},
            },
            "fill_value": 0,
            "codecs": codecs,
            "attributes": {
                "nrrd": self._header["nrrd_attrs"],
            },
        }
        return json.dumps(meta, separators=(",", ":"))

    def _build_key_set(self) -> set[str]:
        """Pre-compute all valid chunk keys."""
        keys = {"zarr.json"}
        ndim = len(self._shape)
        n_chunks = [
            math.ceil(self._shape[i] / self._chunk_shape[i])
            for i in range(ndim)
        ]

        def _recurse(dim: int, prefix: str) -> None:
            if dim == ndim:
                keys.add(f"c/{prefix}")
                return
            for idx in range(n_chunks[dim]):
                next_prefix = f"{prefix}/{idx}" if prefix else str(idx)
                _recurse(dim + 1, next_prefix)

        _recurse(0, "")
        return keys

    def _chunk_key_to_indices(self, key: str) -> list[int]:
        """Parse 'c/0/0/0' → [0, 0, 0]."""
        parts = key.split("/")
        if parts[0] != "c":
            raise KeyError(key)
        return [int(p) for p in parts[1:]]

    def _read_raw_chunk(self, indices: list[int]) -> bytes:
        """Read a chunk from the raw data section by computing its byte offset.

        NRRD raw data is stored in C order (matching the sizes field).
        Chunks along the first axis are contiguous slabs.
        """
        dtype = self._dtype
        shape = self._shape
        chunk_shape = self._chunk_shape
        data_offset = self._header["data_offset"]
        ndim = len(shape)
        itemsize = dtype.itemsize

        # Check if chunk covers full inner dimensions (contiguous slab)
        inner_full = all(
            indices[d] == 0 and chunk_shape[d] >= shape[d]
            for d in range(1, ndim)
        )

        if inner_full:
            # Contiguous slab along axis 0
            inner_size = math.prod(shape[1:])
            slab_bytes = inner_size * itemsize
            chunk_start = indices[0] * chunk_shape[0]
            actual_dim0 = min(chunk_shape[0], shape[0] - chunk_start)
            offset = data_offset + chunk_start * slab_bytes
            length = actual_dim0 * slab_bytes

            with open(self._file_path, "rb") as f:
                f.seek(offset)
                return f.read(length)
        else:
            # General case: read entire data and slice
            total_bytes = math.prod(shape) * itemsize
            with open(self._file_path, "rb") as f:
                f.seek(data_offset)
                all_data = f.read(total_bytes)

            arr = np.frombuffer(all_data, dtype=dtype).reshape(shape)
            slices = []
            for d in range(ndim):
                start = indices[d] * chunk_shape[d]
                end = min(start + chunk_shape[d], shape[d])
                slices.append(slice(start, end))
            return np.ascontiguousarray(arr[tuple(slices)]).tobytes()

    def _read_compressed_chunk(self) -> bytes:
        """Read the entire compressed data section."""
        data_offset = self._header["data_offset"]
        with open(self._file_path, "rb") as f:
            f.seek(data_offset)
            return f.read()

    # -- Zarr Store interface --

    async def open(self, *args: Any, **kwargs: Any) -> Self:
        self._is_open = True
        return self

    async def close(self) -> None:
        self._is_open = False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NRRDStore):
            return NotImplemented
        return self._file_path == other._file_path

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        if key == "zarr.json":
            raw = self._zarr_json.encode("utf-8")
            raw = _apply_byte_range(raw, byte_range)
            return prototype.buffer.from_bytes(raw)

        if key not in self._keys:
            return None

        if self._is_raw:
            indices = self._chunk_key_to_indices(key)
            raw = self._read_raw_chunk(indices)
        else:
            raw = self._read_compressed_chunk()

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
        return key in self._keys

    async def set(self, key: str, value: Buffer) -> None:
        raise NotImplementedError("NRRDStore is read-only")

    async def delete(self, key: str) -> None:
        raise NotImplementedError("NRRDStore is read-only")

    async def list(self) -> AsyncIterator[str]:
        for k in sorted(self._keys):
            yield k

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        for k in sorted(self._keys):
            if k.startswith(prefix):
                yield k

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"
        if not prefix:
            prefix = ""

        seen: set[str] = set()
        for k in self._keys:
            if not k.startswith(prefix):
                continue
            rest = k[len(prefix):]
            slash = rest.find("/")
            entry = rest if slash == -1 else rest[:slash + 1]
            if entry not in seen:
                seen.add(entry)
                yield entry
