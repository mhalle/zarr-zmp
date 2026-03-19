from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr


def _make_zarr_metadata(
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: str = "float64",
) -> str:
    """Create a zarr v3 array metadata JSON string."""
    return json.dumps(
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": list(shape),
            "data_type": dtype,
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": list(chunks)},
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "/"},
            },
            "fill_value": 0,
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        }
    )


@pytest.fixture
def simple_zmp(tmp_path: Path) -> Path:
    """Create a simple ZMP file with a group and one 1D array (4 chunks, inline data)."""
    group_meta = json.dumps({"zarr_format": 3, "node_type": "group"})
    array_meta = _make_zarr_metadata(shape=(8,), chunks=(2,), dtype="float64")

    # Create chunk data as raw little-endian float64 bytes
    chunks_data = []
    for i in range(4):
        arr = np.array([i * 2.0, i * 2.0 + 1.0], dtype="<f8")
        chunks_data.append(arr.tobytes())

    paths = [
        "zarr.json",
        "temp/zarr.json",
        "temp/c/0",
        "temp/c/1",
        "temp/c/2",
        "temp/c/3",
    ]
    sizes = [
        len(group_meta),
        len(array_meta),
        len(chunks_data[0]),
        len(chunks_data[1]),
        len(chunks_data[2]),
        len(chunks_data[3]),
    ]
    texts = [group_meta, array_meta, None, None, None, None]
    data = [None, None] + chunks_data
    addressing = [
        ["T"], ["T"],  # metadata: inline text
        ["D"], ["D"], ["D"], ["D"],  # chunks: inline data
    ]

    table = pa.table(
        {
            "path": pa.array(paths, type=pa.string()),
            "size": pa.array(sizes, type=pa.int64()),
            "text": pa.array(texts, type=pa.string()),
            "data": pa.array(data, type=pa.binary()),
            "addressing": pa.array(addressing, type=pa.list_(pa.string())),
        }
    )

    file_meta = {
        b"zmp_version": b'"0.1.0"',
        b"zarr_format": b'"3"',
        b"retrieval_scheme": b'"sha256"',
    }
    schema = table.schema.with_metadata(file_meta)
    table = table.cast(schema)

    out = tmp_path / "simple.zmp"
    pq.write_table(table, str(out))
    return out
