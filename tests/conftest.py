from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from zmanifest import Builder


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
    """Create a simple ZMP file with a group and one 1D array (4 chunks)."""
    group_meta = json.dumps({"zarr_format": 3, "node_type": "group"})
    array_meta = _make_zarr_metadata(shape=(8,), chunks=(2,), dtype="float64")

    builder = Builder()
    builder.add("/zarr.json", text=group_meta)
    builder.add("/temp/zarr.json", text=array_meta)
    for i in range(4):
        arr = np.array([i * 2.0, i * 2.0 + 1.0], dtype="<f8")
        builder.add(f"/temp/c/{i}", data=arr.tobytes())

    return builder.write(tmp_path / "simple.zmp")
