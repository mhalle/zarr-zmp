"""Tests for zarr-specific builder integration (build_zmp + ZMPStore)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from zarr_zmp import Builder, Manifest, ZMPStore


async def _collect(ait):
    return [item async for item in ait]


class TestBuilder:
    def test_inline_roundtrip(self, tmp_path: Path) -> None:
        """Build a manifest with inline content, read via ZMPStore + zarr."""
        group_meta = json.dumps({"zarr_format": 3, "node_type": "group"})
        array_meta = json.dumps({
            "zarr_format": 3,
            "node_type": "array",
            "shape": [4],
            "data_type": "float64",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2]}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": 0,
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        })

        chunk0 = np.array([1.0, 2.0], dtype="<f8").tobytes()
        chunk1 = np.array([3.0, 4.0], dtype="<f8").tobytes()

        builder = Builder()
        builder.add("zarr.json", text=group_meta)
        builder.add("arr/zarr.json", text=array_meta)
        builder.add("arr/c/0", data=chunk0)
        builder.add("arr/c/1", data=chunk1)

        zmp_path = builder.write(tmp_path / "out.zmp")

        store = ZMPStore.from_file(str(zmp_path))
        store._is_open = True
        group = zarr.open_group(store=store, mode="r")
        np.testing.assert_array_equal(group["arr"][:], [1.0, 2.0, 3.0, 4.0])

    def test_root_metadata_invisible_to_store(self, tmp_path: Path) -> None:
        """Root row is hidden from zarr Store interface."""
        builder = Builder()
        builder.set_root_metadata({"test": True})
        builder.add("zarr.json", text='{"zarr_format":3,"node_type":"group"}')
        zmp_path = builder.write(tmp_path / "out.zmp")

        store = ZMPStore.from_file(str(zmp_path))
        store._is_open = True

        paths = asyncio.run(_collect(store.list()))
        assert "" not in paths
        assert "zarr.json" in paths

        assert not asyncio.run(store.exists(""))
