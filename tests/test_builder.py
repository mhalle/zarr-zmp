from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from zarr_zmp import Manifest, ZMPStore, build_zmp


class TestBuildZMP:
    def test_roundtrip_simple(self, tmp_path: Path) -> None:
        """Write with zarr -> build ZMP -> read back through ZMPStore."""
        # Write
        src = MemoryStore()
        root = zarr.open_group(store=src, mode="w")
        root.create_array("temp", data=np.arange(8.0), chunks=(2,))

        # Build
        zmp_path = build_zmp(src, tmp_path / "out.zmp")

        # Read back
        store = ZMPStore.from_file(str(zmp_path))
        store._is_open = True
        group = zarr.open_group(store=store, mode="r")
        data = group["temp"][:]
        np.testing.assert_array_equal(data, np.arange(8.0))

    def test_metadata_inlined_as_text(self, tmp_path: Path) -> None:
        src = MemoryStore()
        zarr.open_group(store=src, mode="w")

        zmp_path = build_zmp(src, tmp_path / "out.zmp")
        manifest = Manifest(str(zmp_path))

        entry = manifest.get_entry("zarr.json")
        assert entry is not None

        assert entry.text is not None
        parsed = json.loads(entry.text)
        assert parsed["node_type"] == "group"

    def test_chunks_inlined_as_data(self, tmp_path: Path) -> None:
        src = MemoryStore()
        root = zarr.open_group(store=src, mode="w")
        root.create_array("arr", data=np.arange(4.0), chunks=(4,))

        zmp_path = build_zmp(src, tmp_path / "out.zmp")
        manifest = Manifest(str(zmp_path))

        entry = manifest.get_entry("arr/c/0")
        assert entry is not None

        assert entry.text is None
        # Data column should have content
        data = manifest.get_data("arr/c/0")
        assert data is not None
        assert len(data) > 0

    def test_chunk_entries_present(self, tmp_path: Path) -> None:
        """Chunk entries are written with correct paths."""
        src = MemoryStore()
        root = zarr.open_group(store=src, mode="w")
        root.create_array("myarr", data=np.arange(16.0).reshape(4, 4), chunks=(2, 2))

        zmp_path = build_zmp(src, tmp_path / "out.zmp")
        manifest = Manifest(str(zmp_path))
        assert manifest.has("myarr/c/0/1")

    def test_file_level_metadata(self, tmp_path: Path) -> None:
        src = MemoryStore()
        zarr.open_group(store=src, mode="w")

        zmp_path = build_zmp(
            src,
            tmp_path / "out.zmp",
            zarr_format="3",
            metadata={"description": "test dataset"},
        )
        manifest = Manifest(str(zmp_path))
        assert manifest.metadata["zmp_version"] == "0.2.0"
        assert manifest.metadata["zarr_format"] == "3"
        assert manifest.metadata.get("extra", {}).get("description") == "test dataset"

    def test_listing_roundtrip(self, tmp_path: Path) -> None:
        src = MemoryStore()
        root = zarr.open_group(store=src, mode="w")
        root.create_array("a", data=np.zeros(2), chunks=(2,))
        root.create_array("b", data=np.zeros(2), chunks=(2,))

        zmp_path = build_zmp(src, tmp_path / "out.zmp")
        store = ZMPStore.from_file(str(zmp_path))
        store._is_open = True

        group = zarr.open_group(store=store, mode="r")
        assert "a" in group
        assert "b" in group

    def test_multidim_roundtrip(self, tmp_path: Path) -> None:
        """2D array roundtrip."""
        src = MemoryStore()
        root = zarr.open_group(store=src, mode="w")
        original = np.arange(12.0).reshape(3, 4)
        root.create_array("matrix", data=original, chunks=(2, 2))

        zmp_path = build_zmp(src, tmp_path / "out.zmp")
        store = ZMPStore.from_file(str(zmp_path))
        store._is_open = True

        group = zarr.open_group(store=store, mode="r")
        result = group["matrix"][:]
        np.testing.assert_array_equal(result, original)
