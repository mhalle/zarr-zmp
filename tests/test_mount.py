from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from zarr_zmp import Addressing, Manifest, Builder, ZMPStore, ZMPWritableStore


async def _collect(ait):
    return [item async for item in ait]


@pytest.fixture
def child_zmp(tmp_path: Path) -> Path:
    """A self-contained ZMP with one array."""
    zmp_path = tmp_path / "child.zmp"
    with ZMPWritableStore.create(zmp_path) as store:
        root = zarr.open_group(store=store, mode="w")
        root.create_array("arr", data=np.arange(8.0), chunks=(2,))
    return zmp_path


@pytest.fixture
def child_zip(tmp_path: Path) -> Path:
    """A zarr.zip with one array."""
    zip_path = tmp_path / "child.zarr.zip"
    from zarr.storage import ZipStore
    zs = ZipStore(str(zip_path), mode="w")
    root = zarr.open_group(store=zs, mode="w")
    root.create_array("arr", data=np.arange(4.0, 8.0), chunks=(2,))
    zs.close()
    return zip_path


class TestMount:
    def test_mount_zmp_read(self, tmp_path: Path, child_zmp: Path) -> None:
        """Mount a ZMP file and read through the parent store."""
        builder = Builder()
        builder.add("zarr.json", text=json.dumps({"zarr_format": 3, "node_type": "group"}))
        builder.mount("scans/ct", str(child_zmp))
        parent = tmp_path / "parent.zmp"
        builder.write(parent)

        store = ZMPStore.from_file(str(parent))
        store._is_open = True
        group = zarr.open_group(store=store, mode="r")
        np.testing.assert_array_equal(group["scans/ct/arr"][:], np.arange(8.0))

    def test_mount_zip_read(self, tmp_path: Path, child_zip: Path) -> None:
        """Mount a zarr.zip and read through the parent store."""
        builder = Builder()
        builder.add("zarr.json", text=json.dumps({"zarr_format": 3, "node_type": "group"}))
        builder.mount("scans/mri", str(child_zip))
        parent = tmp_path / "parent.zmp"
        builder.write(parent)

        store = ZMPStore.from_file(str(parent))
        store._is_open = True
        group = zarr.open_group(store=store, mode="r")
        np.testing.assert_array_equal(group["scans/mri/arr"][:], np.arange(4.0, 8.0))

    def test_mount_addressing_flags(self, tmp_path: Path, child_zmp: Path) -> None:
        builder = Builder()
        builder.add("zarr.json", text='{}')
        builder.mount("sub", str(child_zmp))
        parent = tmp_path / "parent.zmp"
        builder.write(parent)

        manifest = Manifest(str(parent))
        entry = manifest.get_entry("sub/")
        assert entry is not None
        assert Addressing.MOUNT in entry.addressing
        assert Addressing.URI in entry.addressing

    def test_mount_exists(self, tmp_path: Path, child_zmp: Path) -> None:
        builder = Builder()
        builder.add("zarr.json", text='{}')
        builder.mount("sub", str(child_zmp))
        parent = tmp_path / "parent.zmp"
        builder.write(parent)

        store = ZMPStore.from_file(str(parent))
        store._is_open = True

        # Mount point itself is an annotation — not visible
        assert not asyncio.run(store.exists("sub/"))
        # But paths inside the mount work
        assert asyncio.run(store.exists("sub/zarr.json"))
        assert asyncio.run(store.exists("sub/arr/zarr.json"))

    def test_mount_list(self, tmp_path: Path, child_zmp: Path) -> None:
        builder = Builder()
        builder.add("zarr.json", text='{}')
        builder.mount("sub", str(child_zmp))
        parent = tmp_path / "parent.zmp"
        builder.write(parent)

        store = ZMPStore.from_file(str(parent))
        store._is_open = True

        paths = asyncio.run(_collect(store.list()))
        assert "zarr.json" in paths
        assert any(p.startswith("sub/") for p in paths)
        # The mount annotation row should not be in the list
        assert "sub/" not in paths

    def test_mount_list_dir(self, tmp_path: Path, child_zmp: Path) -> None:
        builder = Builder()
        builder.add("zarr.json", text='{}')
        builder.mount("sub", str(child_zmp))
        parent = tmp_path / "parent.zmp"
        builder.write(parent)

        store = ZMPStore.from_file(str(parent))
        store._is_open = True

        # Top level should show sub/ as a directory
        top = asyncio.run(_collect(store.list_dir("")))
        assert "sub/" in top
        assert "zarr.json" in top

        # Inside the mount
        sub = asyncio.run(_collect(store.list_dir("sub/")))
        assert "zarr.json" in sub
        assert "arr/" in sub

    def test_mount_mixed_local_and_mounted(self, tmp_path: Path, child_zmp: Path) -> None:
        """Parent has local entries alongside a mount."""
        builder = Builder()
        builder.add("zarr.json", text=json.dumps({"zarr_format": 3, "node_type": "group"}))
        builder.add("local/zarr.json", text=json.dumps({
            "zarr_format": 3, "node_type": "array", "shape": [2],
            "data_type": "float64",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2]}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": 0,
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        }))
        builder.add("local/c/0", data=np.array([10.0, 20.0], dtype="<f8").tobytes())
        builder.mount("mounted", str(child_zmp))
        parent = tmp_path / "parent.zmp"
        builder.write(parent)

        store = ZMPStore.from_file(str(parent))
        store._is_open = True
        group = zarr.open_group(store=store, mode="r")

        # Local array
        np.testing.assert_array_equal(group["local"][:], [10.0, 20.0])
        # Mounted array
        np.testing.assert_array_equal(group["mounted/arr"][:], np.arange(8.0))

    def test_mount_unsupported_target(self, tmp_path: Path) -> None:
        builder = Builder()
        builder.add("zarr.json", text='{}')
        builder.mount("sub", "/data/store.zarr")  # directory, not supported
        parent = tmp_path / "parent.zmp"
        builder.write(parent)

        store = ZMPStore.from_file(str(parent))
        store._is_open = True

        with pytest.raises(ValueError, match="Unsupported mount target"):
            asyncio.run(store.exists("sub/zarr.json"))
