from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from zarr_zmp import Manifest, Builder, ZMPStore


async def _collect(ait):
    return [item async for item in ait]


class TestBuilder:
    def test_inline_roundtrip(self, tmp_path: Path) -> None:
        """Build a manifest with inline metadata and chunks, read via ZMPStore."""
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

    def test_virtual_refs(self, tmp_path: Path) -> None:
        """Virtual entries have external_uri, offset, length."""
        builder = Builder()
        builder.add("zarr.json", text='{"zarr_format":3,"node_type":"group"}')
        builder.add(
            "arr/c/0",
            uri="s3://bucket/file.nc",
            offset=1024,
            length=4096,
        )

        zmp_path = builder.write(tmp_path / "out.zmp")
        manifest = Manifest(str(zmp_path))

        entry = manifest.get_entry("arr/c/0")
        assert entry is not None
        assert entry.uri == "s3://bucket/file.nc"
        assert entry.offset == 1024
        assert entry.length == 4096
        assert entry.size == 4096

    def test_ref_entries(self, tmp_path: Path) -> None:
        """Reference entries have retrieval_key but no inline data."""
        builder = Builder()
        builder.add("zarr.json", text='{"zarr_format":3,"node_type":"group"}')
        builder.add(
            "arr/c/0",
            retrieval_key="abcdef1234567890abcdef1234567890abcdef12",
            size=32768,
        )

        zmp_path = builder.write(tmp_path / "out.zmp")
        manifest = Manifest(str(zmp_path))

        entry = manifest.get_entry("arr/c/0")
        assert entry is not None
        assert entry.retrieval_key == "abcdef1234567890abcdef1234567890abcdef12"
        assert entry.size == 32768
        assert entry.text is None
        assert manifest.get_data("arr/c/0") is None

    def test_retrieval_keys_raw_text(self, tmp_path: Path) -> None:
        """Builder hashes raw text bytes (no implicit canonicalization)."""
        import hashlib

        text = '{"b": 1, "a": 2}'
        builder = Builder()
        rk = builder.add("zarr.json", text=text)

        raw = text.encode("utf-8")
        header = f"blob {len(raw)}\0".encode()
        expected = hashlib.sha1(header + raw).hexdigest()

        assert rk == expected

    def test_data_hash_is_raw(self, tmp_path: Path) -> None:
        """Data hashes are git-sha1 of raw bytes (not canonicalized)."""
        import hashlib

        data = b"\x00\x01\x02\x03"
        builder = Builder()
        rk = builder.add("c/0", data=data)

        header = f"blob {len(data)}\0".encode()
        expected = hashlib.sha1(header + data).hexdigest()
        assert rk == expected

    def test_array_path_inferred(self, tmp_path: Path) -> None:
        builder = Builder()
        builder.add("myarray/c/0/1", data=b"\x00")
        zmp_path = builder.write(tmp_path / "out.zmp")

        manifest = Manifest(str(zmp_path))
        entry = manifest.get_entry("myarray/c/0/1")
        assert entry.array_path == "myarray"
        assert entry.chunk_key == "0/1"

    def test_array_path_explicit(self, tmp_path: Path) -> None:
        builder = Builder()
        builder.add(
            "data/c/0",
            data=b"\x00",
            array_path="custom_array",
            chunk_key="0",
        )
        zmp_path = builder.write(tmp_path / "out.zmp")

        manifest = Manifest(str(zmp_path))
        entry = manifest.get_entry("data/c/0")
        assert entry.array_path == "custom_array"
        assert entry.chunk_key == "0"

    def test_sorted_output(self, tmp_path: Path) -> None:
        """Non-data rows come first (sorted), then data rows (sorted)."""
        builder = Builder()
        builder.add("zarr.json", text='{}')
        builder.add("b/c/0", data=b"\x01")
        builder.add("a/c/0", data=b"\x00")

        zmp_path = builder.write(tmp_path / "out.zmp")
        manifest = Manifest(str(zmp_path))
        paths = list(manifest.list_paths())
        # root index row first, then text, then data rows sorted
        assert paths == ["", "zarr.json", "a/c/0", "b/c/0"]

    def test_file_level_metadata(self, tmp_path: Path) -> None:
        builder = Builder(
            zarr_format="3",
            retrieval_scheme="git-sha1",
            metadata={"description": "test", "doi": "10.1234/test"},
        )
        builder.add("zarr.json", text='{}')
        zmp_path = builder.write(tmp_path / "out.zmp")

        manifest = Manifest(str(zmp_path))
        assert manifest.metadata["zmp_version"] == "0.1.0"
        assert manifest.metadata["zarr_format"] == "3"
        assert manifest.metadata.get("extra", {}).get("description") == "test"
        assert manifest.metadata.get("extra", {}).get("doi") == "10.1234/test"

    def test_row_group_sizing_inline(self, tmp_path: Path) -> None:
        """Inline data uses 2 rows per group."""
        import pyarrow.parquet as pq

        builder = Builder()
        builder.add("zarr.json", text='{}')
        for i in range(10):
            builder.add(f"c/{i}", data=b"\x00" * 100)
        zmp_path = builder.write(tmp_path / "out.zmp")

        pf = pq.ParquetFile(str(zmp_path))
        for i in range(pf.metadata.num_row_groups):
            assert pf.metadata.row_group(i).num_rows <= 2

    def test_row_group_sizing_refs(self, tmp_path: Path) -> None:
        """Reference-only uses ~16 row groups."""
        import pyarrow.parquet as pq

        builder = Builder()
        for i in range(1000):
            builder.add(f"c/{i}", retrieval_key="a" * 40, size=100)
        zmp_path = builder.write(tmp_path / "out.zmp")

        pf = pq.ParquetFile(str(zmp_path))
        assert pf.metadata.num_row_groups <= 20

    def test_mixed_inline_and_virtual(self, tmp_path: Path) -> None:
        """Builder handles a mix of inline and virtual entries."""
        builder = Builder()
        builder.add("zarr.json", text='{"zarr_format":3,"node_type":"group"}')
        builder.add("native/c/0", data=b"\x00" * 16)
        builder.add(
            "virtual/c/0",
            uri="https://example.com/data.bin",
            offset=0,
            length=1024,
        )
        zmp_path = builder.write(tmp_path / "out.zmp")

        manifest = Manifest(str(zmp_path))
        assert manifest.get_data("native/c/0") is not None
        assert manifest.get_entry("virtual/c/0").uri == "https://example.com/data.bin"

    def test_root_metadata(self, tmp_path: Path) -> None:
        builder = Builder()
        builder.set_root_metadata({
            "description": "CT scan",
            "modality": "CT",
            "series_uid": "1.2.840.113619.1234",
        })
        builder.add("zarr.json", text='{"zarr_format":3,"node_type":"group"}')
        zmp_path = builder.write(tmp_path / "out.zmp")

        manifest = Manifest(str(zmp_path))
        rm = manifest.root_metadata
        assert rm is not None
        assert rm["description"] == "CT scan"
        assert rm["modality"] == "CT"
        assert rm["series_uid"] == "1.2.840.113619.1234"

    def test_root_metadata_invisible_to_store(self, tmp_path: Path) -> None:
        """Root row is hidden from zarr Store interface."""
        builder = Builder()
        builder.set_root_metadata({"test": True})
        builder.add("zarr.json", text='{"zarr_format":3,"node_type":"group"}')
        zmp_path = builder.write(tmp_path / "out.zmp")

        store = ZMPStore.from_file(str(zmp_path))
        store._is_open = True

        import asyncio
        paths = asyncio.run(_collect(store.list()))
        assert "" not in paths
        assert "zarr.json" in paths

        assert not asyncio.run(store.exists(""))

    def test_per_entry_metadata(self, tmp_path: Path) -> None:
        """Per-entry metadata is stored as JSON and queryable."""
        import pyarrow.parquet as pq

        builder = Builder()
        builder.add("zarr.json", text='{"zarr_format":3,"node_type":"group"}')
        builder.add(
            "vol/c/0",
            data=b"\x00" * 16,
            metadata={"SliceLocation": 42.5, "InstanceNumber": 1},
        )
        builder.add(
            "vol/c/1",
            uri="s3://bucket/file.dcm",
            offset=1024,
            length=4096,
            metadata={"SliceLocation": 45.0, "InstanceNumber": 2},
        )
        builder.add(
            "vol/c/2",
            retrieval_key="a" * 40,
            size=4096,
            metadata={"SliceLocation": 47.5},
        )
        zmp_path = builder.write(tmp_path / "out.zmp")

        pf = pq.ParquetFile(str(zmp_path))
        table = pf.read()
        meta_col = table.column("metadata")

        metas = {}
        for i in range(len(table)):
            path = table.column("path")[i].as_py()
            m = meta_col[i].as_py()
            if m is not None:
                metas[path] = json.loads(m)

        assert metas["vol/c/0"]["SliceLocation"] == 42.5
        assert metas["vol/c/0"]["InstanceNumber"] == 1
        assert metas["vol/c/1"]["SliceLocation"] == 45.0
        assert metas["vol/c/2"]["SliceLocation"] == 47.5

    def test_auto_size(self, tmp_path: Path) -> None:
        """Size is auto-computed from content."""
        builder = Builder()
        builder.add("a.json", text="hello")
        builder.add("b.bin", data=b"\x00" * 100)
        builder.add("c.bin", uri="s3://x", offset=0, length=500)
        builder.add("d.bin", retrieval_key="a" * 40, size=200)
        zmp_path = builder.write(tmp_path / "out.zmp")

        manifest = Manifest(str(zmp_path))
        assert manifest["a.json"].size == 5
        assert manifest["b.bin"].size == 100
        assert manifest["c.bin"].size == 500
        assert manifest["d.bin"].size == 200

    def test_addressing_flags(self, tmp_path: Path) -> None:
        """Addressing column is auto-computed."""
        import pyarrow.parquet as pq

        builder = Builder()
        builder.add("a", text="hello")                          # T, K
        builder.add("b", data=b"\x00")                          # D, K
        builder.add("c", uri="s3://x", offset=0, length=10)    # U
        builder.add("d", retrieval_key="a" * 40, size=10)       # K
        builder.add("e", data=b"\x01", uri="s3://y", offset=0, length=1)  # D, K, U
        zmp_path = builder.write(tmp_path / "out.zmp")

        pf = pq.ParquetFile(str(zmp_path))
        table = pf.read()
        addr = {
            table.column("path")[i].as_py(): table.column("addressing")[i].as_py()
            for i in range(len(table))
        }
        assert set(addr["a"]) == {"T", "K"}
        assert set(addr["b"]) == {"D", "K"}
        assert set(addr["c"]) == {"U"}
        assert set(addr["d"]) == {"K"}
        assert set(addr["e"]) == {"D", "K", "U"}
