from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import rfc8785
import zarr

from zarr_zmp import Manifest, ZMPStore, ZMPWritableStore


def git_blob_hash(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


class TestEmbeddedWrite:
    def test_roundtrip(self, tmp_path: Path) -> None:
        """Write with zarr through ZMPWritableStore, read back with ZMPStore."""
        zmp_path = tmp_path / "out.zmp"
        with ZMPWritableStore.create(zmp_path) as store:
            root = zarr.open_group(store=store, mode="w")
            root.create_array("temp", data=np.arange(8.0), chunks=(2,))

        store = ZMPStore.from_file(str(zmp_path))
        store._is_open = True
        group = zarr.open_group(store=store, mode="r")
        np.testing.assert_array_equal(group["temp"][:], np.arange(8.0))

    def test_multidim(self, tmp_path: Path) -> None:
        zmp_path = tmp_path / "out.zmp"
        original = np.arange(12.0).reshape(3, 4)
        with ZMPWritableStore.create(zmp_path) as store:
            root = zarr.open_group(store=store, mode="w")
            root.create_array("matrix", data=original, chunks=(2, 2))

        store = ZMPStore.from_file(str(zmp_path))
        store._is_open = True
        group = zarr.open_group(store=store, mode="r")
        np.testing.assert_array_equal(group["matrix"][:], original)

    def test_nested_groups(self, tmp_path: Path) -> None:
        zmp_path = tmp_path / "out.zmp"
        with ZMPWritableStore.create(zmp_path) as store:
            root = zarr.open_group(store=store, mode="w")
            g = root.create_group("sub")
            g.create_array("arr", data=np.array([1.0, 2.0, 3.0]), chunks=(3,))

        store = ZMPStore.from_file(str(zmp_path))
        store._is_open = True
        group = zarr.open_group(store=store, mode="r")
        np.testing.assert_array_equal(group["sub/arr"][:], [1.0, 2.0, 3.0])

    def test_metadata_inlined_as_text(self, tmp_path: Path) -> None:
        zmp_path = tmp_path / "out.zmp"
        with ZMPWritableStore.create(zmp_path) as store:
            zarr.open_group(store=store, mode="w")

        manifest = Manifest(str(zmp_path))
        entry = manifest.get_entry("zarr.json")
        assert entry is not None
        assert entry.text is not None


    def test_chunks_inlined_as_data(self, tmp_path: Path) -> None:
        zmp_path = tmp_path / "out.zmp"
        with ZMPWritableStore.create(zmp_path) as store:
            root = zarr.open_group(store=store, mode="w")
            root.create_array("arr", data=np.arange(4.0), chunks=(4,))

        manifest = Manifest(str(zmp_path))
        entry = manifest.get_entry("arr/c/0")
        assert entry is not None

        data = manifest.get_data("arr/c/0")
        assert data is not None

    def test_row_group_sizing_embedded(self, tmp_path: Path) -> None:
        """Embedded mode: data rows first, non-data second-to-last, index last."""
        zmp_path = tmp_path / "out.zmp"
        with ZMPWritableStore.create(zmp_path) as store:
            root = zarr.open_group(store=store, mode="w")
            root.create_array("arr", data=np.arange(20.0), chunks=(2,))

        import pyarrow.parquet as pq

        pf = pq.ParquetFile(str(zmp_path))
        num_rgs = pf.metadata.num_row_groups
        # Data rows: one per group
        for i in range(num_rgs - 2):
            assert pf.metadata.row_group(i).num_rows == 1
        # Second-to-last: non-data rows (metadata)
        assert pf.metadata.row_group(num_rgs - 2).num_rows >= 1
        # Last: index row alone
        assert pf.metadata.row_group(num_rgs - 1).num_rows == 1

    def test_canonical_json_hashing(self, tmp_path: Path) -> None:
        """Metadata hashes use canonical JSON (RFC 8785)."""
        zmp_path = tmp_path / "out.zmp"
        with ZMPWritableStore.create(zmp_path) as store:
            root = zarr.open_group(store=store, mode="w")

        manifest = Manifest(str(zmp_path))
        entry = manifest.get_entry("zarr.json")
        assert entry is not None

        # The retrieval_key should be the git hash of the canonical JSON,
        # not the raw bytes zarr wrote
        raw_json = entry.text.encode("utf-8")
        canonical = rfc8785.dumps(json.loads(raw_json))
        expected_hash = git_blob_hash(canonical)
        assert entry.retrieval_key == expected_hash

    def test_retrieval_keys_present(self, tmp_path: Path) -> None:
        zmp_path = tmp_path / "out.zmp"
        with ZMPWritableStore.create(zmp_path) as store:
            root = zarr.open_group(store=store, mode="w")
            root.create_array("arr", data=np.arange(4.0), chunks=(4,))

        manifest = Manifest(str(zmp_path))
        for path in manifest.list_paths():
            if path == "":
                continue  # root/index row has no retrieval key
            entry = manifest.get_entry(path)
            assert entry is not None
            assert entry.retrieval_key is not None
            assert len(entry.retrieval_key) == 40  # SHA-1 hex


class TestExternalWrite:
    def test_chunks_written_to_dir(self, tmp_path: Path) -> None:
        zmp_path = tmp_path / "out.zmp"
        chunk_dir = tmp_path / "chunks"
        with ZMPWritableStore.create(zmp_path, chunk_dir=chunk_dir) as store:
            root = zarr.open_group(store=store, mode="w")
            root.create_array("arr", data=np.arange(4.0), chunks=(2,))

        # Chunks should be in chunk_dir
        chunk_files = list(chunk_dir.iterdir())
        assert len(chunk_files) >= 2

        # Chunk filenames should be git-sha1 hashes
        for f in chunk_files:
            assert len(f.name) == 40

    def test_no_inline_data(self, tmp_path: Path) -> None:
        """External mode should not inline chunk data."""
        zmp_path = tmp_path / "out.zmp"
        chunk_dir = tmp_path / "chunks"
        with ZMPWritableStore.create(zmp_path, chunk_dir=chunk_dir) as store:
            root = zarr.open_group(store=store, mode="w")
            root.create_array("arr", data=np.arange(4.0), chunks=(2,))

        manifest = Manifest(str(zmp_path))
        entry = manifest.get_entry("arr/c/0")
        assert entry is not None
        assert manifest.get_data("arr/c/0") is None
        assert entry.retrieval_key is not None

    def test_metadata_still_inlined(self, tmp_path: Path) -> None:
        """Even in external mode, metadata is inlined as text."""
        zmp_path = tmp_path / "out.zmp"
        chunk_dir = tmp_path / "chunks"
        with ZMPWritableStore.create(zmp_path, chunk_dir=chunk_dir) as store:
            zarr.open_group(store=store, mode="w")

        manifest = Manifest(str(zmp_path))
        entry = manifest.get_entry("zarr.json")
        assert entry is not None
        assert entry.text is not None

    def test_roundtrip_via_resolver(self, tmp_path: Path) -> None:
        """Write external, read back via TemplateResolver."""
        zmp_path = tmp_path / "out.zmp"
        chunk_dir = tmp_path / "chunks"
        with ZMPWritableStore.create(zmp_path, chunk_dir=chunk_dir) as store:
            root = zarr.open_group(store=store, mode="w")
            root.create_array("arr", data=np.arange(8.0), chunks=(2,))

        store = ZMPStore.from_url(str(zmp_path), blobs=f"{chunk_dir}/{{hash}}")
        group = zarr.open_group(store=store, mode="r")
        np.testing.assert_array_equal(group["arr"][:], np.arange(8.0))

    def test_dedup(self, tmp_path: Path) -> None:
        """Identical chunks are only written once to chunk_dir."""
        zmp_path = tmp_path / "out.zmp"
        chunk_dir = tmp_path / "chunks"
        # Two arrays with identical data = identical chunks
        with ZMPWritableStore.create(zmp_path, chunk_dir=chunk_dir) as store:
            root = zarr.open_group(store=store, mode="w")
            data = np.zeros(4)
            data[0] = 1.0  # make non-fill so zarr stores the chunk
            root.create_array("a", data=data, chunks=(4,))
            root.create_array("b", data=data, chunks=(4,))

        chunk_files = [f for f in chunk_dir.iterdir() if len(f.name) == 40]
        # Should be 1 chunk file, not 2 (deduped by content hash)
        assert len(chunk_files) == 1

    def test_row_group_sizing_external(self, tmp_path: Path) -> None:
        """External mode uses ~16 row groups (no inline data)."""
        zmp_path = tmp_path / "out.zmp"
        chunk_dir = tmp_path / "chunks"
        with ZMPWritableStore.create(zmp_path, chunk_dir=chunk_dir) as store:
            root = zarr.open_group(store=store, mode="w")
            root.create_array("arr", data=np.arange(100.0), chunks=(2,))

        import pyarrow.parquet as pq

        pf = pq.ParquetFile(str(zmp_path))
        # Should have roughly 16 row groups, not 1-per-row
        assert pf.metadata.num_row_groups <= 20
