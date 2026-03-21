from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.abc.store import RangeByteRequest, OffsetByteRequest, SuffixByteRequest
from zarr.core.buffer import default_buffer_prototype

from zarr_zmp import Manifest, ZMPStore


# --- Manifest tests ---


class TestManifest:
    def test_metadata(self, simple_zmp: Path) -> None:
        m = Manifest(str(simple_zmp))
        assert m.metadata["zmp_version"] == "0.2.0"
        assert m.metadata["zarr_format"] == "3"

    def test_has(self, simple_zmp: Path) -> None:
        m = Manifest(str(simple_zmp))
        assert m.has("zarr.json")
        assert m.has("temp/c/0")
        assert not m.has("nonexistent")

    def test_get_entry(self, simple_zmp: Path) -> None:
        m = Manifest(str(simple_zmp))
        entry = m.get_entry("zarr.json")
        assert entry is not None
        assert entry.text is not None
        assert entry.text is not None
        parsed = json.loads(entry.text)
        assert parsed["node_type"] == "group"

    def test_get_entry_missing(self, simple_zmp: Path) -> None:
        m = Manifest(str(simple_zmp))
        assert m.get_entry("nonexistent") is None

    def test_get_data(self, simple_zmp: Path) -> None:
        m = Manifest(str(simple_zmp))
        data = m.get_data("temp/c/0")
        assert data is not None
        arr = np.frombuffer(data, dtype="<f8")
        np.testing.assert_array_equal(arr, [0.0, 1.0])

    def test_list_paths(self, simple_zmp: Path) -> None:
        m = Manifest(str(simple_zmp))
        paths = list(m.list_paths())
        assert len(paths) == 6
        assert "zarr.json" in paths

    def test_list_prefix(self, simple_zmp: Path) -> None:
        m = Manifest(str(simple_zmp))
        assert len(list(m.list_prefix("temp/c/"))) == 4
        assert len(list(m.list_prefix("temp/"))) == 5

    def test_list_dir(self, simple_zmp: Path) -> None:
        m = Manifest(str(simple_zmp))
        top = list(m.list_dir(""))
        assert "zarr.json" in top
        assert "temp/" in top
        assert len(top) == 2

        temp = list(m.list_dir("temp"))
        assert "zarr.json" in temp
        assert "c/" in temp


# --- Store tests ---


class TestZMPStore:
    @pytest.fixture
    def store(self, simple_zmp: Path) -> ZMPStore:
        return ZMPStore.from_file(str(simple_zmp))

    @pytest.mark.asyncio
    async def test_exists(self, store: ZMPStore) -> None:
        assert await store.exists("zarr.json")
        assert not await store.exists("nonexistent")

    @pytest.mark.asyncio
    async def test_get_metadata(self, store: ZMPStore) -> None:
        proto = default_buffer_prototype()
        buf = await store.get("zarr.json", proto)
        assert buf is not None
        parsed = json.loads(buf.to_bytes())
        assert parsed["zarr_format"] == 3
        assert parsed["node_type"] == "group"

    @pytest.mark.asyncio
    async def test_get_chunk(self, store: ZMPStore) -> None:
        proto = default_buffer_prototype()
        buf = await store.get("temp/c/2", proto)
        assert buf is not None
        arr = np.frombuffer(buf.to_bytes(), dtype="<f8")
        np.testing.assert_array_equal(arr, [4.0, 5.0])

    @pytest.mark.asyncio
    async def test_get_missing(self, store: ZMPStore) -> None:
        proto = default_buffer_prototype()
        assert await store.get("nonexistent", proto) is None

    @pytest.mark.asyncio
    async def test_byte_range(self, store: ZMPStore) -> None:
        proto = default_buffer_prototype()
        # RangeByteRequest: get bytes [0, 8) = first float64
        buf = await store.get("temp/c/0", proto, RangeByteRequest(0, 8))
        assert buf is not None
        arr = np.frombuffer(buf.to_bytes(), dtype="<f8")
        np.testing.assert_array_equal(arr, [0.0])

    @pytest.mark.asyncio
    async def test_offset_byte_request(self, store: ZMPStore) -> None:
        proto = default_buffer_prototype()
        buf = await store.get("temp/c/0", proto, OffsetByteRequest(8))
        assert buf is not None
        arr = np.frombuffer(buf.to_bytes(), dtype="<f8")
        np.testing.assert_array_equal(arr, [1.0])

    @pytest.mark.asyncio
    async def test_suffix_byte_request(self, store: ZMPStore) -> None:
        proto = default_buffer_prototype()
        buf = await store.get("temp/c/0", proto, SuffixByteRequest(8))
        assert buf is not None
        arr = np.frombuffer(buf.to_bytes(), dtype="<f8")
        np.testing.assert_array_equal(arr, [1.0])

    @pytest.mark.asyncio
    async def test_list(self, store: ZMPStore) -> None:
        paths = [p async for p in store.list()]
        assert len(paths) == 6

    @pytest.mark.asyncio
    async def test_list_prefix(self, store: ZMPStore) -> None:
        paths = [p async for p in store.list_prefix("temp/c/")]
        assert len(paths) == 4

    @pytest.mark.asyncio
    async def test_list_dir(self, store: ZMPStore) -> None:
        entries = [p async for p in store.list_dir("")]
        assert "zarr.json" in entries
        assert "temp/" in entries

    @pytest.mark.asyncio
    async def test_set_raises(self, store: ZMPStore) -> None:
        proto = default_buffer_prototype()
        with pytest.raises(NotImplementedError):
            await store.set("key", proto.buffer.from_bytes(b"data"))

    @pytest.mark.asyncio
    async def test_delete_raises(self, store: ZMPStore) -> None:
        with pytest.raises(NotImplementedError):
            await store.delete("key")


# --- Integration: open with zarr ---


class TestZarrIntegration:
    @pytest.mark.asyncio
    async def test_open_group(self, simple_zmp: Path) -> None:
        store = ZMPStore.from_file(str(simple_zmp))
        store._is_open = True
        group = zarr.open_group(store=store, mode="r")
        assert "temp" in group

    @pytest.mark.asyncio
    async def test_read_array(self, simple_zmp: Path) -> None:
        store = ZMPStore.from_file(str(simple_zmp))
        store._is_open = True
        group = zarr.open_group(store=store, mode="r")
        arr = group["temp"]
        data = arr[:]
        expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        np.testing.assert_array_equal(data, expected)
