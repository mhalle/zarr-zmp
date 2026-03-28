# Changelog

## v0.16.0 (2026-03-28)

### Features

- **NRRDStore**: Read-only zarr v3 store wrapping NRRD files. Raw
  encoding gives chunked random access; compressed uses zarr codecs.
  NRRD metadata preserved in `attributes.nrrd`.
- **Streaming writer**: `ZMPWritableStore` streams chunk data to disk
  during `set()` instead of buffering in memory.
- **Edge chunk padding**: Virtual references with `content_encoding`
  are padded to full chunk size for zarr compatibility.
- **Cleaned up base_resolve**: `from_file()` skips location base when
  manifest has its own `base_resolve`.

### Dependencies

- Requires `zmanifest >= 0.16.0`.

## v0.12.0 (2026-03-22)

### Breaking changes

- Requires `zmanifest >= 0.12.0` (absolute paths, adaptive row groups).

### Changes

- **ZPath throughout**: All internal path logic uses `ZPath` from
  zmanifest instead of raw string manipulation with `/`. The zarr Store
  boundary converts with `ZPath.from_zarr()` (input) and `.to_zarr()`
  (output).
- **Manifest paths are absolute**: `ZMPStore` reads `/zarr.json` from the
  manifest and yields `zarr.json` to zarr. Old manifests (bare paths)
  still work — zmanifest normalizes on load.
- **Conftest uses Builder**: Test fixtures use `zmanifest.Builder` instead
  of raw pyarrow table construction.
- **zmanifest test separation**: Manifest and Builder tests moved to
  zmanifest. zarr-zmp retains only zarr Store interface tests (40 tests).

## v0.1.0 (2026-03-19)

Initial release of zarr-zmp — Zarr v3 store adapter for zmanifest files.

### Features

- **`ZMPStore`**: Read-only Zarr v3 `Store` backed by a `.zmp` manifest, with mount support, relative URI resolution, and pluggable `mount_opener` callback
- **`ZMPWritableStore`**: Writable Zarr v3 `Store` that buffers writes and flushes to a `.zmp` file (embedded or external blob mode)
- **`build_zmp()` / `async_build_zmp()`**: Build a self-contained `.zmp` from any Zarr store
- **`default_zmp_mount_opener()`**: Default mount opener for `.zmp` and `.zarr.zip` targets, with resolver and base_uri inheritance
- **Resolver inheritance**: Child mounts inherit the parent's blob resolver, mount opener, and base URI
- **Zarr metadata canonicalization**: JSON metadata is canonicalized via RFC 8785 before hashing for deterministic retrieval keys
- **Byte range support**: `RangeByteRequest`, `OffsetByteRequest`, `SuffixByteRequest`
- **Re-exports**: All core `zmanifest` symbols (`Manifest`, `Builder`, `Addressing`, resolvers, etc.) available from `zarr_zmp`

### Dependencies

- `zmanifest >= 0.1.0`
- `zarr >= 3.0`
