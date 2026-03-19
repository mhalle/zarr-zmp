# Changelog

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
