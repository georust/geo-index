# Changelog

**This is the changelog for the core Rust library**. There's a [separate changelog](./python/CHANGELOG.md) for the Python bindings.

## Unreleased

### Breaking

- BREAKING: Introduce `NeighborsOptions` struct and change signatures of `neighbors_with_simple_distance`, `neighbors_with_distance`, `neighbors_coord_with_distance`, and `neighbors_geometry` to accept `NeighborsOptions` instead of separate `max_results`/`max_distance` parameters, and return `Vec<(u32, N)>` instead of `Vec<u32>` to include distances with results. `NeighborsOptions` also adds support for tie breaker inclusion at rank k. by @Kontinuation in https://github.com/georust/geo-index/pull/154

### Added

- Add `RTreeIndex::neighbors_with_callbacks` for ranked queries with caller-defined item distances and bounding-box lower bounds, without enabling a geometry dependency. The existing `use-geo_0_31` APIs delegate to the callback implementation, including distance results and tie handling.

## [0.3.4] - 2026-02-27

### Bug fixes

- Replace custom quicksort with `select_nth_unstable_by` in HilbertSort and STRSort to fix stack overflow with large datasets by @Kontinuation in https://github.com/georust/geo-index/pull/157

## [0.3.3] - 2026-02-11

## Bug fixes

* Fix distance computation in nearest neighbor search when the query window is not a point by @kontinuation in https://github.com/georust/geo-index/pull/151
* Fix panic when performing nearest neighbor search on an empty RTree by @kontinuation in https://github.com/georust/geo-index/pull/151

## [0.3.2] - 2025-11-23

## What's Changed
* chore: Fix lints in clippy 1.89 by @kylebarron in https://github.com/kylebarron/geo-index/pull/132
* Check buffer length before parsing tree headers by @kylebarron in https://github.com/kylebarron/geo-index/pull/131
* Codex format suggestions by @kylebarron in https://github.com/kylebarron/geo-index/pull/133
* perf: Optimize the performance of RTree search by @kylebarron in https://github.com/kylebarron/geo-index/pull/136
* fix: Fix stack overflow in degenerate cases by @kylebarron in https://github.com/kylebarron/geo-index/pull/137
* Bump deps and fix lints by @kylebarron in https://github.com/kylebarron/geo-index/pull/138
* ci: Set up Rust trusted publishing by @kylebarron in https://github.com/kylebarron/geo-index/pull/142
* feat: Add distance metric support for RTree neighbor queries by @zhangfengcdt in https://github.com/kylebarron/geo-index/pull/141

## New Contributors

* @zhangfengcdt made their first contribution in https://github.com/kylebarron/geo-index/pull/141

**Full Changelog**: https://github.com/kylebarron/geo-index/compare/v0.3.1...v0.3.2

## [0.3.1] - 2025-06-20

## Bug fixes

- Fix hanging when building RTree with 0 items by @kontinuation in https://github.com/kylebarron/geo-index/pull/129
- Fix panic when building STR tree with certain number of items by @kontinuation in https://github.com/kylebarron/geo-index/pull/129

## [0.3.0] - 2025-06-13

### Breaking

- Upgrade geo-traits to 0.3 by @kontinuation in https://github.com/kylebarron/geo-index/pull/127
- Update geo to 0.30.0 by @kontinuation in https://github.com/kylebarron/geo-index/pull/127

## [0.2.0] - 2025-01-06

### Breaking

- Use u32 and u16 in public API for num_items and node_size by @kylebarron in https://github.com/kylebarron/geo-index/pull/69
- Rename `OwnedRTree` to `RTree` and `OwnedKDTree` to `KDTree` by @kylebarron in https://github.com/kylebarron/geo-index/pull/81

### Bug fixes

- Fix `intersection_candidates_with_other_tree` by @kylebarron in https://github.com/kylebarron/geo-index/pull/51
- Improve precision in f64 to f32 box cast by @kylebarron in https://github.com/kylebarron/geo-index/pull/76
- Avoid panic for rtree with one item by @kylebarron in https://github.com/kylebarron/geo-index/pull/91

### New Features

- Implement nearest neighbor searches on RTree by @kylebarron in https://github.com/kylebarron/geo-index/pull/79
- Add geo-traits integration by @kylebarron in https://github.com/kylebarron/geo-index/pull/71
  - Implement RectTrait for Node by @kylebarron in https://github.com/kylebarron/geo-index/pull/75
- KDTree traversal by @kylebarron in https://github.com/kylebarron/geo-index/pull/96
- Expose RTreeMetadata & KDTreeMetadata (allowing you to infer the memory usage a tree would incur) by @kylebarron in https://github.com/kylebarron/geo-index/pull/77

### Performance

- Remove unnecessary `Cow` in kdtree trait by @kylebarron in https://github.com/kylebarron/geo-index/pull/72

### Documentation

- Use "immutable" over "static" wording in docs by @kylebarron in https://github.com/kylebarron/geo-index/pull/70
- improved rtree & kdtree docs by @kylebarron in https://github.com/kylebarron/geo-index/pull/93

### What's Changed

- Don't panic for accessing level out of bounds by @kylebarron in https://github.com/kylebarron/geo-index/pull/49

### New Contributors

- @H-Plus-Time made their first contribution in https://github.com/kylebarron/geo-index/pull/55

**Full Changelog**: https://github.com/kylebarron/geo-index/compare/v0.1.1...v0.2.0

## [0.1.1] - 2024-01-14

- Updated benchmarks in documentation by @kylebarron in #27

## [0.1.0] - 2024-01-14

- Initial public release.
