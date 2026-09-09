# geo-index

[![crates.io version][crates.io_badge]][crates.io_link]
[![docs.rs docs][docs.rs_badge]][docs.rs_link]
[![PyPI][pypi_badge]][pypi_link]

[crates.io_badge]: https://img.shields.io/crates/v/geo-index.svg
[crates.io_link]: https://crates.io/crates/geo-index
[docs.rs_badge]: https://docs.rs/geo-index/badge.svg
[docs.rs_link]: https://docs.rs/geo-index
[pypi_badge]: https://badge.fury.io/py/geoindex-rs.svg
[pypi_link]: https://pypi.org/project/geoindex-rs/

A Rust crate and [Python library](https://kylebarron.dev/geo-index/latest/) for packed, immutable, zero-copy spatial indexes.

## Features

- **An R-tree and k-d tree written in safe rust.**
- **Fast.** Because of optimizations available by using immutable indexes, tends to be faster than dynamic implementations like [`rstar`](https://github.com/georust/rstar).
- **Memory efficient.** The index is fully _packed_, meaning that all nodes are at full capacity (except for the last node at each tree level). This means the RTree and k-d tree use less memory. And because the index is backed by a single buffer, it exhibits excellent memory locality.
- **Bounded memory**. For any given number of items and node size, you can infer the total memory used by the RTree or KDTree.
- **Multiple R-tree sorting methods.** Currently, [hilbert](https://en.wikipedia.org/wiki/Hilbert_R-tree) and [sort-tile-recursive (STR)](https://ia600900.us.archive.org/27/items/nasa_techdoc_19970016975/19970016975.pdf) sorting methods are implemented, but it's extensible to other spatial sorting algorithms in the future, like [overlap-minimizing top-down (OMT)](https://ceur-ws.org/Vol-74/files/FORUM_18.pdf).
- **ABI-stable:** the index is contained in a single `Vec<u8>`, compatible with the [`flatbush`](https://github.com/mourner/flatbush) and [`kdbush`](https://github.com/mourner/kdbush) JavaScript libraries. Being ABI-stable means that the spatial index can be persisted for later use or shared zero-copy between Rust and another program like Python.
- **Generic over a set of coordinate types:** `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `f32`, `f64`. These are the only coordinate types currently supported to maintain binary compatibility with the upstream JS libraries.
- **Efficient bulk loading.** As an immutable index, _only_ bulk loading is supported.
- Optional `rayon` feature for parallelizing part of the sort in the sort-tile-recursive (`STRSort`) method.

## Custom ranked queries

`RTreeIndex::neighbors_with_callbacks` ranks items using two callbacks: a lower bound on distance to an internal bounding box, and an exact distance to an item identified by its insertion index. The callbacks can capture any query or geometry storage and use planar, spherical, or other distance calculations. This API requires no optional feature or additional dependency. See its Rust API documentation for an example and the lower-bound requirements for correct ranking and pruning.

The `use-geo_0_31` feature and its existing distance APIs remain available for compatibility. New integrations can use the callbacks with their preferred geometry library and version.

## Drawbacks

- Trees are _immutable_. After creating the index, items can no longer be added or removed.
- Only two-dimensional indexes is supported. This can still be used with higher-dimensional input data as long as it's fine to only index two of the dimensions.
- Queries return insertion indexes into the input set, so you must manage your own collections.
- Only the set of coordinate types that exist in JavaScript are allowed, to maintain FFI compatibility with the reference JavaScript implementations. Hence this does not support other types like `u64`.

## Alternative crates

- [`rstar`](https://github.com/georust/rstar): a dynamic RTree implementation.
- [`kdtree`](https://github.com/mrhooray/kdtree-rs): a dynamic KDTree implementation.
- [`kdbush`](https://github.com/pka/rust-kdbush): a port of `kdbush` but does not strive for FFI-compatibility.
- [`static_aabb2d_index`](https://github.com/jbuckmccready/static_aabb2d_index): a port of `flatbush` but does not strive for FFI-compatibility.
- [`static-bushes`](https://github.com/apendleton/static-bushes): a port of `flatbush` and `kdbush` but does not strive for FFI-compatibility.

## Inspiration

[@mourner](https://github.com/mourner)'s amazing [`flatbush`](https://github.com/mourner/flatbush) and [`kdbush`](https://github.com/mourner/kdbush) libraries are the fastest JavaScript libraries for immutable R-trees and k-d trees. Part of their appeal in the browser is that they're backed by a single, contiguous buffer, and thus can be moved from the main thread to another thread (a [web worker](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API)) [without any copies](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Transferable_objects).

By porting and expanding on those JavaScript libraries and ensuring that the internal memory layout is _exactly_ maintained, we can bootstrap zero-copy use cases both inside and outside the browser. In-browser use cases can interop between a rust-based WebAssembly module and the upstream JS libraries _without copies_. Outside-browser use cases can interop between multiple Rust libraries or between Rust and Python without copies.

## Why zero-copy?

I'm excited about Rust to speed up Python and JavaScript via compiled extension modules. It's true that you can create Python bindings to a Rust library, have Rust manage the memory, and never need to worry about zero-copy data structures. But when someone else writes a C library that would like to interface with your data, if you don't have an ABI-stable way to share the data, you need to serialize it and they need to deserialize it, which is costly.

For example, in Python, Shapely (and by extension the C library GEOS) is used for most geospatial data storage. But separate Python libraries with C extensions can't use the same GEOS memory because the underlying storage isn't ABI-stable. So there has to be a serde step in between.

[GeoArrow](https://geoarrow.org/) solves this problem for geospatial vector data, because it defines a language-independent, ABI-stable memory layout. So you can safely move memory between Python/Rust/C just by exchanging pointer information.

But it's very useful to be able to share large spatial data, declare that the data is already spatially ordered, _and_ share a spatial index, all at no cost.

Currently, this library is used under the hood in [`geoarrow-rs`](https://github.com/geoarrow/geoarrow-rs) to speed up boolean operations and spatial joins. But over a medium-term time horizon, I believe that exposing the raw index data will enable exciting FFI use cases that are presently impossible.

## Future work

- Built-in geographic query metrics and bounding-box lower bounds. Custom ranked queries can already supply these through callbacks.

## Benchmarks

This is just _one_ benchmark; I recommend benchmarking with your own data, but this indicates construction is ~2x faster than `rstar` and search is ~33% faster.

```ignore
cargo bench --bench rtree
```

```ignore
construction (geo-index, hilbert)
                        time:   [80.503 ms 80.891 ms 81.350 ms]
construction (geo-index, STRTree)
                        time:   [115.60 ms 116.52 ms 117.64 ms]
construction (geo-index, hilbert, f64 to f32, including casting)
                        time:   [86.409 ms 86.681 ms 86.984 ms]
construction (geo-index f32)
                        time:   [78.292 ms 78.393 ms 78.514 ms]
construction (rstar bulk)
                        time:   [158.48 ms 159.34 ms 160.29 ms]

search (flatbush)       time:   [115.97 µs 116.41 µs 116.86 µs]
search (flatbush STRTree)
                        time:   [115.85 µs 117.57 µs 118.95 µs]
search (flatbush f32)   time:   [113.04 µs 114.56 µs 115.99 µs]
search (rstar)          time:   [151.53 µs 153.62 µs 155.84 µs]
```

With the `rayon` feature, the sorting phase of the `STRTree` is faster:

```ignore
cargo bench --bench rtree --features rayon
```

```ignore
construction (geo-index, STRTree)
                        time:   [71.825 ms 72.099 ms 72.382 ms]
                        change: [-38.738% -38.125% -37.570%] (p = 0.00 < 0.05)
                        Performance has improved.
```
