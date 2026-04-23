# WebAssembly bindings to geo-index

Example WebAssembly bindings for the Rust [`geo-index`](https://github.com/kylebarron/geo-index) crate.

This is not expected to be used _directly_ for JavaScript applications that only need an RTree or KDTree. In those cases, using [flatbush](https://github.com/mourner/flatbush) or [kdbush](https://github.com/mourner/kdbush) directly would probably give similar performance while being easier to use and smaller code size.

However this is designed to show _how the underlying data can be used zero-copy_ between WebAssembly and JavaScript. A larger application could have some use for generating the RTree index in Rust in WebAssembly but then allowing JavaScript to view it and run queries on it.

## Building

```
wasm-pack build --target web
```
