mod rtree;
mod utils;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet() {
    alert("Hello, flatbush-wasm!");
}

/// Returns a handle to this wasm instance's `WebAssembly.Memory`
#[wasm_bindgen(js_name = wasmMemory)]
pub fn memory() -> JsValue {
    wasm_bindgen::memory()
}
