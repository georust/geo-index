use geo_index::rtree::sort::HilbertSort;
use geo_index::rtree::{RTree, RTreeBuilder};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct FlatbushBuilder(RTreeBuilder<f64>);

#[wasm_bindgen]
impl FlatbushBuilder {
    /// Construct a new FlatbushBuilder with number of items and node size
    #[wasm_bindgen(constructor)]
    pub fn new(num_items: u32, node_size: Option<u16>) -> Self {
        let builder = if let Some(node_size) = node_size {
            RTreeBuilder::new_with_node_size(num_items, node_size)
        } else {
            RTreeBuilder::new(num_items)
        };
        Self(builder)
    }

    /// Add a single box at a time to the builder
    ///
    /// This is less efficient than vectorized APIs.
    pub fn add_single_box(&mut self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> u32 {
        self.0.add(min_x, min_y, max_x, max_y)
    }

    /// Add arrays of min_x, min_y, max_x, max_y to the builder
    ///
    /// Each array must be a Float64Array of the same length.
    pub fn add_separated(
        &mut self,
        min_x: &[f64],
        min_y: &[f64],
        max_x: &[f64],
        max_y: &[f64],
    ) -> Vec<u32> {
        assert_eq!(min_x.len(), min_y.len());
        assert_eq!(min_x.len(), max_x.len());
        assert_eq!(min_x.len(), max_y.len());

        let mut out = Vec::with_capacity(min_x.len());
        for i in 0..min_x.len() {
            out.push(self.0.add(min_x[i], min_y[i], max_x[i], max_y[i]));
        }
        out
    }

    /// Finish this builder, sorting the index.
    ///
    /// This will consume this instance and convert it to a [Flatbush].
    pub fn finish(self) -> Flatbush {
        Flatbush(self.0.finish::<HilbertSort>())
    }
}

#[wasm_bindgen]
pub struct Flatbush(RTree<f64>);

#[wasm_bindgen]
impl Flatbush {
    /// The byte offset within WebAssembly memory where the Flatbush memory starts.
    #[wasm_bindgen]
    pub fn byte_offset(&self) -> *const u8 {
        self.0.as_ref().as_ptr()
    }

    /// The number of bytes in Flatbush memory
    #[wasm_bindgen]
    pub fn byte_length(&self) -> usize {
        self.0.as_ref().len()
    }
}
