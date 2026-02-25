use std::fs::read;

use bytemuck::cast_slice;

use crate::rtree::sort::HilbertSort;
use crate::rtree::RTreeIndex;
use crate::rtree::{RTree, RTreeBuilder, RTreeRef};

fn create_flatbush_from_data_path(data_path: &str) -> RTree<f64> {
    let buffer = read(data_path).unwrap();
    let boxes_buf: &[f64] = cast_slice(&buffer);

    let mut builder = RTreeBuilder::new((boxes_buf.len() / 4) as _);
    for box_ in boxes_buf.chunks(4) {
        let min_x = box_[0];
        let min_y = box_[1];
        let max_x = box_[2];
        let max_y = box_[3];
        builder.add(min_x, min_y, max_x, max_y);
    }
    builder.finish::<HilbertSort>()
}

pub(crate) fn flatbush_js_test_data() -> Vec<f64> {
    let buffer = read("fixtures/data1_input.raw").unwrap();
    let boxes_buf: &[f64] = cast_slice(&buffer);
    boxes_buf.to_vec()
}

pub(crate) fn flatbush_js_test_index() -> RTree<f64> {
    create_flatbush_from_data_path("fixtures/data1_input.raw")
}

/// Brute-force search: scan all input boxes and return indices that intersect the query box.
fn brute_force_search(boxes: &[f64], min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<u32> {
    let mut results = Vec::new();
    for (i, box_) in boxes.chunks(4).enumerate() {
        let b_min_x = box_[0];
        let b_min_y = box_[1];
        let b_max_x = box_[2];
        let b_max_y = box_[3];
        if max_x >= b_min_x && max_y >= b_min_y && min_x <= b_max_x && min_y <= b_max_y {
            results.push(i as u32);
        }
    }
    results
}

#[test]
fn test_flatbush_js_test_data() {
    let input_data = flatbush_js_test_data();
    let num_items = input_data.len() / 4;

    // Build the tree
    let tree = create_flatbush_from_data_path("fixtures/data1_input.raw");
    let tree_buf = tree.into_inner();

    // Verify the tree can be deserialized
    let tree_ref = RTreeRef::<f64>::try_new(&tree_buf).unwrap();

    // Verify metadata
    assert_eq!(tree_ref.num_items() as usize, num_items);

    // Also verify against the JS reference: same buffer length, same header, same metadata
    let js_buf = read("fixtures/data1_flatbush_js.raw").unwrap();
    assert_eq!(
        tree_buf.len(),
        js_buf.len(),
        "Tree buffer length should match JS reference"
    );
    assert_eq!(
        &tree_buf[..8],
        &js_buf[..8],
        "Header bytes should match JS reference"
    );
    let js_ref = RTreeRef::<f64>::try_new(&js_buf).unwrap();
    assert_eq!(
        tree_ref.metadata, js_ref.metadata,
        "Tree metadata should match JS reference"
    );

    // Functional correctness: search results should match brute-force for several query boxes
    let query_boxes: &[(f64, f64, f64, f64)] = &[
        (0.0, 0.0, 1000.0, 1000.0),   // everything
        (0.0, 0.0, 10.0, 10.0),       // small corner
        (40.0, 40.0, 60.0, 60.0),     // middle region
        (100.0, 100.0, 200.0, 200.0), // larger region
        (200.0, 200.0, 300.0, 300.0), // upper region
        (500.0, 500.0, 600.0, 600.0), // should be empty for this dataset
        (10.0, 20.0, 80.0, 90.0),     // arbitrary rectangle
    ];

    for &(min_x, min_y, max_x, max_y) in query_boxes {
        let mut tree_results = tree_ref.search(min_x, min_y, max_x, max_y);
        let mut brute_results = brute_force_search(&input_data, min_x, min_y, max_x, max_y);
        tree_results.sort();
        brute_results.sort();
        assert_eq!(
            tree_results, brute_results,
            "Search results mismatch for query box ({}, {}, {}, {})",
            min_x, min_y, max_x, max_y
        );
    }
}
