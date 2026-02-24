#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::indices::MutableIndices;
use crate::r#type::IndexableNum;
use crate::rtree::sort::util::swap;
use crate::rtree::sort::{Sort, SortParams};

/// An implementation of sort-tile-recursive (STR) sorting.
///
/// The implementation is derived from [this
/// paper](https://ia600900.us.archive.org/27/items/nasa_techdoc_19970016975/19970016975.pdf).
#[derive(Debug, Clone, Copy)]
pub struct STRSort;

impl<N: IndexableNum> Sort<N> for STRSort {
    fn sort(params: &mut SortParams<N>, boxes: &mut [N], indices: &mut MutableIndices) {
        // We'll reuse the same buffer first for the x coordinate of the centers and then for the y
        // coordinate.
        let mut center_values: Vec<N> = Vec::with_capacity(params.num_items);
        let two = N::from(2).unwrap();

        // Get x value of box centers
        for i in 0..params.num_items {
            let min_x = boxes[i * 4];
            let max_x = boxes[(i * 4) + 2];
            center_values.push((min_x + max_x) / two);
        }

        // Sort items by their x values
        sort(
            &mut center_values,
            boxes,
            indices,
            0,
            params.num_items - 1,
            params.node_size,
        );

        center_values.clear();

        // Get y value of box centers
        for i in 0..params.num_items {
            let min_y = boxes[(i * 4) + 1];
            let max_y = boxes[(i * 4) + 3];
            center_values.push((min_y + max_y) / two);
        }

        let num_leaf_nodes = (params.num_items as f64 / params.node_size as f64).ceil();
        let num_vertical_slices = num_leaf_nodes.sqrt().ceil() as usize;
        let num_items_per_slice =
            (params.num_items as f64 / num_vertical_slices as f64).ceil() as usize;

        #[cfg(feature = "rayon")]
        {
            let center_slices = center_values
                .chunks_mut(num_items_per_slice)
                .collect::<Vec<_>>();
            let boxes_slices = boxes
                .chunks_mut(num_items_per_slice * 4)
                .collect::<Vec<_>>();
            let indices_slices = indices.chunks_mut(num_items_per_slice);

            center_slices
                .into_par_iter()
                .zip(boxes_slices)
                .zip(indices_slices)
                .for_each(|((center_chunk, boxes_chunk), mut indices_chunk)| {
                    // Within each x partition, sort by y values
                    // If the last slice, it won't be a full node
                    let chunk_len = center_chunk.len();
                    sort(
                        center_chunk,
                        boxes_chunk,
                        &mut indices_chunk,
                        0,
                        num_items_per_slice.min(chunk_len) - 1,
                        params.node_size,
                    );
                })
        }

        #[cfg(not(feature = "rayon"))]
        {
            for i in 0..num_vertical_slices {
                let partition_start = i * num_items_per_slice;
                let partition_end = (i + 1) * num_items_per_slice;
                // Within each x partition, sort by y values
                sort(
                    &mut center_values,
                    boxes,
                    indices,
                    partition_start,
                    partition_end.min(params.num_items) - 1,
                    params.node_size,
                );
            }
        }
    }
}

/// Max is only implemented for `Ord` types, but float types do not implement `Ord`.
///
/// So we use this as a hack to get the maximum of two PartialOrd values.
fn partial_ord_max<N: IndexableNum>(a: N, b: N) -> N {
    if a > b {
        a
    } else {
        b
    }
}

/// Custom quicksort that partially sorts bbox data alongside their sort values.
// Partially taken from static_aabb2d_index under the MIT/Apache license
fn sort<N: IndexableNum>(
    values: &mut [N],
    boxes: &mut [N],
    indices: &mut MutableIndices,
    mut left: usize,
    mut right: usize,
    node_size: usize,
) {
    loop {
        debug_assert!(left <= right);

        if left / node_size >= right / node_size {
            return;
        }

        // apply median of three method
        let start = values[left];
        let mid = values[(left + right) >> 1];
        let end = values[right];

        let x = partial_ord_max(start, mid);
        let pivot = if end > x {
            x
        } else if x == start {
            partial_ord_max(mid, end)
        } else if x == mid {
            partial_ord_max(start, end)
        } else {
            end
        };

        let mut i = left.wrapping_sub(1);
        let mut j = right.wrapping_add(1);

        loop {
            loop {
                i = i.wrapping_add(1);
                if values[i] >= pivot {
                    break;
                }
            }

            loop {
                j = j.wrapping_sub(1);
                if values[j] <= pivot {
                    break;
                }
            }

            if i >= j {
                break;
            }

            swap(values, boxes, indices, i, j);
        }

        // Recurse into the smaller partition, loop on the larger one.
        // This bounds the recursion depth to O(log n).
        if j - left < right - j {
            sort(values, boxes, indices, left, j, node_size);
            left = j.wrapping_add(1);
        } else {
            sort(values, boxes, indices, j.wrapping_add(1), right, node_size);
            right = j;
        }
    }
}
