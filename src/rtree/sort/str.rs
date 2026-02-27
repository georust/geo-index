use std::cmp::Ordering;
use std::ops::Range;

#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::indices::MutableIndices;
use crate::r#type::IndexableNum;
use crate::rtree::sort::util::{apply_permutation, k_block_sort_by};
use crate::rtree::sort::{Sort, SortParams};

/// An implementation of sort-tile-recursive (STR) sorting.
///
/// The implementation is derived from [this
/// paper](https://ia600900.us.archive.org/27/items/nasa_techdoc_19970016975/19970016975.pdf).
#[derive(Debug, Clone, Copy)]
pub struct STRSort;

impl<N: IndexableNum> Sort<N> for STRSort {
    fn sort(params: &mut SortParams<N>, boxes: &mut [N], indices: &mut MutableIndices) {
        let two = N::from(2).unwrap();

        // Compute the number of vertical slices to create, and the number of items per slice.
        // The number of items per slice must be multiple of the node size to ensure that
        // no nodes will span multiple slices when we group items into nodes later.
        let num_leaf_nodes = (params.num_items as f64 / params.node_size as f64).ceil();
        let num_items_per_slice = {
            let num_vertical_slices = num_leaf_nodes.sqrt().ceil() as usize;
            num_vertical_slices * params.node_size
        };

        // We'll reuse the same buffer first for the x coordinate of the centers and then for the y
        // coordinate.
        let mut center_with_indices: Vec<(N, u32)> = Vec::with_capacity(params.num_items);

        // Get x value of box centers
        for i in 0..params.num_items {
            let min_x = boxes[i * 4];
            let max_x = boxes[(i * 4) + 2];
            let center_x = (min_x + max_x) / two;
            center_with_indices.push((center_x, 0));
        }

        // Sort items by their x values
        sort(
            &mut center_with_indices,
            boxes,
            indices,
            0..params.num_items,
            num_items_per_slice,
        );

        center_with_indices.clear();

        // Get y value of box centers
        for i in 0..params.num_items {
            let min_y = boxes[(i * 4) + 1];
            let max_y = boxes[(i * 4) + 3];
            let center_y = (min_y + max_y) / two;
            center_with_indices.push((center_y, 0));
        }

        #[cfg(feature = "rayon")]
        {
            let center_slices = center_with_indices
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
                        0..chunk_len,
                        params.node_size,
                    );
                })
        }

        #[cfg(not(feature = "rayon"))]
        {
            for partition_start in (0..params.num_items).step_by(num_items_per_slice) {
                let partition_end = (partition_start + num_items_per_slice).min(params.num_items);
                // Within each x partition, sort by y values
                sort(
                    &mut center_with_indices,
                    boxes,
                    indices,
                    partition_start..partition_end,
                    params.node_size,
                );
            }
        }
    }
}

/// Sorts the given range of items in `center_with_indices` and applies the same permutation to
/// `boxes` and `indices`.
fn sort<N: IndexableNum>(
    center_with_indices: &mut [(N, u32)],
    boxes: &mut [N],
    indices: &mut MutableIndices,
    range: Range<usize>,
    node_size: usize,
) {
    let center_with_indices = &mut center_with_indices[range.clone()];
    for (idx, val) in center_with_indices.iter_mut().enumerate() {
        val.1 = idx as u32;
    }

    let boxes = &mut boxes[range.start * 4..range.end * 4];
    let mut indices = match indices {
        MutableIndices::U16(arr) => MutableIndices::U16(&mut arr[range]),
        MutableIndices::U32(arr) => MutableIndices::U32(&mut arr[range]),
    };

    k_block_sort_by(center_with_indices, node_size, |&a, &b| {
        a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal)
    });
    apply_permutation(center_with_indices, boxes, &mut indices);
}
