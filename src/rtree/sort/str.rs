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

        // --- Phase 1: Sort all items by x-center ---
        {
            // Compute x-center values
            let center_x: Vec<N> = (0..params.num_items)
                .map(|i| {
                    let min_x = boxes[i * 4];
                    let max_x = boxes[(i * 4) + 2];
                    (min_x + max_x) / two
                })
                .collect();

            // Build index array [0, 1, 2, ..., n-1]
            let mut order: Vec<u32> = (0..params.num_items as u32).collect();

            // k-block sort indices by x-center
            k_block_sort_by(&mut order, params.node_size, |&a, &b| {
                partial_cmp_unwrap(center_x[a as usize], center_x[b as usize])
            });

            // Apply permutation to boxes and indices
            apply_permutation(&mut order, boxes, indices);
        }

        // --- Phase 2: Within each vertical slice, sort by y-center ---
        let num_leaf_nodes = (params.num_items as f64 / params.node_size as f64).ceil();
        let num_vertical_slices = num_leaf_nodes.sqrt().ceil() as usize;
        let num_items_per_slice =
            (params.num_items as f64 / num_vertical_slices as f64).ceil() as usize;

        // Chunk ONLY the item portion of boxes and indices into slices.
        // boxes/indices contain tree node data beyond num_items that must not be touched.
        let item_boxes = &mut boxes[..params.num_items * 4];
        let box_chunks: Vec<&mut [N]> = item_boxes.chunks_mut(num_items_per_slice * 4).collect();
        let (mut item_indices, _) = indices.split_at_mut(params.num_items);
        let index_chunks = item_indices.chunks_mut(num_items_per_slice);

        let node_size = params.node_size;

        #[cfg(feature = "rayon")]
        {
            box_chunks.into_par_iter().zip(index_chunks).for_each(
                |(box_chunk, mut index_chunk)| {
                    sort_slice_by_y(box_chunk, &mut index_chunk, node_size, two);
                },
            );
        }

        #[cfg(not(feature = "rayon"))]
        {
            for (box_chunk, mut index_chunk) in box_chunks.into_iter().zip(index_chunks) {
                sort_slice_by_y(box_chunk, &mut index_chunk, node_size, two);
            }
        }
    }
}

/// Sort a single vertical slice by y-center using k-block sort + permutation.
fn sort_slice_by_y<N: IndexableNum>(
    box_chunk: &mut [N],
    index_chunk: &mut MutableIndices,
    node_size: usize,
    two: N,
) {
    let slice_items = box_chunk.len() / 4;
    if slice_items <= 1 {
        return;
    }

    // Compute y-center values for this slice
    let center_y: Vec<N> = (0..slice_items)
        .map(|j| {
            let min_y = box_chunk[(j * 4) + 1];
            let max_y = box_chunk[(j * 4) + 3];
            (min_y + max_y) / two
        })
        .collect();

    // Build local index array
    let mut order: Vec<u32> = (0..slice_items as u32).collect();

    // k-block sort by y-center
    k_block_sort_by(&mut order, node_size, |&a, &b| {
        partial_cmp_unwrap(center_y[a as usize], center_y[b as usize])
    });

    // Apply permutation to box and index slices
    apply_permutation(&mut order, box_chunk, index_chunk);
}

/// Compare two `PartialOrd` values, treating incomparable (NaN) as equal.
#[inline]
fn partial_cmp_unwrap<N: PartialOrd>(a: N, b: N) -> std::cmp::Ordering {
    a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
}
