use crate::indices::MutableIndices;
use crate::r#type::IndexableNum;
use crate::rtree::sort::util::{apply_permutation, k_block_sort_by};
use crate::rtree::sort::{Sort, SortParams};

/// An implementation of hilbert sorting.
///
/// The implementation is ported from the original [flatbush](https://github.com/mourner/flatbush)
/// JavaScript library. The hilbert calculations are originally derived from [a C++
/// implementation](https://github.com/rawrunprotected/hilbert_curves).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HilbertSort;

impl<N: IndexableNum> Sort<N> for HilbertSort {
    fn sort(params: &mut SortParams<N>, boxes: &mut [N], indices: &mut MutableIndices) {
        let width = params.max_x - params.min_x;
        let height = params.max_y - params.min_y;
        let hilbert_max = ((1 << 16) - 1) as f64;

        // Build packed tuples as (hilbert_value, original_index).
        let mut hilbert_values_with_indices: Vec<(u32, u32)> = Vec::with_capacity(params.num_items);
        {
            let mut pos = 0;
            for i in 0..params.num_items {
                let min_x = boxes[pos];
                pos += 1;
                let min_y = boxes[pos];
                pos += 1;
                let max_x = boxes[pos];
                pos += 1;
                let max_y = boxes[pos];
                pos += 1;

                let x = (hilbert_max
                    * ((min_x + max_x).to_f64().unwrap() / 2. - params.min_x.to_f64().unwrap())
                    / width.to_f64().unwrap())
                .floor() as u32;
                let y = (hilbert_max
                    * ((min_y + max_y).to_f64().unwrap() / 2. - params.min_y.to_f64().unwrap())
                    / height.to_f64().unwrap())
                .floor() as u32;

                hilbert_values_with_indices.push((hilbert(x, y), (i as u32)));
            }
        }

        // sort items by their Hilbert value (for packing later)
        k_block_sort_by(
            &mut hilbert_values_with_indices,
            params.node_size,
            |a, b| a.0.cmp(&b.0),
        );
        apply_permutation(hilbert_values_with_indices.as_mut_slice(), boxes, indices);
    }
}

// Taken from static_aabb2d_index under the mit/apache license
// https://github.com/jbuckmccready/static_aabb2d_index/blob/9e6add59d77b74d4de0ac32159db47fbcb3acc28/src/static_aabb2d_index.rs#L486C1-L544C2
#[inline]
fn hilbert(x: u32, y: u32) -> u32 {
    // Fast Hilbert curve algorithm by http://threadlocalmutex.com/
    // Ported from C++ https://github.com/rawrunprotected/hilbert_curves (public domain)
    let mut a_1 = x ^ y;
    let mut b_1 = 0xFFFF ^ a_1;
    let mut c_1 = 0xFFFF ^ (x | y);
    let mut d_1 = x & (y ^ 0xFFFF);

    let mut a_2 = a_1 | (b_1 >> 1);
    let mut b_2 = (a_1 >> 1) ^ a_1;
    let mut c_2 = ((c_1 >> 1) ^ (b_1 & (d_1 >> 1))) ^ c_1;
    let mut d_2 = ((a_1 & (c_1 >> 1)) ^ (d_1 >> 1)) ^ d_1;

    a_1 = a_2;
    b_1 = b_2;
    c_1 = c_2;
    d_1 = d_2;
    a_2 = (a_1 & (a_1 >> 2)) ^ (b_1 & (b_1 >> 2));
    b_2 = (a_1 & (b_1 >> 2)) ^ (b_1 & ((a_1 ^ b_1) >> 2));
    c_2 ^= (a_1 & (c_1 >> 2)) ^ (b_1 & (d_1 >> 2));
    d_2 ^= (b_1 & (c_1 >> 2)) ^ ((a_1 ^ b_1) & (d_1 >> 2));

    a_1 = a_2;
    b_1 = b_2;
    c_1 = c_2;
    d_1 = d_2;
    a_2 = (a_1 & (a_1 >> 4)) ^ (b_1 & (b_1 >> 4));
    b_2 = (a_1 & (b_1 >> 4)) ^ (b_1 & ((a_1 ^ b_1) >> 4));
    c_2 ^= (a_1 & (c_1 >> 4)) ^ (b_1 & (d_1 >> 4));
    d_2 ^= (b_1 & (c_1 >> 4)) ^ ((a_1 ^ b_1) & (d_1 >> 4));

    a_1 = a_2;
    b_1 = b_2;
    c_1 = c_2;
    d_1 = d_2;
    c_2 ^= (a_1 & (c_1 >> 8)) ^ (b_1 & (d_1 >> 8));
    d_2 ^= (b_1 & (c_1 >> 8)) ^ ((a_1 ^ b_1) & (d_1 >> 8));

    a_1 = c_2 ^ (c_2 >> 1);
    b_1 = d_2 ^ (d_2 >> 1);

    let mut i0 = x ^ y;
    let mut i1 = b_1 | (0xFFFF ^ (i0 | a_1));

    i0 = (i0 | (i0 << 8)) & 0x00FF_00FF;
    i0 = (i0 | (i0 << 4)) & 0x0F0F_0F0F;
    i0 = (i0 | (i0 << 2)) & 0x3333_3333;
    i0 = (i0 | (i0 << 1)) & 0x5555_5555;

    i1 = (i1 | (i1 << 8)) & 0x00FF_00FF;
    i1 = (i1 | (i1 << 4)) & 0x0F0F_0F0F;
    i1 = (i1 | (i1 << 2)) & 0x3333_3333;
    i1 = (i1 | (i1 << 1)) & 0x5555_5555;

    (i1 << 1) | i0
}
