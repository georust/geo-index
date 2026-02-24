use crate::indices::MutableIndices;
use crate::r#type::IndexableNum;
use crate::rtree::sort::util::swap;
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
        let width = params.max_x - params.min_x; // || 1.0;
        let height = params.max_y - params.min_y; // || 1.0;
        let mut hilbert_values: Vec<u32> = Vec::with_capacity(params.num_items);
        let hilbert_max = ((1 << 16) - 1) as f64;

        {
            // map item centers into Hilbert coordinate space and calculate Hilbert values
            let mut pos = 0;
            for _ in 0..params.num_items {
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
                hilbert_values.push(hilbert(x, y));
            }
        }

        // sort items by their Hilbert value (for packing later)
        sort(
            &mut hilbert_values,
            boxes,
            indices,
            0,
            params.num_items - 1,
            params.node_size,
        );
    }
}

/// Custom quicksort that partially sorts bbox data alongside the hilbert values.
// Partially taken from static_aabb2d_index under the MIT/Apache license
fn sort<N: IndexableNum>(
    values: &mut [u32],
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

        let x = start.max(mid);
        let pivot = if end > x {
            x
        } else if x == start {
            mid.max(end)
        } else if x == mid {
            start.max(end)
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
