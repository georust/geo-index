use std::cmp::Ordering;

use crate::indices::MutableIndices;
use crate::IndexableNum;

/// Partition `arr` into blocks of size `k` such that all elements in block `i`
/// are ≤ all elements in block `i+1`.
///
/// Uses [`select_nth_unstable_by`] under the hood, which is O(n) per level.
/// Total work: O(n log(n/k)). Max recursion depth: ceil(log2(n/k)).
///
/// For n = 11M and k = 16, recursion depth is ~20 — completely safe.
///
/// [`select_nth_unstable_by`]: slice::select_nth_unstable_by
pub(super) fn k_block_sort_by<T, F>(arr: &mut [T], k: usize, compare: F)
where
    F: Fn(&T, &T) -> Ordering + Copy,
{
    if k == 0 || arr.len() <= k {
        return;
    }

    let len = arr.len();
    let num_blocks = len / k;

    if num_blocks < 2 {
        // One full block + a partial remainder. Partition so the first k elements
        // are the smallest block.
        arr.select_nth_unstable_by(k, |a, b| compare(a, b));
        return;
    }

    let mid_block = num_blocks / 2;
    let mid_idx = mid_block * k;
    arr.select_nth_unstable_by(mid_idx, |a, b| compare(a, b));

    let (left, right) = arr.split_at_mut(mid_idx);
    k_block_sort_by(left, k, compare);
    k_block_sort_by(right, k, compare);
}

/// Apply a permutation to `boxes` and `indices` using cycle-following.
///
/// `perm[i]` is the original index of the element that should end up at position `i`.
/// After this call, `boxes` and `indices` are reordered according to `perm`,
/// and `perm` is consumed (set to the identity).
///
/// Time: O(n), each element moved exactly once.
/// Extra space: O(1) beyond the `perm` array.
pub(super) fn apply_permutation<N: IndexableNum>(
    perm: &mut [u32],
    boxes: &mut [N],
    indices: &mut MutableIndices,
) {
    let n = perm.len();
    for i in 0..n {
        if perm[i] as usize == i {
            continue; // already in place
        }

        // Save the item currently at position i
        let bi = i * 4;
        let saved_box = [boxes[bi], boxes[bi + 1], boxes[bi + 2], boxes[bi + 3]];
        let saved_index = indices.get(i);

        let mut j = i;
        loop {
            let k = perm[j] as usize;
            perm[j] = j as u32; // mark as done

            if k == i {
                // End of cycle: place the saved item at position j
                let bj = j * 4;
                boxes[bj] = saved_box[0];
                boxes[bj + 1] = saved_box[1];
                boxes[bj + 2] = saved_box[2];
                boxes[bj + 3] = saved_box[3];
                indices.set(j, saved_index);
                break;
            }

            // Move item from position k to position j
            let bj = j * 4;
            let bk = k * 4;
            boxes[bj] = boxes[bk];
            boxes[bj + 1] = boxes[bk + 1];
            boxes[bj + 2] = boxes[bk + 2];
            boxes[bj + 3] = boxes[bk + 3];
            indices.set(j, indices.get(k));

            j = k;
        }
    }
}
