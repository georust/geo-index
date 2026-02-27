use std::cmp::Ordering;

use crate::indices::MutableIndices;
use crate::IndexableNum;

/// Partition `arr` into blocks of size `k` such that all elements in block `i`
/// are ≤ all elements in block `i+1`.
///
/// Uses [`select_nth_unstable_by`] under the hood, which is O(n) per level.
/// Total work: O(n log(n/k)). Max recursion depth: ceil(log2(n/k)).
/// This is a Rust standard-library approach to an upstream update that migrated
/// the Hilbert sort to a non-recursive algorithm: https://github.com/mourner/flatbush/pull/70
///
/// [`select_nth_unstable_by`]: slice::select_nth_unstable_by
pub(super) fn k_block_sort_by<T, F>(arr: &mut [T], k: usize, mut compare: F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    assert!(k > 0, "k must be positive");
    if k == 1 {
        // No need to partition into blocks if block size is 1, just sort the whole array.
        arr.sort_unstable_by(compare);
    } else {
        do_k_block_sort_by(arr, k, &mut compare);
    }
}

fn do_k_block_sort_by<T, F>(arr: &mut [T], k: usize, compare: &mut F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    if arr.len() <= k {
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
    do_k_block_sort_by(left, k, compare);
    do_k_block_sort_by(right, k, compare);
}

/// Abstraction over permutation storage used by [`apply_permutation`].
///
/// Implementations represent a mapping where position `i` stores the original
/// index of the element that should be moved to `i`.
///
/// This allows reusing the same cycle-following permutation logic for both a
/// plain permutation buffer (`Vec<u32>`) and packed sort buffers
/// (`Vec<(u32, u32)>`) without extra allocation.
pub(super) trait PermIndices {
    fn len(&self) -> usize;
    fn get(&self, i: usize) -> u32;
}

impl PermIndices for &mut [u32] {
    #[inline]
    fn len(&self) -> usize {
        <[u32]>::len(self)
    }

    #[inline]
    fn get(&self, i: usize) -> u32 {
        self[i]
    }
}

impl<T> PermIndices for &mut [(T, u32)] {
    #[inline]
    fn len(&self) -> usize {
        <[(T, u32)]>::len(self)
    }

    #[inline]
    fn get(&self, i: usize) -> u32 {
        self[i].1
    }
}

/// Apply a permutation to `boxes` and `indices` using cycle-following.
///
/// `perm[i]` is the original index of the element that should end up at position `i`.
/// After this call, `boxes` and `indices` are reordered according to `perm`,
/// and `perm` is consumed (set to the identity).
///
/// Time: O(n), each element moved exactly once.
/// Extra space: O(1) beyond the `perm` array.
pub(super) fn apply_permutation<N: IndexableNum, P: PermIndices>(
    perm: P,
    boxes: &mut [N],
    indices: &mut MutableIndices,
) {
    match indices {
        MutableIndices::U16(indices) => do_apply_permutation(perm, boxes, indices),
        MutableIndices::U32(indices) => do_apply_permutation(perm, boxes, indices),
    }
}

fn do_apply_permutation<N: IndexableNum, I: Copy, P: PermIndices>(
    perm: P,
    boxes: &mut [N],
    indices: &mut [I],
) {
    let n = perm.len();

    let mut new_boxes: Vec<N> = Vec::with_capacity(n * 4);
    let mut new_indices: Vec<I> = Vec::with_capacity(n);

    for i in 0..n {
        let src = perm.get(i) as usize;
        let bs = src * 4;
        new_boxes.push(boxes[bs]);
        new_boxes.push(boxes[bs + 1]);
        new_boxes.push(boxes[bs + 2]);
        new_boxes.push(boxes[bs + 3]);
        new_indices.push(indices[src]);
    }

    boxes[..n * 4].copy_from_slice(&new_boxes);
    indices[..n].copy_from_slice(&new_indices);
}

#[cfg(test)]
mod test {
    use std::fmt::Debug;

    use super::*;
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    /// Assert the k-block invariant: max of block i <= min of block i+1.
    fn assert_k_block_invariant(arr: &[u32], k: usize) {
        if arr.len() <= k {
            return;
        }
        let num_blocks = arr.len().div_ceil(k);

        for b in 0..num_blocks - 1 {
            let block_start = b * k;
            let block_end = ((b + 1) * k).min(arr.len());
            let next_start = block_end;
            let next_end = ((b + 2) * k).min(arr.len());

            let block_max = arr[block_start..block_end].iter().copied().max().unwrap();
            let next_min = arr[next_start..next_end].iter().copied().min().unwrap();

            assert!(
                block_max <= next_min,
                "k-block invariant violated: block {} max ({}) > block {} min ({}), k={}, arr={:?}",
                b,
                block_max,
                b + 1,
                next_min,
                k,
                arr
            );
        }
    }

    /// Assert that `arr` contains exactly the same elements as `expected` (as a multiset).
    fn assert_same_elements(arr: &[u32], expected: &[u32]) {
        let mut a = arr.to_vec();
        let mut b = expected.to_vec();
        a.sort();
        b.sort();
        assert_eq!(a, b, "element sets differ");
    }

    #[test]
    fn k_block_sort_empty() {
        let mut arr: Vec<u32> = vec![];
        for k in 1..10 {
            k_block_sort_by(&mut arr, k, |a, b| a.cmp(b));
            assert!(arr.is_empty());
        }
    }

    #[test]
    fn k_block_sort_single_element() {
        let mut arr = vec![42u32];
        for k in 1..10 {
            k_block_sort_by(&mut arr, k, |a, b| a.cmp(b));
            assert_eq!(arr, vec![42]);
        }
    }

    #[test]
    fn k_block_sort_already_sorted() {
        for k in 1..20 {
            for len in 0..20 {
                let original: Vec<u32> = (0..len).collect();
                let mut arr = original.clone();
                k_block_sort_by(&mut arr, k, |a, b| a.cmp(b));
                assert_same_elements(&arr, &original);
                assert_k_block_invariant(&arr, k);
            }
        }
    }

    #[test]
    fn k_block_sort_reverse_order() {
        for k in 1..20 {
            for len in 0..20 {
                let original: Vec<u32> = (0..len).rev().collect();
                let expected: Vec<u32> = (0..len).collect();
                let mut arr = original.clone();
                k_block_sort_by(&mut arr, k, |a, b| a.cmp(b));
                assert_same_elements(&arr, &expected);
                assert_k_block_invariant(&arr, k);
            }
        }
    }

    #[test]
    fn k_block_sort_all_equal_elements() {
        let mut arr = vec![7u32; 50];
        let expected = arr.clone();
        k_block_sort_by(&mut arr, 4, |a, b| a.cmp(b));
        assert_eq!(arr, expected);
    }

    #[test]
    fn k_block_sort_random_fuzz() {
        for seed in 1..=20u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            for k in 1..20 {
                for len in 0..20 {
                    let mut arr: Vec<u32> = (0..len as u32).collect();
                    arr.shuffle(&mut rng);
                    let expected: Vec<u32> = (0..len as u32).collect();

                    k_block_sort_by(&mut arr, k, |a, b| a.cmp(b));
                    assert_same_elements(&arr, &expected);
                    if len > 0 {
                        assert_k_block_invariant(&arr, k);
                    }
                }
            }
        }
    }

    /// Build test data: boxes[i] = [i*10, i*10+1, i*10+2, i*10+3], indices[i] = i.
    fn make_test_data(n: usize) -> (Vec<f64>, Vec<u16>) {
        let boxes: Vec<f64> = (0..n)
            .flat_map(|i| {
                let v = i as f64 * 10.0;
                [v, v + 1.0, v + 2.0, v + 3.0]
            })
            .collect();
        let indices: Vec<u16> = (0..n as u16).collect();
        (boxes, indices)
    }

    /// Verify that after applying permutation `perm`, boxes and indices at
    /// position `i` correspond to the original item at `perm[i]`.
    fn verify_permutation_result<F: PartialEq + Debug, I: PartialEq + Debug>(
        perm: &[u32],
        boxes: &[F],
        indices: &[I],
        orig_boxes: &[F],
        orig_indices: &[I],
    ) {
        for (i, &src) in perm.iter().enumerate() {
            let src = src as usize;
            let bi = i * 4;
            let bs = src * 4;
            assert_eq!(
                &boxes[bi..bi + 4],
                &orig_boxes[bs..bs + 4],
                "box mismatch at position {}: expected item from original position {}",
                i,
                src
            );
            assert_eq!(
                indices[i], orig_indices[src],
                "index mismatch at position {}: expected item from original position {}",
                i, src
            );
        }
    }

    #[test]
    fn apply_permutation_identity() {
        for n in [0, 1, 2, 5, 16, 100] {
            let (orig_boxes, orig_indices) = make_test_data(n);
            let (mut boxes, mut indices) = (orig_boxes.clone(), orig_indices.clone());
            let mut perm: Vec<u32> = (0..n as u32).collect();
            let perm_copy = perm.clone();

            apply_permutation(
                // &mut perm,
                perm.as_mut_slice(),
                &mut boxes,
                &mut MutableIndices::U16(&mut indices),
            );

            verify_permutation_result(&perm_copy, &boxes, &indices, &orig_boxes, &orig_indices);
        }
    }

    #[test]
    fn apply_permutation_reverse() {
        for n in [0, 1, 2, 5, 16, 100] {
            let (orig_boxes, orig_indices) = make_test_data(n);
            let (mut boxes, mut indices) = (orig_boxes.clone(), orig_indices.clone());
            let mut perm: Vec<u32> = (0..n as u32).rev().collect();
            let perm_copy = perm.clone();

            apply_permutation(
                perm.as_mut_slice(),
                &mut boxes,
                &mut MutableIndices::U16(&mut indices),
            );

            verify_permutation_result(&perm_copy, &boxes, &indices, &orig_boxes, &orig_indices);
        }
    }

    #[test]
    fn apply_permutation_single_cycle() {
        // Rotate all elements: perm = [1, 2, 3, ..., n-1, 0]
        let n = 8;
        let (orig_boxes, orig_indices) = make_test_data(n);
        let (mut boxes, mut indices) = (orig_boxes.clone(), orig_indices.clone());
        let mut perm: Vec<u32> = (1..n as u32).chain(std::iter::once(0)).collect();
        let perm_copy = perm.clone();

        apply_permutation(
            perm.as_mut_slice(),
            &mut boxes,
            &mut MutableIndices::U16(&mut indices),
        );

        verify_permutation_result(&perm_copy, &boxes, &indices, &orig_boxes, &orig_indices);
    }

    #[test]
    fn apply_permutation_u32_indices() {
        // Test with U32 variant of MutableIndices
        let n = 10;
        let (orig_boxes, _) = make_test_data(n);
        let mut boxes = orig_boxes.clone();
        let orig_u32_indices: Vec<u32> = (0..n as u32).collect();
        let mut u32_indices = orig_u32_indices.clone();
        let mut perm: Vec<u32> = (0..n as u32).rev().collect();
        let perm_copy = perm.clone();

        apply_permutation(
            perm.as_mut_slice(),
            &mut boxes,
            &mut MutableIndices::U32(&mut u32_indices),
        );

        verify_permutation_result(
            &perm_copy,
            &boxes,
            &u32_indices,
            &orig_boxes,
            &orig_u32_indices,
        );
    }

    #[test]
    fn apply_permutation_from_packed_pairs() {
        let n = 10;
        let (orig_boxes, orig_indices) = make_test_data(n);
        let (mut boxes, mut indices) = (orig_boxes.clone(), orig_indices.clone());

        let perm: Vec<u32> = (0..n as u32).rev().collect();
        let mut packed: Vec<(u32, u32)> = perm
            .iter()
            .copied()
            .enumerate()
            .map(|(i, p)| (i as u32, p))
            .collect();

        apply_permutation(
            packed.as_mut_slice(),
            &mut boxes,
            &mut MutableIndices::U16(&mut indices),
        );

        verify_permutation_result(&perm, &boxes, &indices, &orig_boxes, &orig_indices);
    }

    #[test]
    fn apply_permutation_random_fuzz() {
        for seed in 1..=20u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            for n in 0..100 {
                let (orig_boxes, orig_indices) = make_test_data(n);
                let (mut boxes, mut indices) = (orig_boxes.clone(), orig_indices.clone());
                let mut perm: Vec<u32> = (0..n as u32).collect();
                perm.shuffle(&mut rng);
                let perm_copy = perm.clone();

                apply_permutation(
                    perm.as_mut_slice(),
                    &mut boxes,
                    &mut MutableIndices::U16(&mut indices),
                );

                verify_permutation_result(&perm_copy, &boxes, &indices, &orig_boxes, &orig_indices);
            }
        }
    }
}
