use geo_traits::{CoordTrait, RectTrait};
use tinyvec::TinyVec;

use crate::indices::Indices;
use crate::kdtree::{KDTree, KDTreeMetadata, KDTreeRef, Node};
use crate::r#type::IndexableNum;

/// A trait for searching and accessing data out of a KDTree.
pub trait KDTreeIndex<N: IndexableNum>: Sized {
    /// The underlying raw coordinate buffer of this tree
    fn coords(&self) -> &[N];

    /// The underlying raw indices buffer of this tree
    fn indices(&self) -> Indices<'_>;

    /// Access the metadata describing this KDTree
    fn metadata(&self) -> &KDTreeMetadata<N>;

    /// The number of items in this KDTree
    fn num_items(&self) -> u32 {
        self.metadata().num_items()
    }

    /// The node size of this KDTree
    fn node_size(&self) -> u16 {
        self.metadata().node_size()
    }

    /// Search the index for items within a given bounding box.
    ///
    /// - min_x: bbox
    /// - min_y: bbox
    /// - max_x: bbox
    /// - max_y: bbox
    ///
    /// Returns indices of found items
    fn range(&self, min_x: N, min_y: N, max_x: N, max_y: N) -> Vec<u32> {
        let indices = self.indices();
        let coords = self.coords();
        let node_size = self.node_size();

        // Use TinyVec to avoid heap allocations
        let mut stack: TinyVec<[usize; 33]> = TinyVec::new();
        stack.push(0);
        stack.push(indices.len() - 1);
        stack.push(0);

        let mut result: Vec<u32> = vec![];

        // recursively search for items in range in the kd-sorted arrays
        while !stack.is_empty() {
            let axis = stack.pop().unwrap_or(0);
            let right = stack.pop().unwrap_or(0);
            let left = stack.pop().unwrap_or(0);

            // if we reached "tree node", search linearly
            if right - left <= node_size as usize {
                for i in left..=right {
                    let x = coords[2 * i];
                    let y = coords[2 * i + 1];
                    if x >= min_x && x <= max_x && y >= min_y && y <= max_y {
                        result.push(indices.get(i).try_into().unwrap());
                    }
                }
                continue;
            }

            // otherwise find the middle index
            let m = (left + right) >> 1;

            // include the middle item if it's in range
            let x = coords[2 * m];
            let y = coords[2 * m + 1];
            if x >= min_x && x <= max_x && y >= min_y && y <= max_y {
                result.push(indices.get(m).try_into().unwrap());
            }

            // queue search in halves that intersect the query
            let lte = if axis == 0 { min_x <= x } else { min_y <= y };
            if lte {
                // Note: these are pushed in backwards order to what gets popped
                stack.push(left);
                stack.push(m - 1);
                stack.push(1 - axis);
            }

            let gte = if axis == 0 { max_x >= x } else { max_y >= y };
            if gte {
                // Note: these are pushed in backwards order to what gets popped
                stack.push(m + 1);
                stack.push(right);
                stack.push(1 - axis);
            }
        }

        result
    }

    /// Search the index for items within a given bounding box.
    ///
    /// Returns indices of found items
    fn range_rect(&self, rect: &impl RectTrait<T = N>) -> Vec<u32> {
        self.range(
            rect.min().x(),
            rect.min().y(),
            rect.max().x(),
            rect.max().y(),
        )
    }

    /// Search the index for items within a given radius.
    ///
    /// - qx: x value of query point
    /// - qy: y value of query point
    /// - r: radius
    ///
    /// Returns indices of found items
    fn within(&self, qx: N, qy: N, r: N) -> Vec<u32> {
        let indices = self.indices();
        let coords = self.coords();
        let node_size = self.node_size();

        // Use TinyVec to avoid heap allocations
        let mut stack: TinyVec<[usize; 33]> = TinyVec::new();
        stack.push(0);
        stack.push(indices.len() - 1);
        stack.push(0);

        let mut result: Vec<u32> = vec![];
        let r2 = r * r;

        // recursively search for items within radius in the kd-sorted arrays
        while !stack.is_empty() {
            let axis = stack.pop().unwrap_or(0);
            let right = stack.pop().unwrap_or(0);
            let left = stack.pop().unwrap_or(0);

            // if we reached "tree node", search linearly
            if right - left <= node_size as usize {
                for i in left..=right {
                    if sq_dist(coords[2 * i], coords[2 * i + 1], qx, qy) <= r2 {
                        result.push(indices.get(i).try_into().unwrap());
                    }
                }
                continue;
            }

            // otherwise find the middle index
            let m = (left + right) >> 1;

            // include the middle item if it's in range
            let x = coords[2 * m];
            let y = coords[2 * m + 1];
            if sq_dist(x, y, qx, qy) <= r2 {
                result.push(indices.get(m).try_into().unwrap());
            }

            // queue search in halves that intersect the query
            let lte = if axis == 0 { qx - r <= x } else { qy - r <= y };
            if lte {
                stack.push(left);
                stack.push(m - 1);
                stack.push(1 - axis);
            }

            let gte = if axis == 0 { qx + r >= x } else { qy + r >= y };
            if gte {
                stack.push(m + 1);
                stack.push(right);
                stack.push(1 - axis);
            }
        }
        result
    }

    /// Search the index for closest points. If ties returns multiple
    ///
    /// - qx: x value of query point
    /// - qy: y value of query point
    ///
    /// Returns distance squared and indices of found items
    fn query(&self, qx: N, qy: N) -> (N, Vec<u32>) {
        let indices = self.indices();
        let coords = self.coords();
        let node_size = self.node_size();

        // Use TinyVec to avoid heap allocations
        let mut stack: TinyVec<[usize; 33]> = TinyVec::new();
        stack.push(0);
        stack.push(indices.len() - 1);
        stack.push(0);

        let mut result: Vec<u32> = vec![];
        let mut min_val = N::max_value();

        // recursively search for items within radius in the kd-sorted arrays
        while !stack.is_empty() {
            let axis = stack.pop().unwrap_or(0);
            let right = stack.pop().unwrap_or(0);
            let left = stack.pop().unwrap_or(0);

            // if we reached "tree node", search linearly
            if right - left <= node_size as usize {
                for i in left..=right {
                    let d = sq_dist(coords[2 * i], coords[2 * i + 1], qx, qy);
                    if d < min_val {
                        result.clear();
                        result.push(indices.get(i).try_into().unwrap());
                        min_val = d;
                    } else if d == min_val {
                        result.push(indices.get(i).try_into().unwrap());
                    }
                }
                continue;
            }

            // otherwise find the middle index
            let m = (left + right) >> 1;

            // include the middle item if it's in range
            let x = coords[2 * m];
            let y = coords[2 * m + 1];
            let d = sq_dist(x, y, qx, qy);
            if d < min_val {
                result.clear();
                result.push(indices.get(m).try_into().unwrap());
                min_val = d;
            } else if d == min_val {
                result.push(indices.get(m).try_into().unwrap());
            }

            // queue search in halves that intersect the query
            let lte = if axis == 0 {
                qx - min_val <= x
            } else {
                qy - min_val <= y
            };
            if lte {
                stack.push(left);
                stack.push(m - 1);
                stack.push(1 - axis);
            }

            let gte = if axis == 0 {
                qx + min_val >= x
            } else {
                qy + min_val >= y
            };
            if gte {
                stack.push(m + 1);
                stack.push(right);
                stack.push(1 - axis);
            }
        }
        let d = min_val.sqrt().unwrap();
        (d, result)
    }

    /// Search the index for items within a given radius.
    ///
    /// - coord: coordinate of query point
    /// - r: radius
    ///
    /// Returns indices of found items
    fn within_coord(&self, coord: &impl CoordTrait<T = N>, r: N) -> Vec<u32> {
        self.within(coord.x(), coord.y(), r)
    }

    /// Access the root node of the KDTree for manual traversal.
    fn root(&self) -> Node<'_, N, Self> {
        Node::from_root(self)
    }
}

impl<N: IndexableNum> KDTreeIndex<N> for KDTree<N> {
    fn coords(&self) -> &[N] {
        self.metadata.coords_slice(&self.buffer)
    }

    fn indices(&self) -> Indices<'_> {
        self.metadata.indices_slice(&self.buffer)
    }

    fn metadata(&self) -> &KDTreeMetadata<N> {
        &self.metadata
    }
}

impl<N: IndexableNum> KDTreeIndex<N> for KDTreeRef<'_, N> {
    fn coords(&self) -> &[N] {
        self.coords
    }

    fn indices(&self) -> Indices<'_> {
        self.indices
    }

    fn metadata(&self) -> &KDTreeMetadata<N> {
        &self.metadata
    }
}

#[inline]
pub(crate) fn sq_dist<N: IndexableNum>(ax: N, ay: N, bx: N, by: N) -> N {
    let dx = ax - bx;
    let dy = ay - by;
    dx * dx + dy * dy
}
