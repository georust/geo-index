use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::vec;

#[cfg(feature = "use-geo_0_31")]
use geo_0_31::Geometry;
use geo_traits::{CoordTrait, RectTrait};

use crate::error::Result;
use crate::indices::Indices;
use crate::r#type::IndexableNum;
#[cfg(feature = "use-geo_0_31")]
use crate::rtree::distance::DistanceMetric;
use crate::rtree::index::{RTree, RTreeRef};
use crate::rtree::traversal::{IntersectionIterator, Node};
use crate::rtree::util::upper_bound;
use crate::rtree::RTreeMetadata;
use crate::GeoIndexError;

/// A simple distance metric trait that doesn't depend on geo.
///
/// This trait is used for basic distance calculations without geometry support.
pub trait SimpleDistanceMetric<N: IndexableNum> {
    /// Calculate the distance between two points (x1, y1) and (x2, y2).
    fn distance(&self, x1: N, y1: N, x2: N, y2: N) -> N;

    /// Calculate the distance from a point to a bounding box.
    fn distance_to_bbox(&self, x: N, y: N, min_x: N, min_y: N, max_x: N, max_y: N) -> N;

    /// Return the maximum distance value for this metric.
    fn max_distance(&self) -> N {
        N::max_value()
    }
}

/// A trait for accessing geometries by index.
///
/// This trait allows different storage strategies for geometries (direct storage,
/// WKB decoding, caching, etc.) to be used with spatial index queries.
#[cfg(feature = "use-geo_0_31")]
pub trait GeometryAccessor {
    /// Get the geometry at the given index.
    ///
    /// # Arguments
    /// * `item_index` - Index of the item to retrieve
    ///
    /// # Returns
    /// A reference to the geometry at the given index, or None if the index is out of bounds
    fn get_geometry(&self, item_index: usize) -> Option<&Geometry<f64>>;
}

/// Options for nearest neighbor searches.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NeighborsOptions<N: IndexableNum> {
    /// Maximum number of neighbors to return. None means unbounded.
    pub k: Option<usize>,
    /// Optional maximum distance threshold.
    pub max_distance: Option<N>,
    /// If true, include all items tied at rank k.
    pub include_tie_breakers: bool,
}

impl<N: IndexableNum> Default for NeighborsOptions<N> {
    fn default() -> Self {
        Self {
            k: Some(1),
            max_distance: None,
            include_tie_breakers: false,
        }
    }
}

impl<N: IndexableNum> NeighborsOptions<N> {
    /// Create options for a single nearest neighbor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create options for k nearest neighbors.
    pub fn k(k: usize) -> Self {
        Self {
            k: Some(k),
            ..Self::default()
        }
    }

    /// Create options for an unbounded neighbor search.
    pub fn all() -> Self {
        Self {
            k: None,
            ..Self::default()
        }
    }

    /// Set a maximum distance threshold.
    pub fn max_distance(mut self, max_distance: N) -> Self {
        self.max_distance = Some(max_distance);
        self
    }

    /// Enable or disable tie breaker inclusion.
    pub fn include_tie_breakers(mut self, include_tie_breakers: bool) -> Self {
        self.include_tie_breakers = include_tie_breakers;
        self
    }
}

/// A trait for searching and accessing data out of an RTree.
pub trait RTreeIndex<N: IndexableNum>: Sized {
    /// A slice representing all the bounding boxes of all elements contained within this tree,
    /// including the bounding boxes of each internal node.
    fn boxes(&self) -> &[N];

    /// A slice representing the indices within the `boxes` slice, including internal nodes.
    fn indices(&self) -> Indices<'_>;

    /// Access the metadata describing this RTree
    fn metadata(&self) -> &RTreeMetadata<N>;

    /// The total number of items contained in this RTree.
    fn num_items(&self) -> u32 {
        self.metadata().num_items()
    }

    /// The total number of nodes in this RTree, including both leaf and intermediate nodes.
    fn num_nodes(&self) -> usize {
        self.metadata().num_nodes()
    }

    /// The maximum number of elements in each node.
    fn node_size(&self) -> u16 {
        self.metadata().node_size()
    }

    /// The offsets into [RTreeIndex::boxes] where each level's boxes starts and ends. The tree is
    /// laid out bottom-up, and there's an implicit initial 0. So the boxes of the lowest level of
    /// the tree are located from `boxes[0..self.level_bounds()[0]]`.
    fn level_bounds(&self) -> &[usize] {
        self.metadata().level_bounds()
    }

    /// The number of levels (height) of the tree.
    fn num_levels(&self) -> usize {
        self.level_bounds().len()
    }

    /// The tree is laid out from bottom to top. Level 0 is the _base_ of the tree. Each integer
    /// higher is one level higher of the tree.
    fn boxes_at_level(&self, level: usize) -> Result<&[N]> {
        let level_bounds = self.level_bounds();
        if level >= level_bounds.len() {
            return Err(GeoIndexError::General("Level out of bounds".to_string()));
        }
        let result = if level == 0 {
            &self.boxes()[0..level_bounds[0]]
        } else if level == level_bounds.len() {
            &self.boxes()[level_bounds[level]..]
        } else {
            &self.boxes()[level_bounds[level - 1]..level_bounds[level]]
        };
        Ok(result)
    }

    /// Search an RTree given the provided bounding box.
    ///
    /// Results are the indexes of the inserted objects in insertion order.
    fn search(&self, min_x: N, min_y: N, max_x: N, max_y: N) -> Vec<u32> {
        let boxes = self.boxes();
        let indices = self.indices();
        if boxes.is_empty() {
            return vec![];
        }

        let mut outer_node_index = boxes.len().checked_sub(4);

        let mut queue = VecDeque::with_capacity(self.node_size() as usize);
        let mut results = vec![];

        while let Some(node_index) = outer_node_index {
            // find the end index of the node
            let end = (node_index + self.node_size() as usize * 4)
                .min(upper_bound(node_index, self.level_bounds()));

            // search through child nodes
            for pos in (node_index..end).step_by(4) {
                // Safety: pos was checked before to be within bounds
                // Justification: avoiding bounds check improves performance by up to 30%
                let (node_min_x, node_min_y, node_max_x, node_max_y) = unsafe {
                    let node_min_x = *boxes.get_unchecked(pos);
                    let node_min_y = *boxes.get_unchecked(pos + 1);
                    let node_max_x = *boxes.get_unchecked(pos + 2);
                    let node_max_y = *boxes.get_unchecked(pos + 3);
                    (node_min_x, node_min_y, node_max_x, node_max_y)
                };

                // check if the query box disjoint with the node box
                if max_x < node_min_x
                    || max_y < node_min_y
                    || min_x > node_max_x
                    || min_y > node_max_y
                {
                    continue;
                }

                let index = indices.get(pos >> 2);

                if node_index >= self.num_items() as usize * 4 {
                    queue.push_back(index); // node; add it to the search queue
                } else {
                    // Since the max items of the index is u32, we can coerce to u32
                    results.push(index.try_into().unwrap()); // leaf item
                }
            }

            outer_node_index = queue.pop_front();
        }

        results
    }

    /// Search an RTree given the provided bounding box.
    ///
    /// Results are the indexes of the inserted objects in insertion order.
    fn search_rect(&self, rect: &impl RectTrait<T = N>) -> Vec<u32> {
        self.search(
            rect.min().x(),
            rect.min().y(),
            rect.max().x(),
            rect.max().y(),
        )
    }

    /// Search items in order of distance from the given point.
    ///
    /// This method uses Euclidean distance by default. For other distance metrics,
    /// use [`neighbors_with_distance`].
    ///
    /// ```
    /// use geo_index::rtree::{RTreeBuilder, RTreeIndex, RTreeRef};
    /// use geo_index::rtree::sort::HilbertSort;
    ///
    /// // Create an RTree
    /// let mut builder = RTreeBuilder::<f64>::new(3);
    /// builder.add(0., 0., 2., 2.);
    /// builder.add(1., 1., 3., 3.);
    /// builder.add(2., 2., 4., 4.);
    /// let tree = builder.finish::<HilbertSort>();
    ///
    /// let results = tree.neighbors(5., 5., None, None);
    /// assert_eq!(results, vec![2, 1, 0]);
    /// ```
    fn neighbors(
        &self,
        x: N,
        y: N,
        max_results: Option<usize>,
        max_distance: Option<N>,
    ) -> Vec<u32> {
        let options = NeighborsOptions {
            k: max_results,
            max_distance,
            include_tie_breakers: false,
        };
        // Use simple squared distance for backward compatibility
        struct SimpleSquaredDistance;
        impl<N: IndexableNum> SimpleDistanceMetric<N> for SimpleSquaredDistance {
            fn distance(&self, x1: N, y1: N, x2: N, y2: N) -> N {
                let dx = x2 - x1;
                let dy = y2 - y1;
                dx * dx + dy * dy
            }
            fn distance_to_bbox(&self, x: N, y: N, min_x: N, min_y: N, max_x: N, max_y: N) -> N {
                let dx = axis_dist(x, min_x, max_x);
                let dy = axis_dist(y, min_y, max_y);
                dx * dx + dy * dy
            }
        }
        let simple_distance = SimpleSquaredDistance;
        self.neighbors_with_simple_distance(x, y, options, &simple_distance)
            .into_iter()
            .map(|(idx, _dist)| idx)
            .collect()
    }

    /// Search items in order of distance from the given point using a simple distance metric.
    ///
    /// This is the base method for distance-based neighbor searches that works without the geo feature.
    ///
    /// # Arguments
    /// * `x` - The x coordinate of the query point
    /// * `y` - The y coordinate of the query point
    /// * `options` - Neighbor search options
    /// * `distance_metric` - The distance metric to use
    ///
    /// # Returns
    /// Vector of tuples (item_index, distance) ordered by increasing distance
    fn neighbors_with_simple_distance<M: SimpleDistanceMetric<N> + ?Sized>(
        &self,
        x: N,
        y: N,
        options: NeighborsOptions<N>,
        distance_metric: &M,
    ) -> Vec<(u32, N)> {
        self.neighbors_with_callbacks(
            NeighborsOptions {
                max_distance: Some(
                    options
                        .max_distance
                        .unwrap_or(distance_metric.max_distance()),
                ),
                ..options
            },
            |[min_x, min_y, max_x, max_y]| {
                distance_metric.distance_to_bbox(x, y, min_x, min_y, max_x, max_y)
            },
            |_, [min_x, min_y, max_x, max_y]| {
                Some(distance_metric.distance_to_bbox(x, y, min_x, min_y, max_x, max_y))
            },
        )
    }

    /// Search items in increasing order of a caller-defined distance, without a geometry dependency.
    ///
    /// `bbox_distance` receives an internal node's bounding box as
    /// `[min_x, min_y, max_x, max_y]` and must return a lower bound on the distance
    /// to every item in that node. `item_distance` receives an item's original
    /// insertion index and bounding box and returns its exact distance, or `None`
    /// to exclude it. Both callbacks may capture the query and external geometry
    /// storage, including data decoded on demand.
    ///
    /// Distances and `max_distance` must use the same units and ordering, and must
    /// not be NaN. The distance type may differ from the tree's coordinate type.
    /// The distance limit is inclusive; `None` uses the distance type's maximum
    /// value. `options.k` limits the number of results (zero returns no items).
    /// Set `options.include_tie_breakers` to include all items tied at rank k.
    /// Returns `(insertion_index, distance)` pairs ordered by increasing distance.
    /// Items with equal distances may be returned in any order.
    ///
    /// Correct ranking and pruning require valid lower bounds. For spherical
    /// distances, use spherical bounds that account for poles and the antimeridian;
    /// a planar closest point on a longitude/latitude box is not generally valid.
    /// Returning zero is always a safe bound for nonnegative distances, but may
    /// require evaluating every item. Indexed boxes must enclose the full geometry
    /// under the chosen metric, including any extrema along curved edges.
    ///
    /// # Example
    ///
    /// Rank externally stored points using Manhattan distance. No feature is required.
    ///
    /// ```
    /// use geo_index::rtree::{NeighborsOptions, RTreeBuilder, RTreeIndex, sort::HilbertSort};
    ///
    /// let points = [(3.0_f64, 4.0_f64), (1.0, 2.0), (8.0, 1.0)];
    /// let mut builder = RTreeBuilder::<f64>::new(points.len() as u32);
    /// for &(x, y) in &points {
    ///     builder.add(x, y, x, y);
    /// }
    /// let tree = builder.finish::<HilbertSort>();
    /// let query = (0.0_f64, 0.0_f64);
    /// let results = tree.neighbors_with_callbacks(
    ///     NeighborsOptions::k(2),
    ///     |[min_x, min_y, max_x, max_y]| {
    ///         (query.0 - query.0.clamp(min_x, max_x)).abs()
    ///             + (query.1 - query.1.clamp(min_y, max_y)).abs()
    ///     },
    ///     |id, _bbox| {
    ///         let (x, y) = points[id as usize];
    ///         Some((x - query.0).abs() + (y - query.1).abs())
    ///     },
    /// );
    /// assert_eq!(results, vec![(1, 3.0), (0, 7.0)]);
    /// ```
    fn neighbors_with_callbacks<D, B, I>(
        &self,
        options: NeighborsOptions<D>,
        mut bbox_distance: B,
        mut item_distance: I,
    ) -> Vec<(u32, D)>
    where
        D: IndexableNum,
        B: FnMut([N; 4]) -> D,
        I: FnMut(u32, [N; 4]) -> Option<D>,
    {
        let NeighborsOptions {
            k,
            max_distance,
            include_tie_breakers,
        } = options;
        if k == Some(0) {
            return vec![];
        }
        let boxes = self.boxes();
        if boxes.is_empty() {
            return vec![];
        }

        let indices = self.indices();
        let max_distance = max_distance.unwrap_or(D::max_value());

        let mut outer_node_index = boxes.len().checked_sub(4);
        let mut queue = BinaryHeap::new();
        let mut results: Vec<(u32, D)> = vec![];
        let mut kth_distance: Option<D> = None;

        'outer: while let Some(node_index) = outer_node_index {
            // find the end index of the node
            let end = (node_index + self.node_size() as usize * 4)
                .min(upper_bound(node_index, self.level_bounds()));

            // add child nodes to the queue
            for pos in (node_index..end).step_by(4) {
                let index = indices.get(pos >> 2);

                let bbox = [boxes[pos], boxes[pos + 1], boxes[pos + 2], boxes[pos + 3]];
                let dist = if node_index >= self.num_items() as usize * 4 {
                    bbox_distance(bbox)
                } else {
                    let Some(dist) = item_distance(index as u32, bbox) else {
                        continue;
                    };
                    dist
                };

                if dist > max_distance {
                    continue;
                }

                if node_index >= self.num_items() as usize * 4 {
                    // node (use even id)
                    queue.push(Reverse(NeighborNode {
                        id: index << 1,
                        dist,
                    }));
                } else {
                    // leaf item (use odd id)
                    queue.push(Reverse(NeighborNode {
                        id: (index << 1) + 1,
                        dist,
                    }));
                }
            }

            // pop items from the queue
            while !queue.is_empty() && queue.peek().is_some_and(|val| (val.0.id & 1) != 0) {
                let dist = queue.peek().unwrap().0.dist;
                if dist > max_distance {
                    break 'outer;
                }

                // If we've reached k items and not including tie breakers, we should stop
                if !include_tie_breakers && k.is_some_and(|k_val| results.len() == k_val) {
                    break 'outer;
                }

                // If including tie breakers and we're about to add the k-th item, record its distance
                if include_tie_breakers
                    && kth_distance.is_none()
                    && k.is_some_and(|k_val| results.len() + 1 == k_val)
                {
                    kth_distance = Some(dist);
                }

                // If we have recorded k-th distance and current distance exceeds it, stop
                if include_tie_breakers && kth_distance.is_some_and(|kth| dist > kth) {
                    break 'outer;
                }

                let item = queue.pop().unwrap();
                let item_index: u32 = (item.0.id >> 1).try_into().unwrap();
                results.push((item_index, item.0.dist));
            }

            if let Some(item) = queue.pop() {
                outer_node_index = Some(item.0.id >> 1);
            } else {
                outer_node_index = None;
            }
        }

        results
    }

    /// Search items in order of distance from the given point using a custom distance metric.
    ///
    /// This method allows you to specify a custom distance calculation method, such as
    /// Euclidean, Haversine, or Spheroid distance.
    ///
    /// # Arguments
    /// * `x` - The x coordinate of the query point
    /// * `y` - The y coordinate of the query point
    /// * `options` - Neighbor search options
    /// * `distance_metric` - The distance metric to use
    ///
    /// # Returns
    /// Vector of tuples (item_index, distance) ordered by increasing distance
    ///
    /// # Examples
    /// ```
    /// use geo_index::rtree::{RTreeBuilder, RTreeIndex};
    /// use geo_index::rtree::distance::{EuclideanDistance, HaversineDistance};
    /// use geo_index::rtree::sort::HilbertSort;
    ///
    /// // Create an RTree with geographic coordinates (longitude, latitude)
    /// let mut builder = RTreeBuilder::<f64>::new(3);
    /// builder.add(-74.0, 40.7, -74.0, 40.7); // New York
    /// builder.add(-0.1, 51.5, -0.1, 51.5);   // London
    /// builder.add(139.7, 35.7, 139.7, 35.7); // Tokyo
    /// let tree = builder.finish::<HilbertSort>();
    ///
    /// // Find nearest neighbors using Haversine distance (great-circle distance)
    /// let haversine = HaversineDistance::default();
    /// use geo_index::rtree::NeighborsOptions;
    /// let results = tree.neighbors_with_distance(
    ///     -74.0,
    ///     40.7,
    ///     NeighborsOptions::k(2),
    ///     &haversine,
    /// );
    /// // Results: [(0, 0.0), (1, 5570000.0)]  // distances in meters
    /// ```
    #[cfg(feature = "use-geo_0_31")]
    fn neighbors_with_distance<M: DistanceMetric<N> + ?Sized>(
        &self,
        x: N,
        y: N,
        options: NeighborsOptions<N>,
        distance_metric: &M,
    ) -> Vec<(u32, N)> {
        self.neighbors_with_simple_distance(x, y, options, distance_metric)
    }

    /// Search items in order of distance from the given coordinate.
    fn neighbors_coord(
        &self,
        coord: &impl CoordTrait<T = N>,
        max_results: Option<usize>,
        max_distance: Option<N>,
    ) -> Vec<u32> {
        self.neighbors(coord.x(), coord.y(), max_results, max_distance)
    }

    /// Search items in order of distance from the given coordinate using a custom distance metric.
    ///
    /// # Arguments
    /// * `coord` - The query coordinate
    /// * `options` - Neighbor search options
    /// * `distance_metric` - The distance metric to use
    ///
    /// # Returns
    /// Vector of tuples (item_index, distance) ordered by increasing distance
    #[cfg(feature = "use-geo_0_31")]
    fn neighbors_coord_with_distance<M: DistanceMetric<N> + ?Sized>(
        &self,
        coord: &impl CoordTrait<T = N>,
        options: NeighborsOptions<N>,
        distance_metric: &M,
    ) -> Vec<(u32, N)> {
        self.neighbors_with_distance(coord.x(), coord.y(), options, distance_metric)
    }

    /// Search items in order of distance from a query geometry using a distance metric and geometry accessor.
    ///
    /// This method allows searching with geometry-to-geometry distance calculations.
    /// The distance metric defines how distances are computed, and the geometry accessor
    /// provides access to the actual geometries by index.
    ///
    /// # Arguments
    /// * `query_geometry` - The query geometry
    /// * `options` - Neighbor search options
    /// * `distance_metric` - The distance metric to use
    /// * `accessor` - Provides access to geometries by index
    ///
    /// # Returns
    /// Vector of tuples (item_index, distance) ordered by increasing distance
    ///
    /// # Examples
    /// ```
    /// use geo_index::rtree::{RTreeBuilder, RTreeIndex};
    /// use geo_index::rtree::distance::{EuclideanDistance, SliceGeometryAccessor};
    /// use geo_index::rtree::sort::HilbertSort;
    /// use geo_0_31::{Point, Geometry};
    ///
    /// // Create an RTree
    /// let mut builder = RTreeBuilder::<f64>::new(3);
    /// builder.add(0., 0., 2., 2.);
    /// builder.add(5., 5., 7., 7.);
    /// builder.add(10., 10., 12., 12.);
    /// let tree = builder.finish::<HilbertSort>();
    ///
    /// // Example geometries
    /// let geometries: Vec<Geometry<f64>> = vec![
    ///     Geometry::Point(Point::new(1.0, 1.0)),
    ///     Geometry::Point(Point::new(6.0, 6.0)),
    ///     Geometry::Point(Point::new(11.0, 11.0)),
    /// ];
    ///
    /// let metric = EuclideanDistance;
    /// let accessor = SliceGeometryAccessor::new(&geometries);
    /// let query_geom = Geometry::Point(Point::new(3.0, 3.0));
    /// use geo_index::rtree::NeighborsOptions;
    /// let results = tree.neighbors_geometry(
    ///     &query_geom,
    ///     NeighborsOptions::all(),
    ///     &metric,
    ///     &accessor,
    /// );
    /// // Results: [(0, 2.82...), (1, 4.24...), (2, 11.31...)]
    /// ```
    #[cfg(feature = "use-geo_0_31")]
    fn neighbors_geometry<M: DistanceMetric<N> + ?Sized, A: GeometryAccessor + ?Sized>(
        &self,
        query_geometry: &Geometry<f64>,
        options: NeighborsOptions<N>,
        distance_metric: &M,
        accessor: &A,
    ) -> Vec<(u32, N)> {
        self.neighbors_with_callbacks(
            NeighborsOptions {
                max_distance: Some(
                    options
                        .max_distance
                        .unwrap_or(distance_metric.max_distance()),
                ),
                ..options
            },
            |[min_x, min_y, max_x, max_y]| {
                distance_metric.distance_geometry_to_bbox(
                    query_geometry,
                    min_x,
                    min_y,
                    max_x,
                    max_y,
                )
            },
            |index, _bbox| {
                Some(match accessor.get_geometry(index as usize) {
                    Some(item_geom) => {
                        distance_metric.distance_to_geometry(query_geometry, item_geom)
                    }
                    None => distance_metric.max_distance(),
                })
            },
        )
    }

    /// Returns an iterator over the indexes of objects in this and another tree that intersect.
    ///
    /// Each returned object is of the form `(u32, u32)`, where the first is the positional
    /// index of the "left" tree and the second is the index of the "right" tree.
    fn intersection_candidates_with_other_tree<'a>(
        &'a self,
        other: &'a impl RTreeIndex<N>,
    ) -> impl Iterator<Item = (u32, u32)> + 'a {
        IntersectionIterator::from_trees(self, other)
    }

    /// Access the root node of the RTree for manual traversal.
    fn root(&self) -> Node<'_, N, Self> {
        Node::from_root(self)
    }
}

/// A wrapper around a node and its distance for use in the priority queue.
#[derive(Debug, Clone, Copy, PartialEq)]
struct NeighborNode<N: IndexableNum> {
    id: usize,
    dist: N,
}

impl<N: IndexableNum> Eq for NeighborNode<N> {}

impl<N: IndexableNum> Ord for NeighborNode<N> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // We don't allow NaN. This should only panic on NaN
        self.dist.partial_cmp(&other.dist).unwrap()
    }
}

impl<N: IndexableNum> PartialOrd for NeighborNode<N> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<N: IndexableNum> RTreeIndex<N> for RTree<N> {
    fn boxes(&self) -> &[N] {
        self.metadata.boxes_slice(&self.buffer)
    }

    fn indices(&self) -> Indices<'_> {
        self.metadata.indices_slice(&self.buffer)
    }

    fn metadata(&self) -> &RTreeMetadata<N> {
        &self.metadata
    }
}

impl<N: IndexableNum> RTreeIndex<N> for RTreeRef<'_, N> {
    fn boxes(&self) -> &[N] {
        self.boxes
    }

    fn indices(&self) -> Indices<'_> {
        self.indices
    }

    fn metadata(&self) -> &RTreeMetadata<N> {
        &self.metadata
    }
}

/// 1D distance from a value to a range.
#[inline]
pub(crate) fn axis_dist<N: IndexableNum>(k: N, min: N, max: N) -> N {
    if k < min {
        min - k
    } else if k <= max {
        N::zero()
    } else {
        k - max
    }
}

#[cfg(test)]
mod test {
    // Replication of tests from flatbush js
    mod js {
        use crate::rtree::RTreeIndex;
        use crate::test::{flatbush_js_test_data, flatbush_js_test_index};

        #[test]
        fn performs_bbox_search() {
            let data = flatbush_js_test_data();
            let index = flatbush_js_test_index();
            let ids = index.search(40., 40., 60., 60.);

            let mut results: Vec<usize> = vec![];
            for id in ids {
                results.push(data[4 * id as usize] as usize);
                results.push(data[4 * id as usize + 1] as usize);
                results.push(data[4 * id as usize + 2] as usize);
                results.push(data[4 * id as usize + 3] as usize);
            }

            results.sort();

            let mut expected = vec![
                57, 59, 58, 59, 48, 53, 52, 56, 40, 42, 43, 43, 43, 41, 47, 43,
            ];
            expected.sort();

            assert_eq!(results, expected);
        }
    }
    #[cfg(feature = "use-geo_0_31")]
    mod distance_metrics {
        use crate::rtree::distance::{EuclideanDistance, HaversineDistance};
        use crate::rtree::r#trait::SimpleDistanceMetric;
        use crate::rtree::sort::HilbertSort;
        use crate::rtree::{NeighborsOptions, RTreeBuilder, RTreeIndex};

        #[test]
        fn test_euclidean_distance_neighbors() {
            let mut builder = RTreeBuilder::<f64>::new(3);
            builder.add(0., 0., 1., 1.);
            builder.add(2., 2., 3., 3.);
            builder.add(4., 4., 5., 5.);
            let tree = builder.finish::<HilbertSort>();

            let euclidean = EuclideanDistance;
            let results = tree.neighbors_with_distance(0., 0., NeighborsOptions::all(), &euclidean);

            // Should return items in order of distance from (0,0)
            assert_eq!(results.len(), 3);
            assert_eq!(results[0].0, 0);
            assert_eq!(results[1].0, 1);
            assert_eq!(results[2].0, 2);
            // Verify distances are returned
            assert!(results[0].1 < results[1].1);
            assert!(results[1].1 < results[2].1);
        }

        #[test]
        fn test_haversine_distance_neighbors() {
            let mut builder = RTreeBuilder::<f64>::new(3);
            // Add some geographic points (longitude, latitude)
            builder.add(-74.0, 40.7, -74.0, 40.7); // New York
            builder.add(-0.1, 51.5, -0.1, 51.5); // London
            builder.add(139.7, 35.7, 139.7, 35.7); // Tokyo
            let tree = builder.finish::<HilbertSort>();

            let haversine = HaversineDistance::default();
            let results =
                tree.neighbors_with_distance(-74.0, 40.7, NeighborsOptions::all(), &haversine);

            // From New York, should find New York first, then London, then Tokyo
            assert_eq!(results.len(), 3);
            assert_eq!(results[0].0, 0);
            assert_eq!(results[1].0, 1);
            assert_eq!(results[2].0, 2);
            // Verify distances: New York should be 0, London and Tokyo non-zero
            assert_eq!(results[0].1, 0.0);
            assert!(results[1].1 > 0.0);
            assert!(results[2].1 > 0.0);
        }

        #[test]
        fn test_backward_compatibility() {
            let mut builder = RTreeBuilder::<f64>::new(3);
            builder.add(0., 0., 1., 1.);
            builder.add(2., 2., 3., 3.);
            builder.add(4., 4., 5., 5.);
            let tree = builder.finish::<HilbertSort>();

            // Test that original neighbors method still works
            let results_original = tree.neighbors(0., 0., None, None);

            // Test that new method with Euclidean distance gives same order (just extract indices)
            let euclidean = EuclideanDistance;
            let results_new = tree
                .neighbors_with_distance(0., 0., NeighborsOptions::all(), &euclidean)
                .into_iter()
                .map(|(idx, _dist)| idx)
                .collect::<Vec<_>>();

            assert_eq!(results_original, results_new);
        }

        #[test]
        fn test_max_distance_filtering() {
            let mut builder = RTreeBuilder::<f64>::new(3);
            builder.add(0., 0., 1., 1.);
            builder.add(2., 2., 3., 3.);
            builder.add(10., 10., 11., 11.);
            let tree = builder.finish::<HilbertSort>();

            let euclidean = EuclideanDistance;
            // Only find neighbors within distance 5
            let results = tree.neighbors_with_distance(
                0.,
                0.,
                NeighborsOptions::all().max_distance(5.0),
                &euclidean,
            );

            // Should only find first two items, not the distant third one
            assert_eq!(results.len(), 2);
            assert_eq!(results[0].0, 0);
            assert_eq!(results[1].0, 1);
        }

        #[test]
        fn test_tie_breakers_enabled_without_ties() {
            let mut builder = RTreeBuilder::<f64>::new(5);
            builder.add(0., 0., 0., 0.); // Item 0: distance 0
            builder.add(1., 0., 1., 0.); // Item 1: distance 1
            builder.add(0., 2., 0., 2.); // Item 2: distance 2
            builder.add(3., 0., 3., 0.); // Item 3: distance 3
            builder.add(0., 4., 0., 4.); // Item 4: distance 4
            let tree = builder.finish::<HilbertSort>();

            let euclidean = EuclideanDistance;
            let results = tree.neighbors_with_distance(
                0.,
                0.,
                NeighborsOptions::k(3).include_tie_breakers(true),
                &euclidean,
            );

            assert_eq!(results.len(), 3);
            assert_eq!(results[0].0, 0);
            assert_eq!(results[1].0, 1);
            assert_eq!(results[2].0, 2);
        }

        #[test]
        fn test_tie_breakers_k1_includes_all_nearest() {
            let mut builder = RTreeBuilder::<f64>::new(4);
            builder.add(1., 0., 1., 0.);
            builder.add(-1., 0., -1., 0.);
            builder.add(0., 1., 0., 1.);
            builder.add(0., -1., 0., -1.);
            let tree = builder.finish::<HilbertSort>();

            let euclidean = EuclideanDistance;
            let results = tree.neighbors_with_distance(
                0.,
                0.,
                NeighborsOptions::k(1).include_tie_breakers(true),
                &euclidean,
            );

            assert_eq!(results.len(), 4);
            for (_, dist) in &results {
                assert!((*dist - 1.0).abs() < 1e-10);
            }

            let results_no_ties =
                tree.neighbors_with_distance(0., 0., NeighborsOptions::k(1), &euclidean);
            assert_eq!(results_no_ties.len(), 1);
        }

        #[test]
        fn test_tie_breakers_enabled_when_unbounded() {
            let mut builder = RTreeBuilder::<f64>::new(3);
            builder.add(1., 0., 1., 0.); // distance 1
            builder.add(2., 0., 2., 0.); // distance 2
            builder.add(3., 0., 3., 0.); // distance 3
            let tree = builder.finish::<HilbertSort>();

            let euclidean = EuclideanDistance;
            let results_default =
                tree.neighbors_with_distance(0., 0., NeighborsOptions::all(), &euclidean);
            let results_ties = tree.neighbors_with_distance(
                0.,
                0.,
                NeighborsOptions::all().include_tie_breakers(true),
                &euclidean,
            );

            assert_eq!(results_default, results_ties);
        }

        #[test]
        fn test_tie_breakers_with_max_distance_boundary() {
            let mut builder = RTreeBuilder::<f64>::new(4);
            builder.add(1., 0., 1., 0.); // distance 1
            builder.add(2., 0., 2., 0.); // distance 2
            builder.add(0., 2., 0., 2.); // distance 2
            builder.add(3., 0., 3., 0.); // distance 3
            let tree = builder.finish::<HilbertSort>();

            let euclidean = EuclideanDistance;
            let results = tree.neighbors_with_distance(
                0.,
                0.,
                NeighborsOptions::k(2)
                    .include_tie_breakers(true)
                    .max_distance(2.0),
                &euclidean,
            );

            assert_eq!(results.len(), 3);
            let mut indices: Vec<u32> = results.iter().map(|(idx, _)| *idx).collect();
            indices.sort();
            assert_eq!(indices, vec![0, 1, 2]);
            for (_, dist) in &results {
                assert!(*dist <= 2.0 + 1e-10);
            }

            let results = tree.neighbors_with_distance(
                0.,
                0.,
                NeighborsOptions::k(2)
                    .include_tie_breakers(true)
                    .max_distance(1.5),
                &euclidean,
            );
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].0, 0);
        }

        #[test]
        fn test_tie_at_kth_distance_is_the_largest_distance() {
            let mut builder = RTreeBuilder::<f64>::new(9);
            builder.add(1., 0., 1., 0.);
            builder.add(2., 0., 2., 0.);
            builder.add(-2., 0., -2., 0.);
            builder.add(0., 2., 0., 2.);
            builder.add(0., -2., 0., -2.);
            builder.add(2., 0., 2., 0.);
            builder.add(-2., 0., -2., 0.);
            builder.add(0., 2., 0., 2.);
            builder.add(0., -2., 0., -2.);
            let tree = builder.finish::<HilbertSort>();

            let euclidean = EuclideanDistance;
            for k in 2..=10 {
                let results = tree.neighbors_with_distance(
                    0.,
                    0.,
                    NeighborsOptions::k(k).include_tie_breakers(true),
                    &euclidean,
                );

                assert_eq!(results.len(), 9);
                let tie_count = results
                    .iter()
                    .filter(|(_, dist)| (*dist - 2.0).abs() < 1e-10)
                    .count();
                assert_eq!(tie_count, 8);
            }
        }

        #[test]
        fn test_many_tie_at_kth_distance() {
            let mut builder = RTreeBuilder::<f64>::new(7);
            builder.add(1., 0., 1., 0.);
            builder.add(2., 0., 2., 0.);
            builder.add(-2., 0., -2., 0.);
            builder.add(0., 2., 0., 2.);
            builder.add(0., -2., 0., -2.);
            builder.add(3., 0., 3., 0.);
            builder.add(-3., 0., -3., 0.);
            let tree = builder.finish::<HilbertSort>();

            let euclidean = EuclideanDistance;
            for k in 2..=5 {
                let results = tree.neighbors_with_distance(
                    0.,
                    0.,
                    NeighborsOptions::k(k).include_tie_breakers(true),
                    &euclidean,
                );

                assert_eq!(results.len(), 5);
                let tie_count = results
                    .iter()
                    .filter(|(_, dist)| (*dist - 2.0).abs() < 1e-10)
                    .count();
                assert_eq!(tie_count, 4);
            }

            for k in 6..10 {
                let results = tree.neighbors_with_distance(
                    0.,
                    0.,
                    NeighborsOptions::k(k).include_tie_breakers(true),
                    &euclidean,
                );

                assert_eq!(results.len(), 7);
                let tie_count = results
                    .iter()
                    .filter(|(_, dist)| (*dist - 2.0).abs() < 1e-10)
                    .count();
                assert_eq!(tie_count, 4);
                let tie_count = results
                    .iter()
                    .filter(|(_, dist)| (*dist - 3.0).abs() < 1e-10)
                    .count();
                assert_eq!(tie_count, 2);
            }
        }

        #[test]
        #[cfg(feature = "use-geo_0_31")]
        fn test_geometry_neighbors_euclidean() {
            use crate::r#type::IndexableNum;
            use crate::rtree::distance::{DistanceMetric, SliceGeometryAccessor};
            use geo_0_31::algorithm::{Distance, Euclidean};
            use geo_0_31::{Geometry, Point};

            let mut builder = RTreeBuilder::<f64>::new(3);
            builder.add(0., 0., 2., 2.); // Item 0
            builder.add(5., 5., 7., 7.); // Item 1
            builder.add(10., 10., 12., 12.); // Item 2
            let tree = builder.finish::<HilbertSort>();

            // Geometries corresponding to the bboxes
            let geometries: Vec<Geometry<f64>> = vec![
                Geometry::Point(Point::new(1.0, 1.0)),   // Item 0
                Geometry::Point(Point::new(6.0, 6.0)),   // Item 1
                Geometry::Point(Point::new(11.0, 11.0)), // Item 2
            ];

            struct SimpleMetric;
            impl<N: IndexableNum> SimpleDistanceMetric<N> for SimpleMetric {
                fn distance(&self, x1: N, y1: N, x2: N, y2: N) -> N {
                    let dx = x2 - x1;
                    let dy = y2 - y1;
                    (dx * dx + dy * dy).sqrt().unwrap_or(N::max_value())
                }
                fn distance_to_bbox(
                    &self,
                    x: N,
                    y: N,
                    min_x: N,
                    min_y: N,
                    max_x: N,
                    max_y: N,
                ) -> N {
                    let dx = if x < min_x {
                        min_x - x
                    } else if x > max_x {
                        x - max_x
                    } else {
                        N::zero()
                    };
                    let dy = if y < min_y {
                        min_y - y
                    } else if y > max_y {
                        y - max_y
                    } else {
                        N::zero()
                    };
                    (dx * dx + dy * dy).sqrt().unwrap_or(N::max_value())
                }
            }
            impl<N: IndexableNum> DistanceMetric<N> for SimpleMetric {
                fn distance_to_geometry(&self, geom1: &Geometry<f64>, geom2: &Geometry<f64>) -> N {
                    N::from_f64(Euclidean.distance(geom1, geom2)).unwrap_or(N::max_value())
                }
            }

            let query_geom = Geometry::Point(Point::new(3.0, 3.0));
            let metric = SimpleMetric;
            let accessor = SliceGeometryAccessor::new(&geometries);
            let results =
                tree.neighbors_geometry(&query_geom, NeighborsOptions::all(), &metric, &accessor);

            // Item 0 should be closest to query point (3,3)
            assert_eq!(results[0].0, 0);
            assert_eq!(results[1].0, 1);
            assert_eq!(results[2].0, 2);
        }

        #[test]
        #[cfg(feature = "use-geo_0_31")]
        fn test_geometry_neighbors_linestring() {
            use crate::r#type::IndexableNum;
            use crate::rtree::distance::{DistanceMetric, SliceGeometryAccessor};
            use geo_0_31::algorithm::{Distance, Euclidean};
            use geo_0_31::{coord, Geometry, LineString, Point};

            let mut builder = RTreeBuilder::<f64>::new(3);
            builder.add(0., 0., 10., 0.); // Item 0 - horizontal line
            builder.add(5., 5., 15., 5.); // Item 1 - horizontal line higher up
            builder.add(0., 10., 10., 10.); // Item 2 - horizontal line at top
            let tree = builder.finish::<HilbertSort>();

            // Geometries corresponding to the bboxes
            let geometries: Vec<Geometry<f64>> = vec![
                Geometry::LineString(LineString::new(vec![
                    coord! { x: 0.0, y: 0.0 },
                    coord! { x: 10.0, y: 0.0 },
                ])),
                Geometry::LineString(LineString::new(vec![
                    coord! { x: 5.0, y: 5.0 },
                    coord! { x: 15.0, y: 5.0 },
                ])),
                Geometry::LineString(LineString::new(vec![
                    coord! { x: 0.0, y: 10.0 },
                    coord! { x: 10.0, y: 10.0 },
                ])),
            ];

            struct SimpleMetric;
            impl<N: IndexableNum> SimpleDistanceMetric<N> for SimpleMetric {
                fn distance(&self, x1: N, y1: N, x2: N, y2: N) -> N {
                    let dx = x2 - x1;
                    let dy = y2 - y1;
                    (dx * dx + dy * dy).sqrt().unwrap_or(N::max_value())
                }
                fn distance_to_bbox(
                    &self,
                    x: N,
                    y: N,
                    min_x: N,
                    min_y: N,
                    max_x: N,
                    max_y: N,
                ) -> N {
                    let dx = if x < min_x {
                        min_x - x
                    } else if x > max_x {
                        x - max_x
                    } else {
                        N::zero()
                    };
                    let dy = if y < min_y {
                        min_y - y
                    } else if y > max_y {
                        y - max_y
                    } else {
                        N::zero()
                    };
                    (dx * dx + dy * dy).sqrt().unwrap_or(N::max_value())
                }
            }
            impl<N: IndexableNum> DistanceMetric<N> for SimpleMetric {
                fn distance_to_geometry(&self, geom1: &Geometry<f64>, geom2: &Geometry<f64>) -> N {
                    N::from_f64(Euclidean.distance(geom1, geom2)).unwrap_or(N::max_value())
                }
            }

            let query_geom = Geometry::Point(Point::new(5.0, 2.0));
            let metric = SimpleMetric;
            let accessor = SliceGeometryAccessor::new(&geometries);
            let results =
                tree.neighbors_geometry(&query_geom, NeighborsOptions::all(), &metric, &accessor);

            // Item 0 (bottom line) should be closest to point (5, 2)
            assert_eq!(results[0].0, 0);
        }

        #[test]
        #[cfg(feature = "use-geo_0_31")]
        fn test_geometry_neighbors_with_max_results() {
            use crate::r#type::IndexableNum;
            use crate::rtree::distance::{DistanceMetric, SliceGeometryAccessor};
            use geo_0_31::algorithm::{Distance, Euclidean};
            use geo_0_31::{Geometry, Point};

            let mut builder = RTreeBuilder::<f64>::new(5);
            for i in 0..5 {
                let x = (i * 3) as f64;
                builder.add(x, x, x + 1., x + 1.);
            }
            let tree = builder.finish::<HilbertSort>();

            // Create geometries for each bbox
            let geometries: Vec<Geometry<f64>> = (0..5)
                .map(|i| {
                    let x = (i * 3) as f64;
                    Geometry::Point(Point::new(x + 0.5, x + 0.5))
                })
                .collect();

            struct SimpleMetric;
            impl<N: IndexableNum> SimpleDistanceMetric<N> for SimpleMetric {
                fn distance(&self, x1: N, y1: N, x2: N, y2: N) -> N {
                    let dx = x2 - x1;
                    let dy = y2 - y1;
                    (dx * dx + dy * dy).sqrt().unwrap_or(N::max_value())
                }
                fn distance_to_bbox(
                    &self,
                    x: N,
                    y: N,
                    min_x: N,
                    min_y: N,
                    max_x: N,
                    max_y: N,
                ) -> N {
                    let dx = if x < min_x {
                        min_x - x
                    } else if x > max_x {
                        x - max_x
                    } else {
                        N::zero()
                    };
                    let dy = if y < min_y {
                        min_y - y
                    } else if y > max_y {
                        y - max_y
                    } else {
                        N::zero()
                    };
                    (dx * dx + dy * dy).sqrt().unwrap_or(N::max_value())
                }
            }
            impl<N: IndexableNum> DistanceMetric<N> for SimpleMetric {
                fn distance_to_geometry(&self, geom1: &Geometry<f64>, geom2: &Geometry<f64>) -> N {
                    N::from_f64(Euclidean.distance(geom1, geom2)).unwrap_or(N::max_value())
                }
            }

            let query_geom = Geometry::Point(Point::new(5.0, 5.0));
            let metric = SimpleMetric;
            let accessor = SliceGeometryAccessor::new(&geometries);
            let results =
                tree.neighbors_geometry(&query_geom, NeighborsOptions::k(3), &metric, &accessor);

            assert_eq!(results.len(), 3);
            // Should get the 3 closest items
        }

        #[test]
        #[cfg(feature = "use-geo_0_31")]
        fn test_geometry_neighbors_haversine() {
            use crate::r#type::IndexableNum;
            use crate::rtree::distance::{DistanceMetric, SliceGeometryAccessor};
            use geo_0_31::algorithm::{Centroid, Distance, Haversine};
            use geo_0_31::{Geometry, Point};

            let mut builder = RTreeBuilder::<f64>::new(3);
            // Geographic bounding boxes (lon, lat)
            builder.add(-74.1, 40.6, -74.0, 40.7); // New York area
            builder.add(-0.2, 51.4, -0.1, 51.5); // London area
            builder.add(139.6, 35.6, 139.7, 35.7); // Tokyo area
            let tree = builder.finish::<HilbertSort>();

            let geometries: Vec<Geometry<f64>> = vec![
                Geometry::Point(Point::new(-74.0, 40.7)), // New York
                Geometry::Point(Point::new(-0.1, 51.5)),  // London
                Geometry::Point(Point::new(139.7, 35.7)), // Tokyo
            ];

            struct HaversineMetric;
            impl<N: IndexableNum> SimpleDistanceMetric<N> for HaversineMetric {
                fn distance(&self, lon1: N, lat1: N, lon2: N, lat2: N) -> N {
                    let p1 = Point::new(lon1.to_f64().unwrap_or(0.0), lat1.to_f64().unwrap_or(0.0));
                    let p2 = Point::new(lon2.to_f64().unwrap_or(0.0), lat2.to_f64().unwrap_or(0.0));
                    N::from_f64(Haversine.distance(p1, p2)).unwrap_or(N::max_value())
                }
                fn distance_to_bbox(
                    &self,
                    lon: N,
                    lat: N,
                    min_lon: N,
                    min_lat: N,
                    max_lon: N,
                    max_lat: N,
                ) -> N {
                    let lon_f = lon.to_f64().unwrap_or(0.0);
                    let lat_f = lat.to_f64().unwrap_or(0.0);
                    let min_lon_f = min_lon.to_f64().unwrap_or(0.0);
                    let min_lat_f = min_lat.to_f64().unwrap_or(0.0);
                    let max_lon_f = max_lon.to_f64().unwrap_or(0.0);
                    let max_lat_f = max_lat.to_f64().unwrap_or(0.0);
                    let closest_lon = lon_f.clamp(min_lon_f, max_lon_f);
                    let closest_lat = lat_f.clamp(min_lat_f, max_lat_f);
                    let point = Point::new(lon_f, lat_f);
                    let closest_point = Point::new(closest_lon, closest_lat);
                    N::from_f64(Haversine.distance(point, closest_point)).unwrap_or(N::max_value())
                }
            }
            impl<N: IndexableNum> DistanceMetric<N> for HaversineMetric {
                fn distance_to_geometry(&self, geom1: &Geometry<f64>, geom2: &Geometry<f64>) -> N {
                    let c1 = geom1.centroid().unwrap_or(Point::new(0.0, 0.0));
                    let c2 = geom2.centroid().unwrap_or(Point::new(0.0, 0.0));
                    N::from_f64(Haversine.distance(c1, c2)).unwrap_or(N::max_value())
                }
            }

            let query_geom = Geometry::Point(Point::new(-74.0, 40.7)); // New York
            let metric = HaversineMetric;
            let accessor = SliceGeometryAccessor::new(&geometries);
            let results =
                tree.neighbors_geometry(&query_geom, NeighborsOptions::all(), &metric, &accessor);

            // New York should be closest (distance 0)
            assert_eq!(results[0].0, 0);
        }

        #[test]
        fn test_distance_values_returned() {
            let mut builder = RTreeBuilder::<f64>::new(3);
            builder.add(0., 0., 0., 0.); // Item 0: distance 0
            builder.add(3., 0., 3., 0.); // Item 1: distance 3
            builder.add(0., 4., 0., 4.); // Item 2: distance 4
            let tree = builder.finish::<HilbertSort>();

            let euclidean = EuclideanDistance;
            let results = tree.neighbors_with_distance(0., 0., NeighborsOptions::all(), &euclidean);

            // Verify distance values are correct
            assert_eq!(results.len(), 3);
            assert_eq!(results[0], (0, 0.0));
            assert_eq!(results[1], (1, 3.0));
            assert_eq!(results[2], (2, 4.0));
        }

        #[test]
        fn test_geometry_tie_breakers() {
            use crate::r#type::IndexableNum;
            use crate::rtree::distance::{DistanceMetric, SliceGeometryAccessor};
            use geo_0_31::algorithm::{Distance, Euclidean};
            use geo_0_31::{Geometry, Point};

            let mut builder = RTreeBuilder::<f64>::new(4);
            builder.add(0., 0., 0., 0.); // Item 0
            builder.add(1., 0., 1., 0.); // Item 1
            builder.add(0., 2., 0., 2.); // Item 2
            builder.add(2., 0., 2., 0.); // Item 3 (same distance as item 2)
            let tree = builder.finish::<HilbertSort>();

            let geometries: Vec<Geometry<f64>> = vec![
                Geometry::Point(Point::new(0.0, 0.0)),
                Geometry::Point(Point::new(1.0, 0.0)),
                Geometry::Point(Point::new(0.0, 2.0)),
                Geometry::Point(Point::new(2.0, 0.0)),
            ];

            struct SimpleMetric;
            impl<N: IndexableNum> SimpleDistanceMetric<N> for SimpleMetric {
                fn distance(&self, x1: N, y1: N, x2: N, y2: N) -> N {
                    let dx = x2 - x1;
                    let dy = y2 - y1;
                    (dx * dx + dy * dy).sqrt().unwrap_or(N::max_value())
                }
                fn distance_to_bbox(
                    &self,
                    x: N,
                    y: N,
                    min_x: N,
                    min_y: N,
                    max_x: N,
                    max_y: N,
                ) -> N {
                    let dx = if x < min_x {
                        min_x - x
                    } else if x > max_x {
                        x - max_x
                    } else {
                        N::zero()
                    };
                    let dy = if y < min_y {
                        min_y - y
                    } else if y > max_y {
                        y - max_y
                    } else {
                        N::zero()
                    };
                    (dx * dx + dy * dy).sqrt().unwrap_or(N::max_value())
                }
            }
            impl<N: IndexableNum> DistanceMetric<N> for SimpleMetric {
                fn distance_to_geometry(&self, geom1: &Geometry<f64>, geom2: &Geometry<f64>) -> N {
                    N::from_f64(Euclidean.distance(geom1, geom2)).unwrap_or(N::max_value())
                }
            }

            let query_geom = Geometry::Point(Point::new(0.0, 0.0));
            let metric = SimpleMetric;
            let accessor = SliceGeometryAccessor::new(&geometries);

            // Test with tie breakers enabled
            let results = tree.neighbors_geometry(
                &query_geom,
                NeighborsOptions::k(3).include_tie_breakers(true),
                &metric,
                &accessor,
            );

            // Should return 4 items: item 0 (distance 0), item 1 (distance 1),
            // and both items 2 and 3 (both at distance 2, tied at k=3)
            assert_eq!(results.len(), 4);
            assert_eq!(results[0].0, 0);
            assert_eq!(results[1].0, 1);

            let indices: Vec<u32> = results.iter().map(|(idx, _)| *idx).collect();
            assert!(indices.contains(&2));
            assert!(indices.contains(&3));

            // Test with tie breakers disabled
            let results_no_ties =
                tree.neighbors_geometry(&query_geom, NeighborsOptions::k(3), &metric, &accessor);
            // Should return only 3 items: item 0 (distance 0), item 1 (distance 1),
            // and either item 2 or item 3 (but not both, since tie breakers are disabled)
            assert_eq!(results_no_ties.len(), 3);
            assert_eq!(results_no_ties[0].0, 0);
            assert_eq!(results_no_ties[1].0, 1);
            assert!(results_no_ties[2].0 == 2 || results_no_ties[2].0 == 3);
        }
    }
}
