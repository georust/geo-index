//! Fuzz tests for neighbors_geometry function.
//!
//! These tests verify the correctness of the `neighbors_geometry` function by comparing
//! its results against brute-force ground truth calculations.

use crate::rtree::distance::{EuclideanDistance, SliceGeometryAccessor};
use crate::rtree::sort::HilbertSort;
use crate::rtree::{RTreeBuilder, RTreeIndex};
use geo_0_31::algorithm::{BoundingRect, Distance, Euclidean};
use geo_0_31::{coord, Coord, Geometry, LineString, Point, Polygon, Rect};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

/// Options for generating random geometries
#[derive(Debug, Clone)]
struct RandomGeometryOptions {
    /// Bounding box for geometry generation
    bounds: Rect,
    /// Size range for generated geometries (min, max)
    size_range: (f64, f64),
    /// Number of vertices for polygons (min, max)
    vertices_per_polygon_range: (usize, usize),
}

impl Default for RandomGeometryOptions {
    fn default() -> Self {
        Self {
            bounds: Rect::new(Coord { x: 0.0, y: 0.0 }, Coord { x: 100.0, y: 100.0 }),
            size_range: (1.0, 10.0),
            vertices_per_polygon_range: (4, 8),
        }
    }
}

/// Generate a random point within the given bounds
fn generate_random_point<R: Rng>(rng: &mut R, options: &RandomGeometryOptions) -> Point {
    Point::new(
        rng.random_range(options.bounds.min().x..options.bounds.max().x),
        rng.random_range(options.bounds.min().y..options.bounds.max().y),
    )
}

/// Generate a random polygon within the given bounds
fn generate_random_polygon<R: Rng>(rng: &mut R, options: &RandomGeometryOptions) -> Polygon {
    // Generate random center and size
    let half_size = rng.random_range(options.size_range.0..options.size_range.1) / 2.0;

    // Ensure polygon fits within bounds by constraining center position
    let center_x = rng
        .random_range((options.bounds.min().x + half_size)..(options.bounds.max().x - half_size));
    let center_y = rng
        .random_range((options.bounds.min().y + half_size)..(options.bounds.max().y - half_size));

    // Generate circular vertices
    let num_vertices = rng
        .random_range(options.vertices_per_polygon_range.0..=options.vertices_per_polygon_range.1)
        .max(3);

    let mut coords = Vec::with_capacity(num_vertices + 1);
    let mut angle: f64 = rng.random_range(0.0..(2.0 * PI));
    let dangle = 2.0 * PI / num_vertices as f64;

    for _ in 0..num_vertices {
        coords.push(coord! {
            x: angle.cos() * half_size + center_x,
            y: angle.sin() * half_size + center_y,
        });
        angle += dangle;
    }
    // Close the ring
    coords.push(coords[0]);

    Polygon::new(LineString::from(coords), vec![])
}

/// Generate a vector of random points
fn generate_random_points(
    seed: u64,
    count: usize,
    options: &RandomGeometryOptions,
) -> Vec<Geometry<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| Geometry::Point(generate_random_point(&mut rng, options)))
        .collect()
}

/// Generate a vector of random polygons
fn generate_random_polygons(
    seed: u64,
    count: usize,
    options: &RandomGeometryOptions,
) -> Vec<Geometry<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| Geometry::Polygon(generate_random_polygon(&mut rng, options)))
        .collect()
}

/// Compute K nearest neighbors using brute force for ground truth
fn compute_knn_ground_truth(
    query_geometry: &Geometry<f64>,
    indexed_geometries: &[Geometry<f64>],
    k: usize,
) -> Vec<(usize, f64)> {
    let mut distances: Vec<(usize, f64)> = indexed_geometries
        .iter()
        .enumerate()
        .map(|(idx, geom)| (idx, Euclidean.distance(query_geometry, geom)))
        .collect();

    // Sort by distance, then by index for stability
    distances.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

    distances.into_iter().take(k).collect()
}

/// Build an RTree from geometries
fn build_rtree_from_geometries(geometries: &[Geometry<f64>]) -> crate::rtree::RTree<f64> {
    let mut builder = RTreeBuilder::<f64>::new(geometries.len() as u32);
    for geom in geometries {
        let rect = geom.bounding_rect().unwrap();
        builder.add(rect.min().x, rect.min().y, rect.max().x, rect.max().y);
    }
    builder.finish::<HilbertSort>()
}

/// Verify that neighbors_geometry returns correct results by comparing with ground truth
fn verify_neighbors_geometry(
    query_geometry: &Geometry<f64>,
    indexed_geometries: &[Geometry<f64>],
    k: usize,
    test_description: &str,
) {
    let tree = build_rtree_from_geometries(indexed_geometries);
    let metric = EuclideanDistance;
    let accessor = SliceGeometryAccessor::new(indexed_geometries);

    // Get results from neighbors_geometry
    let rtree_results = tree.neighbors_geometry(query_geometry, Some(k), None, &metric, &accessor);

    // Get ground truth
    let ground_truth = compute_knn_ground_truth(query_geometry, indexed_geometries, k);

    // Compute distances for rtree results
    let rtree_with_distances: Vec<(usize, f64)> = rtree_results
        .iter()
        .map(|&idx| {
            let dist = Euclidean.distance(query_geometry, &indexed_geometries[idx as usize]);
            (idx as usize, dist)
        })
        .collect();

    // Check that results are in non-decreasing distance order
    for i in 1..rtree_with_distances.len() {
        let prev_dist = rtree_with_distances[i - 1].1;
        let curr_dist = rtree_with_distances[i].1;
        assert!(
            prev_dist <= curr_dist + 1e-10, // Small epsilon for floating point
            "neighbors_geometry returned results out of order at position {} in {}: \
             idx {} has dist {}, but previous idx {} has dist {}",
            i,
            test_description,
            rtree_with_distances[i].0,
            curr_dist,
            rtree_with_distances[i - 1].0,
            prev_dist
        );
    }

    // Verify we got the same number of results
    assert_eq!(
        rtree_results.len(),
        ground_truth.len(),
        "neighbors_geometry returned wrong number of results for: {}",
        test_description
    );

    // Verify we got the correct set of K nearest neighbors (order may differ for ties)
    // Group by distance to handle ties
    let rtree_max_dist = rtree_with_distances.last().map(|(_, d)| *d).unwrap_or(0.0);
    let ground_truth_max_dist = ground_truth.last().map(|(_, d)| *d).unwrap_or(0.0);

    // The maximum distance should be approximately the same
    assert!(
        (rtree_max_dist - ground_truth_max_dist).abs() < 1e-10,
        "neighbors_geometry returned different K-th distance for {}: got {} expected {}",
        test_description,
        rtree_max_dist,
        ground_truth_max_dist
    );

    // Verify that all returned items are among the true K nearest neighbors
    // (allowing for ties at the boundary)
    let rtree_indices: std::collections::HashSet<usize> =
        rtree_results.iter().map(|&idx| idx as usize).collect();

    // Check that all rtree results have distances <= K-th distance
    for (idx, dist) in &rtree_with_distances {
        assert!(
            *dist <= ground_truth_max_dist + 1e-10,
            "neighbors_geometry returned item {} with distance {} which exceeds K-th distance {} in {}",
            idx, dist, ground_truth_max_dist, test_description
        );
    }

    // Check that no ground truth result was missed (should have same indices for non-tie cases)
    // For strict non-tie cases, indices should match
    let ground_truth_non_boundary: Vec<usize> = ground_truth
        .iter()
        .filter(|(_, d)| (*d - ground_truth_max_dist).abs() > 1e-10)
        .map(|(idx, _)| *idx)
        .collect();

    for idx in &ground_truth_non_boundary {
        assert!(
            rtree_indices.contains(idx),
            "neighbors_geometry missed item {} which is strictly closer than K-th neighbor in {}",
            idx,
            test_description
        );
    }
}

/// Test case: index points, query using point
#[test]
fn test_neighbors_geometry_point_index_point_query() {
    let options = RandomGeometryOptions::default();

    for seed in 0..20 {
        let indexed_geometries = generate_random_points(seed, 50, &options);
        let query_geometries = generate_random_points(seed + 1000, 10, &options);

        for (query_idx, query_geom) in query_geometries.iter().enumerate() {
            verify_neighbors_geometry(
                query_geom,
                &indexed_geometries,
                5,
                &format!("point_index_point_query seed={} query={}", seed, query_idx),
            );
        }
    }
}

/// Test case: index points, query using polygon
#[test]
fn test_neighbors_geometry_point_index_polygon_query() {
    let options = RandomGeometryOptions::default();

    for seed in 0..20 {
        let indexed_geometries = generate_random_points(seed, 50, &options);
        let query_geometries = generate_random_polygons(seed + 1000, 10, &options);

        for (query_idx, query_geom) in query_geometries.iter().enumerate() {
            verify_neighbors_geometry(
                query_geom,
                &indexed_geometries,
                5,
                &format!(
                    "point_index_polygon_query seed={} query={}",
                    seed, query_idx
                ),
            );
        }
    }
}

/// Test case: index polygons, query using point
#[test]
fn test_neighbors_geometry_polygon_index_point_query() {
    let options = RandomGeometryOptions::default();

    for seed in 0..20 {
        let indexed_geometries = generate_random_polygons(seed, 50, &options);
        let query_geometries = generate_random_points(seed + 1000, 10, &options);

        for (query_idx, query_geom) in query_geometries.iter().enumerate() {
            verify_neighbors_geometry(
                query_geom,
                &indexed_geometries,
                5,
                &format!(
                    "polygon_index_point_query seed={} query={}",
                    seed, query_idx
                ),
            );
        }
    }
}

/// Test case: index polygons, query using polygon
#[test]
fn test_neighbors_geometry_polygon_index_polygon_query() {
    let options = RandomGeometryOptions::default();

    for seed in 0..20 {
        let indexed_geometries = generate_random_polygons(seed, 50, &options);
        let query_geometries = generate_random_polygons(seed + 1000, 10, &options);

        for (query_idx, query_geom) in query_geometries.iter().enumerate() {
            verify_neighbors_geometry(
                query_geom,
                &indexed_geometries,
                5,
                &format!(
                    "polygon_index_polygon_query seed={} query={}",
                    seed, query_idx
                ),
            );
        }
    }
}

/// Test with mixed geometry sizes
#[test]
fn test_neighbors_geometry_mixed_sizes() {
    let mut rng = StdRng::seed_from_u64(42);

    // Generate polygons with varying sizes
    let mut indexed_geometries = Vec::new();
    for _ in 0..50 {
        let size_range = if rng.random_bool(0.5) {
            (0.5, 2.0) // Small polygons
        } else {
            (10.0, 30.0) // Large polygons
        };
        let options = RandomGeometryOptions {
            bounds: Rect::new(Coord { x: 0.0, y: 0.0 }, Coord { x: 100.0, y: 100.0 }),
            size_range,
            vertices_per_polygon_range: (4, 8),
        };
        indexed_geometries.push(Geometry::Polygon(generate_random_polygon(
            &mut rng, &options,
        )));
    }

    let options = RandomGeometryOptions::default();
    let query_geometries = generate_random_polygons(1000, 10, &options);

    for (query_idx, query_geom) in query_geometries.iter().enumerate() {
        verify_neighbors_geometry(
            query_geom,
            &indexed_geometries,
            5,
            &format!("mixed_sizes query={}", query_idx),
        );
    }
}

/// Test requesting more neighbors than available
#[test]
fn test_neighbors_geometry_k_larger_than_dataset() {
    let options = RandomGeometryOptions::default();
    let indexed_geometries = generate_random_polygons(42, 5, &options);
    let query_geom = Geometry::Point(Point::new(50.0, 50.0));

    let tree = build_rtree_from_geometries(&indexed_geometries);
    let metric = EuclideanDistance;
    let accessor = SliceGeometryAccessor::new(&indexed_geometries);

    // Request 10 neighbors but only 5 are available
    let rtree_results = tree.neighbors_geometry(&query_geom, Some(10), None, &metric, &accessor);
    let ground_truth = compute_knn_ground_truth(&query_geom, &indexed_geometries, 10);

    assert_eq!(rtree_results.len(), 5);
    assert_eq!(rtree_results.len(), ground_truth.len());

    let ground_truth_indices: Vec<usize> = ground_truth.iter().map(|(idx, _)| *idx).collect();
    let rtree_indices: Vec<usize> = rtree_results.iter().map(|&idx| idx as usize).collect();
    assert_eq!(rtree_indices, ground_truth_indices);
}

/// Test with max_distance constraint
#[test]
fn test_neighbors_geometry_with_max_distance() {
    let options = RandomGeometryOptions::default();

    for seed in 0..10 {
        let indexed_geometries = generate_random_polygons(seed, 50, &options);
        let query_geom = Geometry::Point(Point::new(50.0, 50.0));
        let max_distance = 20.0;

        let tree = build_rtree_from_geometries(&indexed_geometries);
        let metric = EuclideanDistance;
        let accessor = SliceGeometryAccessor::new(&indexed_geometries);

        let rtree_results =
            tree.neighbors_geometry(&query_geom, None, Some(max_distance), &metric, &accessor);

        // Verify all returned results are within max_distance
        for &idx in &rtree_results {
            let dist = Euclidean.distance(&query_geom, &indexed_geometries[idx as usize]);
            assert!(
                dist <= max_distance,
                "Result at distance {} exceeds max_distance {} (seed={})",
                dist,
                max_distance,
                seed
            );
        }

        // Verify no closer geometries were missed
        for (idx, geom) in indexed_geometries.iter().enumerate() {
            let dist = Euclidean.distance(&query_geom, geom);
            if dist <= max_distance {
                assert!(
                    rtree_results.contains(&(idx as u32)),
                    "Geometry at index {} with distance {} should be in results but isn't (seed={})",
                    idx, dist, seed
                );
            }
        }
    }
}

// =============================================================================
// Empty tree tests
// =============================================================================

#[test]
fn test_search_empty_tree_returns_empty() {
    let builder = RTreeBuilder::<f64>::new(0);
    let tree = builder.finish::<HilbertSort>();

    let results = tree.search(0.0, 0.0, 100.0, 100.0);
    assert!(results.is_empty());
}

#[test]
fn test_neighbors_empty_tree_returns_empty() {
    let builder = RTreeBuilder::<f64>::new(0);
    let tree = builder.finish::<HilbertSort>();

    let results = tree.neighbors(50.0, 50.0, Some(10), None);
    assert!(results.is_empty());
}

#[test]
fn test_neighbors_geometry_empty_tree_returns_empty() {
    let builder = RTreeBuilder::<f64>::new(0);
    let tree = builder.finish::<HilbertSort>();

    let geometries: Vec<Geometry<f64>> = vec![];
    let metric = EuclideanDistance;
    let accessor = SliceGeometryAccessor::new(&geometries);
    let query_geom = Geometry::Point(Point::new(50.0, 50.0));

    let results = tree.neighbors_geometry(&query_geom, Some(10), None, &metric, &accessor);
    assert!(results.is_empty());
}
