//! Distance metrics for spatial queries.
//!
//! This module provides different distance calculation methods for spatial queries,
//! including Euclidean and Haversine distance calculations.
//!
//! This module is retained for compatibility with `geo` 0.31. For integrations
//! with other versions or geometry libraries, use
//! [`RTreeIndex::neighbors_with_callbacks`](crate::rtree::RTreeIndex::neighbors_with_callbacks),
//! which requires no optional feature.

use crate::r#type::IndexableNum;
use crate::rtree::r#trait::{axis_dist, SimpleDistanceMetric};
use geo_0_31::algorithm::{Distance, Euclidean, Haversine};
use geo_0_31::{coord, Geometry, Point, Rect};

pub use crate::rtree::r#trait::GeometryAccessor;

/// A trait for calculating distances between geometries and points.
///
/// This trait extends `SimpleDistanceMetric` to add geometry-to-geometry distance calculations.
pub trait DistanceMetric<N: IndexableNum>: SimpleDistanceMetric<N> {
    /// Calculate the distance between two geometries.
    /// This method is used by geometry-based neighbor searches to compute the actual
    /// distance between a query geometry and an item geometry.
    ///
    /// TODO: Consider changing to accept `&impl GeometryTrait<T = f64>` instead of concrete
    /// `Geometry<f64>` type for better flexibility and integration with geo-traits.
    /// This would be a non-breaking change since Geometry implements GeometryTrait.
    fn distance_to_geometry(&self, geom1: &Geometry<f64>, geom2: &Geometry<f64>) -> N;

    /// Calculate the distance from a geometry to a bounding box.
    ///
    /// This method is used for internal node distance estimation in geometry-based
    /// neighbor searches. The returned distance should be a lower bound on the actual
    /// distance from the query geometry to any geometry contained within the bounding box.
    ///
    /// The default implementation checks if the geometry is a Point, and if so, uses the
    /// faster `distance_to_bbox` method. Otherwise, it wraps the bounding box as a `Rect`
    /// and uses `distance_to_geometry`. Implementations may override this for better performance.
    fn distance_geometry_to_bbox(
        &self,
        geom: &Geometry<f64>,
        min_x: N,
        min_y: N,
        max_x: N,
        max_y: N,
    ) -> N {
        // Fast path for points: use distance_to_bbox directly
        if let Geometry::Point(p) = geom {
            let (Some(x), Some(y)) = (N::from_f64(p.x()), N::from_f64(p.y())) else {
                return self.max_distance();
            };
            self.distance_to_bbox(x, y, min_x, min_y, max_x, max_y)
        } else {
            // General case: wrap bbox as Rect and use distance_to_geometry
            let (Some(min_x), Some(min_y), Some(max_x), Some(max_y)) = (
                min_x.to_f64(),
                min_y.to_f64(),
                max_x.to_f64(),
                max_y.to_f64(),
            ) else {
                return self.max_distance();
            };
            let bbox = Rect::new(
                coord! {
                    x: min_x,
                    y: min_y
                },
                coord! {
                    x: max_x,
                    y: max_y
                },
            );
            self.distance_to_geometry(geom, &Geometry::Rect(bbox))
        }
    }
}

/// Euclidean distance metric.
///
/// This is the standard straight-line distance calculation suitable for
/// planar coordinate systems. When working with longitude/latitude coordinates,
/// the unit of distance will be degrees.
#[derive(Debug, Clone, Copy, Default)]
pub struct EuclideanDistance;

impl<N: IndexableNum> SimpleDistanceMetric<N> for EuclideanDistance {
    #[inline]
    fn distance(&self, x1: N, y1: N, x2: N, y2: N) -> N {
        let p1 = Point::new(x1.to_f64().unwrap_or(0.0), y1.to_f64().unwrap_or(0.0));
        let p2 = Point::new(x2.to_f64().unwrap_or(0.0), y2.to_f64().unwrap_or(0.0));
        N::from_f64(Euclidean.distance(p1, p2)).unwrap_or(N::max_value())
    }

    #[inline]
    fn distance_to_bbox(&self, x: N, y: N, min_x: N, min_y: N, max_x: N, max_y: N) -> N {
        let dx = axis_dist(x, min_x, max_x);
        let dy = axis_dist(y, min_y, max_y);
        (dx * dx + dy * dy).sqrt().unwrap_or(N::max_value())
    }
}

impl<N: IndexableNum> DistanceMetric<N> for EuclideanDistance {
    fn distance_to_geometry(&self, geom1: &Geometry<f64>, geom2: &Geometry<f64>) -> N {
        N::from_f64(Euclidean.distance(geom1, geom2)).unwrap_or(N::max_value())
    }
}

/// Haversine distance metric.
///
/// This calculates the great-circle distance between two points on a sphere.
/// It's more accurate for geographic distances than Euclidean distance.
/// The input coordinates should be in longitude/latitude (degrees), and
/// the output distance is in meters.
#[derive(Debug, Clone, Copy)]
pub struct HaversineDistance {
    /// Earth's radius in meters
    pub earth_radius: f64,
}

impl Default for HaversineDistance {
    fn default() -> Self {
        Self {
            earth_radius: 6378137.0, // WGS84 equatorial radius in meters
        }
    }
}

impl HaversineDistance {
    /// Create a new Haversine distance metric with custom Earth radius.
    pub fn with_radius(earth_radius: f64) -> Self {
        Self { earth_radius }
    }
}

impl<N: IndexableNum> SimpleDistanceMetric<N> for HaversineDistance {
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
        // For geographic distance to bbox, find the closest point on the bbox
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

impl<N: IndexableNum> DistanceMetric<N> for HaversineDistance {
    fn distance_to_geometry(&self, geom1: &Geometry<f64>, geom2: &Geometry<f64>) -> N {
        // For Haversine, use centroid-to-centroid distance as approximation
        use geo_0_31::algorithm::Centroid;
        let c1 = geom1.centroid().unwrap_or(Point::new(0.0, 0.0));
        let c2 = geom2.centroid().unwrap_or(Point::new(0.0, 0.0));
        N::from_f64(Haversine.distance(c1, c2)).unwrap_or(N::max_value())
    }
}

/// Simple geometry accessor that wraps a slice of geometries.
///
/// This accessor provides access to geometries by index for use with distance metrics.
///
/// # Example
/// ```
/// use geo_index::rtree::distance::{EuclideanDistance, SliceGeometryAccessor};
/// use geo_0_31::{Geometry, Point};
///
/// let geometries = vec![
///     Geometry::Point(Point::new(0.0, 0.0)),
///     Geometry::Point(Point::new(1.0, 1.0)),
/// ];
///
/// let accessor = SliceGeometryAccessor::new(&geometries);
/// let metric = EuclideanDistance;
/// // Now accessor and metric can be used with neighbors_geometry
/// ```
pub struct SliceGeometryAccessor<'a> {
    geometries: &'a [Geometry<f64>],
}

impl<'a> SliceGeometryAccessor<'a> {
    /// Create a new accessor with the given geometries.
    pub fn new(geometries: &'a [Geometry<f64>]) -> Self {
        Self { geometries }
    }
}

impl<'a> GeometryAccessor for SliceGeometryAccessor<'a> {
    fn get_geometry(&self, item_index: usize) -> Option<&Geometry<f64>> {
        self.geometries.get(item_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_0_31::{coord, LineString};

    #[test]
    fn test_euclidean_distance() {
        let metric = EuclideanDistance;
        let distance = metric.distance(0.0f64, 0.0f64, 3.0f64, 4.0f64);
        assert!((distance - 5.0f64).abs() < 1e-10);
    }

    #[test]
    fn test_haversine_distance() {
        let metric = HaversineDistance::default();
        // Distance between New York and London (approximately)
        let distance = metric.distance(-74.0f64, 40.7f64, -0.1f64, 51.5f64);
        // Should be approximately 5585 km
        assert!((distance - 5585000.0f64).abs() < 50000.0f64);
    }

    #[test]
    fn test_euclidean_geometry_distance() {
        // Test Euclidean distance between geometries
        let point1 = Geometry::Point(Point::new(0.0, 0.0));
        let point2 = Geometry::Point(Point::new(3.0, 4.0));
        let distance: f64 = Euclidean.distance(&point1, &point2);
        assert!((distance - 5.0).abs() < 1e-10);

        // Test distance to line
        let line = Geometry::LineString(LineString::new(vec![
            coord! { x: 0.0, y: 5.0 },
            coord! { x: 10.0, y: 5.0 },
        ]));
        let query = Geometry::Point(Point::new(0.0, 0.0));
        let distance: f64 = Euclidean.distance(&query, &line);
        assert!((distance - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_wkb_decoding_distance_metric() {
        use geozero::{wkb, GeozeroGeometry};

        /// Custom distance metric that stores WKB-encoded geometries and decodes them on-demand
        struct WkbDistanceMetric<'a> {
            wkb_data: &'a [Vec<u8>], // Array of WKB-encoded geometries
        }

        impl<'a> WkbDistanceMetric<'a> {
            fn new(wkb_data: &'a [Vec<u8>]) -> Self {
                Self { wkb_data }
            }

            /// Decode WKB data on-demand to get geometry
            fn decode_geometry(&self, index: usize) -> Option<Geometry<f64>> {
                if index < self.wkb_data.len() {
                    use geozero::geo_types::GeoWriter;

                    let mut geo_writer = GeoWriter::new();
                    // Pass the byte slice directly to Wkb
                    if wkb::Wkb(self.wkb_data[index].as_slice())
                        .process_geom(&mut geo_writer)
                        .is_ok()
                    {
                        geo_writer.take_geometry()
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        }

        impl<'a, N: IndexableNum> SimpleDistanceMetric<N> for WkbDistanceMetric<'a> {
            fn distance(&self, x1: N, y1: N, x2: N, y2: N) -> N {
                EuclideanDistance.distance(x1, y1, x2, y2)
            }

            fn distance_to_bbox(&self, x: N, y: N, min_x: N, min_y: N, max_x: N, max_y: N) -> N {
                EuclideanDistance.distance_to_bbox(x, y, min_x, min_y, max_x, max_y)
            }
        }

        impl<'a, N: IndexableNum> DistanceMetric<N> for WkbDistanceMetric<'a> {
            fn distance_to_geometry(&self, geom1: &Geometry<f64>, geom2: &Geometry<f64>) -> N {
                N::from_f64(Euclidean.distance(geom1, geom2)).unwrap_or(N::max_value())
            }
        }

        // Create some test WKB data (encoded points)
        let point1 = Geometry::Point(Point::new(0.0, 0.0));
        let point2 = Geometry::Point(Point::new(3.0, 4.0));
        let point3 = Geometry::Point(Point::new(6.0, 8.0));

        // Encode geometries to WKB using geozero
        use geozero::ToWkb;
        let wkb1 = point1.to_wkb(geozero::CoordDimensions::default()).unwrap();
        let wkb2 = point2.to_wkb(geozero::CoordDimensions::default()).unwrap();
        let wkb3 = point3.to_wkb(geozero::CoordDimensions::default()).unwrap();
        let wkb_data = vec![wkb1, wkb2, wkb3];

        // Create the WKB-based distance metric
        let wkb_metric = WkbDistanceMetric::new(&wkb_data);
        let query = Geometry::Point(Point::new(1.0, 1.0));

        // Test distance calculation with on-demand WKB decoding
        // Decode geometries and compute distances
        let geom0 = wkb_metric.decode_geometry(0).unwrap();
        let dist: f64 = wkb_metric.distance_to_geometry(&query, &geom0);
        assert!((dist - 1.414).abs() < 0.01); // Distance from (1,1) to (0,0)

        let geom1 = wkb_metric.decode_geometry(1).unwrap();
        let dist: f64 = wkb_metric.distance_to_geometry(&query, &geom1);
        assert!((dist - 3.605).abs() < 0.01); // Distance from (1,1) to (3,4)

        let geom2 = wkb_metric.decode_geometry(2).unwrap();
        let dist: f64 = wkb_metric.distance_to_geometry(&query, &geom2);
        assert!((dist - 8.602).abs() < 0.01); // Distance from (1,1) to (6,8)
    }

    #[test]
    fn test_cached_geometry_distance_metric() {
        use std::cell::RefCell;
        use std::collections::HashMap;

        /// Custom distance metric with geometry caching to avoid repeated calculations
        struct CachedDistanceMetric<'a> {
            geometries: &'a [Geometry<f64>],
            cache: RefCell<HashMap<usize, Geometry<f64>>>, // Cache for decoded geometries
            cache_hits: RefCell<usize>,                    // Track cache performance
            cache_misses: RefCell<usize>,
        }

        impl<'a> CachedDistanceMetric<'a> {
            fn new(geometries: &'a [Geometry<f64>]) -> Self {
                Self {
                    geometries,
                    cache: RefCell::new(HashMap::new()),
                    cache_hits: RefCell::new(0),
                    cache_misses: RefCell::new(0),
                }
            }

            /// Get geometry with caching - simulates expensive decode operation
            fn get_cached_geometry(&self, index: usize) -> Option<Geometry<f64>> {
                if index >= self.geometries.len() {
                    return None;
                }

                // Check cache first
                if let Some(cached_geom) = self.cache.borrow().get(&index) {
                    *self.cache_hits.borrow_mut() += 1;
                    return Some(cached_geom.clone());
                }

                // Cache miss - "expensive" operation simulation
                *self.cache_misses.borrow_mut() += 1;
                let geometry = self.geometries[index].clone();

                // Store in cache
                self.cache.borrow_mut().insert(index, geometry.clone());
                Some(geometry)
            }

            fn get_cache_stats(&self) -> (usize, usize) {
                (*self.cache_hits.borrow(), *self.cache_misses.borrow())
            }
        }

        impl<'a, N: IndexableNum> SimpleDistanceMetric<N> for CachedDistanceMetric<'a> {
            fn distance(&self, x1: N, y1: N, x2: N, y2: N) -> N {
                EuclideanDistance.distance(x1, y1, x2, y2)
            }

            fn distance_to_bbox(&self, x: N, y: N, min_x: N, min_y: N, max_x: N, max_y: N) -> N {
                EuclideanDistance.distance_to_bbox(x, y, min_x, min_y, max_x, max_y)
            }
        }

        impl<'a, N: IndexableNum> DistanceMetric<N> for CachedDistanceMetric<'a> {
            fn distance_to_geometry(&self, geom1: &Geometry<f64>, geom2: &Geometry<f64>) -> N {
                N::from_f64(Euclidean.distance(geom1, geom2)).unwrap_or(N::max_value())
            }
        }

        // Create test data
        let geometries = vec![
            Geometry::Point(Point::new(0.0, 0.0)),
            Geometry::Point(Point::new(3.0, 4.0)),
            Geometry::Point(Point::new(6.0, 8.0)),
        ];

        let cached_metric = CachedDistanceMetric::new(&geometries);
        let query = Geometry::Point(Point::new(1.0, 1.0));

        // First access - should be cache misses
        let geom0 = cached_metric.get_cached_geometry(0).unwrap();
        let dist1: f64 = cached_metric.distance_to_geometry(&query, &geom0);

        let geom1 = cached_metric.get_cached_geometry(1).unwrap();
        let dist2: f64 = cached_metric.distance_to_geometry(&query, &geom1);

        let geom2 = cached_metric.get_cached_geometry(2).unwrap();
        let dist3: f64 = cached_metric.distance_to_geometry(&query, &geom2);

        assert!((dist1 - 1.414).abs() < 0.01);
        assert!((dist2 - 3.605).abs() < 0.01);
        assert!((dist3 - 8.602).abs() < 0.01);

        let (hits_after_first, misses_after_first) = cached_metric.get_cache_stats();
        assert_eq!(hits_after_first, 0); // No hits yet
        assert_eq!(misses_after_first, 3); // 3 misses

        // Second access to same geometries - should be cache hits
        let geom0_cached = cached_metric.get_cached_geometry(0).unwrap();
        let dist1_cached: f64 = cached_metric.distance_to_geometry(&query, &geom0_cached);

        let geom1_cached = cached_metric.get_cached_geometry(1).unwrap();
        let dist2_cached: f64 = cached_metric.distance_to_geometry(&query, &geom1_cached);

        assert!((dist1_cached - 1.414).abs() < 0.01);
        assert!((dist2_cached - 3.605).abs() < 0.01);

        let (hits_after_second, misses_after_second) = cached_metric.get_cache_stats();
        assert_eq!(hits_after_second, 2); // 2 cache hits
        assert_eq!(misses_after_second, 3); // Still 3 misses total
    }
}
