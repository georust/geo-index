//! An immutable, ABI-stable RTree.
//!
//! ### Creation
//!
//! Use [`RTreeBuilder`] to construct an [`RTree`], which allows you to make queries.
//!
//! ### Search
//!
//! Use [`RTreeIndex::search`] to search an RTree given a bounding box query or
//! [`RTreeIndex::neighbors`] to find the nearest neighbors from a point.
//!
//! ### Persisting
//!
//! You can use [`RTree::into_inner`] to access the underlying `Vec<u8>` it contains.
//!
//! ### Recovering the index
//!
//! You can use [`RTreeRef::try_new`] to construct an RTree as a reference on an external byte
//! slice. If you don't know the coordinate type used in the index, you can use
//! [`CoordType::from_buffer`][crate::CoordType::from_buffer] to infer the coordinate type.
//!
//! ### Coordinate types
//!
//! Supported coordinate types implement [`IndexableNum`][crate::IndexableNum]. Note that float
//! `NaN` is not supported and may panic.
//!
//! ### Alternate sorting methods
//!
//! This crate allows for multiple sorting methods, implemented in [`sort`].
//!
//! ## Example
//!
//! ```
//! use geo_index::rtree::{RTreeBuilder, RTreeIndex, RTreeRef};
//! use geo_index::rtree::sort::HilbertSort;
//!
//! // Create an RTree
//! let mut builder = RTreeBuilder::<f64>::new(3);
//! builder.add(0., 0., 2., 2.);
//! builder.add(1., 1., 3., 3.);
//! builder.add(2., 2., 4., 4.);
//! let tree = builder.finish::<HilbertSort>();
//!
//! // Perform a search
//! assert_eq!(tree.search(0.5, 0.5, 1.5, 1.5), vec![0, 1]);
//!
//! // Convert to underlying buffer
//! let buffer = tree.into_inner();
//!
//! // Create tree as a reference onto this buffer
//! let tree_ref = RTreeRef::<f64>::try_new(&buffer).unwrap();
//!
//! // Perform search again
//! assert_eq!(tree_ref.search(0.5, 0.5, 1.5, 1.5), vec![0, 1]);
//! ```

mod builder;
mod constants;
#[cfg(feature = "use-geo_0_31")]
pub mod distance;
mod index;
pub mod sort;
mod r#trait;
mod traversal;
pub mod util;

pub use builder::{RTreeBuilder, DEFAULT_RTREE_NODE_SIZE};
pub use index::{RTree, RTreeMetadata, RTreeRef};
pub use r#trait::{NeighborsOptions, RTreeIndex, SimpleDistanceMetric};
pub use traversal::Node;
