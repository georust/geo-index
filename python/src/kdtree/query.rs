use std::sync::Arc;

use arrow_array::UInt32Array;
use arrow_buffer::ScalarBuffer;
use geo_index::kdtree::KDTreeIndex;
use pyo3::{prelude::*, types::PyFloat};
use pyo3_arrow::PyArray;

use crate::kdtree::input::PyKDTreeRef;

#[pyfunction]
pub fn query(
    py: Python,
    index: PyKDTreeRef,
    qx: Bound<PyAny>,
    qy: Bound<PyAny>,
) -> PyResult<(Py<PyFloat>, Py<PyAny>)> {
    match index {
        PyKDTreeRef::Float32(tree) => {
            let (d, results) = tree.query(qx.extract()?, qy.extract()?);
            let results = UInt32Array::new(ScalarBuffer::from(results), None);
            Ok((
                PyFloat::new(py, d as f64).unbind(),
                PyArray::from_array_ref(Arc::new(results))
                    .to_arro3(py)?
                    .unbind(),
            ))
        }
        PyKDTreeRef::Float64(tree) => {
            let (d, results) = tree.query(qx.extract()?, qy.extract()?);
            let results = UInt32Array::new(ScalarBuffer::from(results), None);
            Ok((
                PyFloat::new(py, d).unbind(),
                PyArray::from_array_ref(Arc::new(results))
                    .to_arro3(py)?
                    .unbind(),
            ))
        }
    }
}
