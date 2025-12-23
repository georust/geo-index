mod builder;
mod input;
mod metadata;
mod query;
mod range;
mod within;

use pyo3::intern;
use pyo3::prelude::*;

// https://github.com/PyO3/pyo3/issues/1517#issuecomment-808664021
// https://github.com/PyO3/pyo3/issues/759#issuecomment-977835119
pub fn register_kdtree_module(
    py: Python<'_>,
    parent_module: &Bound<'_, PyModule>,
    parent_module_str: &str,
) -> PyResult<()> {
    let full_module_string = format!("{}.kdtree", parent_module_str);

    let child_module = PyModule::new(parent_module.py(), "kdtree")?;

    child_module.add_class::<builder::PyKDTree>()?;
    child_module.add_class::<builder::PyKDTreeBuilder>()?;
    child_module.add_class::<metadata::PyKDTreeMetadata>()?;
    child_module.add_wrapped(wrap_pyfunction!(range::range))?;
    child_module.add_wrapped(wrap_pyfunction!(within::within))?;
    child_module.add_wrapped(wrap_pyfunction!(query::query))?;

    parent_module.add_submodule(&child_module)?;

    py.import(intern!(py, "sys"))?
        .getattr(intern!(py, "modules"))?
        .set_item(full_module_string.as_str(), &child_module)?;

    // needs to be set *after* `add_submodule()`
    child_module.setattr("__name__", full_module_string)?;

    Ok(())
}
