//! Error/validation helpers shared by the bindings: clean `ValueError`s for bad
//! arguments and `from_py_with` count hooks (so a negative int yields our message,
//! not pyo3's raw `OverflowError`), plus finite-ness guards for numeric inputs.

use numpy::ndarray::ArrayView2;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;

pub(crate) fn io_err(e: std::io::Error) -> PyErr {
    PyIOError::new_err(e.to_string())
}

/// Validate a count argument given as a signed Python int, so negatives raise a
/// clean `ValueError` instead of PyO3's raw "can't convert negative int to
/// unsigned" `OverflowError`. Accepting `i64` at the signature keeps the boundary
/// from rejecting negatives before our own message can run.
pub(crate) fn require_count(value: i64, min: i64, name: &str) -> PyResult<usize> {
    if value < min {
        return Err(PyValueError::new_err(format!(
            "{name} must be >= {min}, got {value}"
        )));
    }
    Ok(value as usize)
}

/// Guard that every element of a 2-D feature/covariate/embedding matrix is
/// finite. Called right after `parse_features` returns and the row-count check
/// passes, so the error names the parameter exactly as the user passed it.
pub(crate) fn check_all_finite_2d(name: &str, rows: &[Vec<f64>]) -> PyResult<()> {
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if !v.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "{name} contains non-finite values (NaN or inf) at row {i}, col {j}"
                )));
            }
        }
    }
    Ok(())
}

/// Guard that every element of a 2-D ndarray view is finite. Used when data
/// enters via `PyReadonlyArray2` rather than through `parse_features`.
pub(crate) fn check_all_finite_arr2(name: &str, arr: &ArrayView2<f64>) -> PyResult<()> {
    for ((i, j), &v) in arr.indexed_iter() {
        if !v.is_finite() {
            return Err(PyValueError::new_err(format!(
                "{name} contains non-finite values (NaN or inf) at row {i}, col {j}"
            )));
        }
    }
    Ok(())
}

/// Guard that every element of a 1-D numeric sequence (e.g. timestamps) is
/// finite.
pub(crate) fn check_all_finite_1d(name: &str, vals: &[f64]) -> PyResult<()> {
    for (i, &v) in vals.iter().enumerate() {
        if !v.is_finite() {
            return Err(PyValueError::new_err(format!(
                "{name} contains non-finite values (NaN or inf) at index {i}"
            )));
        }
    }
    Ok(())
}

// `from_py_with` hooks for count constructor arguments. They take the int as a
// signed `i64` so a negative value yields a clean `ValueError` here rather than
// PyO3's raw `OverflowError`. Per-model minimums above 1 (e.g. CTM/STM need >= 2)
// stay enforced by the existing guards inside each constructor body.
pub(crate) fn py_num_topics(ob: &Bound<'_, PyAny>) -> PyResult<usize> {
    require_count(ob.extract()?, 1, "num_topics")
}
pub(crate) fn py_num_pseudo(ob: &Bound<'_, PyAny>) -> PyResult<usize> {
    require_count(ob.extract()?, 1, "num_pseudo")
}
pub(crate) fn py_num_super(ob: &Bound<'_, PyAny>) -> PyResult<usize> {
    require_count(ob.extract()?, 1, "num_super")
}
pub(crate) fn py_num_sub(ob: &Bound<'_, PyAny>) -> PyResult<usize> {
    require_count(ob.extract()?, 1, "num_sub")
}
pub(crate) fn py_depth(ob: &Bound<'_, PyAny>) -> PyResult<usize> {
    require_count(ob.extract()?, 1, "depth")
}
pub(crate) fn py_num_topics_opt(ob: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
    if ob.is_none() {
        return Ok(None);
    }
    Ok(Some(require_count(ob.extract()?, 1, "num_topics")?))
}
