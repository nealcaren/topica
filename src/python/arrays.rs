//! ndarray <-> serializable-state adapters used by every model's save/load.
//!
//! `*_opt` pack an optional ndarray into a `serde`-friendly shape+data record;
//! `*_back` reconstruct the ndarray, returning a `ValueError` (not a panic) on a
//! shape/length mismatch in a corrupt payload.

use numpy::ndarray::{Array1, Array2, Array3};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

/// Serializable form of an ndarray `Array2` (shape + row-major data).
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct Arr2 {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}
/// Serializable form of an ndarray `Array3` (f64).
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct Arr3 {
    pub d0: usize,
    pub d1: usize,
    pub d2: usize,
    pub data: Vec<f64>,
}
/// Serializable form of an ndarray `Array3<f32>` (used for theta_draws).
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct Arr3f32 {
    pub d0: usize,
    pub d1: usize,
    pub d2: usize,
    pub data: Vec<f32>,
}

pub(crate) fn arr2_opt(a: &Option<Array2<f64>>) -> Option<Arr2> {
    a.as_ref().map(|m| Arr2 {
        rows: m.nrows(),
        cols: m.ncols(),
        data: m.iter().copied().collect(),
    })
}
pub(crate) fn arr2_back(s: Option<Arr2>) -> PyResult<Option<Array2<f64>>> {
    s.map(|a| {
        Array2::from_shape_vec((a.rows, a.cols), a.data)
            .map_err(|e| PyValueError::new_err(format!("corrupt saved 2-D array: {e}")))
    })
    .transpose()
}
pub(crate) fn arr3_opt(a: &Option<Array3<f64>>) -> Option<Arr3> {
    a.as_ref().map(|m| {
        let d = m.dim();
        Arr3 {
            d0: d.0,
            d1: d.1,
            d2: d.2,
            data: m.iter().copied().collect(),
        }
    })
}
pub(crate) fn arr3_back(s: Option<Arr3>) -> PyResult<Option<Array3<f64>>> {
    s.map(|a| {
        Array3::from_shape_vec((a.d0, a.d1, a.d2), a.data)
            .map_err(|e| PyValueError::new_err(format!("corrupt saved 3-D array: {e}")))
    })
    .transpose()
}
pub(crate) fn arr3f32_opt(a: &Option<Array3<f32>>) -> Option<Arr3f32> {
    a.as_ref().map(|m| {
        let d = m.dim();
        Arr3f32 {
            d0: d.0,
            d1: d.1,
            d2: d.2,
            data: m.iter().copied().collect(),
        }
    })
}
pub(crate) fn arr3f32_back(s: Option<Arr3f32>) -> PyResult<Option<Array3<f32>>> {
    s.map(|a| {
        Array3::from_shape_vec((a.d0, a.d1, a.d2), a.data)
            .map_err(|e| PyValueError::new_err(format!("corrupt saved 3-D array: {e}")))
    })
    .transpose()
}
pub(crate) fn arr1_opt(a: &Option<Array1<f64>>) -> Option<Vec<f64>> {
    a.as_ref().map(|m| m.to_vec())
}
pub(crate) fn arr1_back(s: Option<Vec<f64>>) -> Option<Array1<f64>> {
    s.map(Array1::from)
}
