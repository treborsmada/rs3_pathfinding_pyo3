pub mod map_section;
pub mod pathfinding;
pub mod preprocessing;
pub mod state;
pub mod util;

use pyo3::prelude::*;
use crate::{map_section::MapSection,
            state::State};
use std::cmp::{min, max};

#[pyfunction]
#[pyo3(signature = (start, end, floor, setup=false))]
fn a_star(start: (u16, u16), end: (u16, u16), floor: usize, setup: bool) -> PyResult<(Vec<State>, usize)> {
    if setup { preprocessing::setup(false); }
    let start_state = State{
        pos_x: start.0,
        pos_y: start.1,
        direction: 0,
        scd: 0,
        sscd: 0,
        ecd: 0,
        secd: 0,
        bdcd: 0,
    };
    let radius = 150;
    let section = MapSection::create_map_section(
        (min(start.0 as usize, end.0 as usize)).saturating_sub(radius),
        max(start.0 as usize, end.0 as usize) + radius,
        (min(start.1 as usize, end.1 as usize)).saturating_sub(radius),
        max(start.1 as usize, end.1 as usize) + radius,
        floor,
    );
    let heuristic = pathfinding::Heuristic::new();
    pathfinding::a_star_end_buffer(start_state, end, &section, heuristic)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
            format!("No path found from {:?} to {:?} on floor {}", start, end, floor)
        ))
}

#[pyfunction]
#[pyo3(signature = (reset=false))]
fn setup(reset: bool) {
    preprocessing::setup(reset);
}

#[pymodule]
fn rs3_pathfinding(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(a_star, m)?)?;
    m.add_function(wrap_pyfunction!(setup, m)?)?;
    Ok(())
}
