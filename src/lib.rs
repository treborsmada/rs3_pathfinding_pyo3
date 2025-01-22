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
fn a_star(start: (u16, u16), end: (u16, u16), floor: usize) -> (Vec<State>, usize) {
    preprocessing::setup(false);
    let start_sate = State{
        pos_x: start.0,
        pos_y: start.1,
        direction: 0,
        secd: 0,
        scd: 0,
        ecd: 0,
        bdcd: 0,
    };
    let radius = 150;
    let section = MapSection::create_map_section(min(start.0 as usize, end.0 as usize) - radius,
                                                            max(start.0 as usize, end.0 as usize) + radius,
                                                            min(start.1 as usize, end.1 as usize) - radius,
                                                            max(start.1 as usize, end.1 as usize) + radius, floor);
    let heuristic = pathfinding::Heuristic::new();
    let (path, ticks) = pathfinding::a_star_end_buffer(start_sate, end, &section, heuristic);
    (path, ticks)
}

#[pymodule]
fn rs3_pathfinding(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(a_star, m)?)?;
    Ok(())
}
