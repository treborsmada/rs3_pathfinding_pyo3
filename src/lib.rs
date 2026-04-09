pub mod map_section;
pub mod pathfinding;
pub mod preprocessing;
pub mod state;
pub mod util;

use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use pyo3::prelude::*;
use crate::{map_section::MapSection,
            state::State};
use std::cmp::{min, max};
use std::sync::OnceLock;

static SECTION_CACHE: OnceLock<MapSection> = OnceLock::new();

#[pyfunction]
#[pyo3(signature = (start, end, floor, setup=false, teleports=None))]
fn a_star(start: (u16, u16), end: (u16, u16), floor: usize, setup: bool, teleports: Option<Vec<(u16, u16, u8)>>) -> PyResult<(Vec<State>, usize)> {
    if setup { preprocessing::setup(false); }
    let teleports = teleports.unwrap_or_default();
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
    let mut x_min = min(start.0 as usize, end.0 as usize);
    let mut x_max = max(start.0 as usize, end.0 as usize);
    let mut y_min = min(start.1 as usize, end.1 as usize);
    let mut y_max = max(start.1 as usize, end.1 as usize);
    for &(tx, ty, _) in &teleports {
        x_min = min(x_min, tx as usize);
        x_max = max(x_max, tx as usize);
        y_min = min(y_min, ty as usize);
        y_max = max(y_max, ty as usize);
    }
    let section = MapSection::create_map_section(
        x_min.saturating_sub(radius),
        x_max + radius,
        y_min.saturating_sub(radius),
        y_max + radius,
        floor,
    );
    let heuristic = pathfinding::Heuristic::new(&teleports);
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

#[pyfunction]
fn preload_section(x_min: usize, x_max: usize, y_min: usize, y_max: usize, floor: usize) {
    SECTION_CACHE.get_or_init(|| {
        MapSection::create_map_section(x_min, x_max, y_min, y_max, floor)
    });
}

#[pyfunction]
#[pyo3(signature = (start, end, direction=0u8, scd=0u8, sscd=0u8, ecd=0u8, secd=0u8, bdcd=0u8, teleports=None))]
fn a_star_preloaded(
    start: (u16, u16), end: (u16, u16),
    direction: u8, scd: u8, sscd: u8, ecd: u8, secd: u8, bdcd: u8,
    teleports: Option<Vec<(u16, u16, u8)>>,
) -> PyResult<(Vec<State>, usize)> {
    let section = SECTION_CACHE.get()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
            "Call preload_section before a_star_preloaded"
        ))?;
    let teleports = teleports.unwrap_or_default();
    let start_state = State { pos_x: start.0, pos_y: start.1, direction, scd, sscd, ecd, secd, bdcd };
    let heuristic = pathfinding::Heuristic::new(&teleports);
    pathfinding::a_star_end_buffer(start_state, end, section, heuristic)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
            format!("No path found from {:?} to {:?}", start, end)
        ))
}

#[pyfunction]
#[pyo3(signature = (x, y, direction=0u8, scd=0u8, sscd=0u8, ecd=0u8, secd=0u8, bdcd=0u8, teleports=None))]
fn get_successors_preloaded(
    x: u16, y: u16, direction: u8,
    scd: u8, sscd: u8, ecd: u8, secd: u8, bdcd: u8,
    teleports: Option<Vec<(u16, u16, u8)>>,
) -> PyResult<Vec<(State, usize)>> {
    let section = SECTION_CACHE.get()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
            "Call preload_section before get_successors_preloaded"
        ))?;
    let teleports = teleports.unwrap_or_default();
    let state = State { pos_x: x, pos_y: y, direction, scd, sscd, ecd, secd, bdcd };
    Ok(pathfinding::successors(&state, section, &teleports))
}

#[pyfunction]
fn heuristic_cost(x: u16, y: u16, tx: u16, ty: u16, scd: u8, sscd: u8, ecd: u8, secd: u8, bdcd: u8) -> usize {
    let h = pathfinding::Heuristic::new(&[]);
    let state = State { pos_x: x, pos_y: y, direction: 0, scd, sscd, ecd, secd, bdcd };
    h.h(&state, (tx, ty))
}

#[pymodule]
fn rs3_pathfinding(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(a_star, m)?)?;
    m.add_function(wrap_pyfunction!(setup, m)?)?;
    m.add_function(wrap_pyfunction!(preload_section, m)?)?;
    m.add_function(wrap_pyfunction!(a_star_preloaded, m)?)?;
    m.add_function(wrap_pyfunction!(get_successors_preloaded, m)?)?;
    m.add_function(wrap_pyfunction!(heuristic_cost, m)?)?;
    Ok(())
}
