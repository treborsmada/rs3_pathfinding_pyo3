use pathfinding::prelude::astar;
use ndarray::Array5;
use ndarray_npy::read_npy;
use std::cmp::max;
use crate::{map_section::MapSection,
            state::State};

pub fn a_star_end_buffer(start: State, end: (u16, u16), map: &MapSection, heuristic: Heuristic) -> (Vec<State>, usize) {
    let result = astar(&start, |s| successors(s, map), |s| heuristic.h(s, end), |s| s.at_goal(&end)).unwrap();
    result
}

fn successors(state: &State, map: &MapSection) -> Vec<(State, usize)> {
    let mut adjacent = Vec::with_capacity(500);
    for pos in map.walk_range(state.pos_x, state.pos_y) {
        adjacent.push((state.r#move(pos.0, pos.1, pos.2).update(), 1));
    }
    if state.can_bd() {
        for pos in map.bd_range(state.pos_x, state.pos_y) {
            adjacent.push((state.bd(pos.0, pos.1, pos.2), 0));
        }
    }
    if state.can_surge() {
        adjacent.push((state.surge(&map), 0));
    }
    if state.can_escape() {
        adjacent.push((state.escape(&map), 0));
    }
    adjacent.push((state.update(), 1));
    adjacent
}

pub struct Heuristic {
    data: Array5<u64>,
}

impl Heuristic {
    pub fn new() -> Heuristic{
        let data: Array5<u64> = read_npy("HeuristicData/l_infinity_cds.npy").unwrap();
        Heuristic {
            data
        }
    }
    pub fn h(&self, state: &State, end: (u16, u16)) -> usize{
        let mut distance = max(state.pos_x.abs_diff(end.0), state.pos_y.abs_diff(end.1)) as usize;
        if distance > 0 {
        distance = distance - 1;
        }
        self.data[[distance, state.secd as usize, state.scd as usize, state.ecd as usize, state.bdcd as usize]] as usize
    }
}
