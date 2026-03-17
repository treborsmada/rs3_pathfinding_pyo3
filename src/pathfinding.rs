use pathfinding::prelude::astar;
use ndarray::Array6;
use crate::util::read_npy_compressed;
use std::{cmp::max, sync::OnceLock};
use crate::{map_section::MapSection,
            state::State};

pub fn a_star_end_buffer(start: State, end: (u16, u16), map: &MapSection, heuristic: Heuristic) -> Option<(Vec<State>, usize)> {
    astar(&start, |s| successors(s, map), |s| heuristic.h(s, end), |s| s.at_goal(&end))
}

fn successors(state: &State, map: &MapSection) -> Vec<(State, usize)> {
    let mut adjacent = Vec::with_capacity(64);
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

static HEURISTIC_DATA: OnceLock<Array6<u64>> = OnceLock::new();

pub struct Heuristic {
    data: &'static Array6<u64>,
}

impl Heuristic {
    pub fn new() -> Heuristic {
        let data = HEURISTIC_DATA.get_or_init(|| {
            read_npy_compressed("HeuristicData/l_infinity_cds.npy")
        });
        Heuristic { data }
    }

    pub fn h(&self, state: &State, end: (u16, u16)) -> usize {
        let mut distance = max(state.pos_x.abs_diff(end.0), state.pos_y.abs_diff(end.1)) as usize;
        if distance > 0 {
            distance -= 1;
        }
        self.data[[distance, state.scd as usize, state.sscd as usize,
                   state.ecd as usize, state.secd as usize, state.bdcd as usize]] as usize
    }
}
