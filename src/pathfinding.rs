use pathfinding::prelude::astar;
use ndarray::Array6;
use crate::util::read_npy_compressed;
use std::{cmp::max, sync::OnceLock};
use crate::{map_section::MapSection,
            state::State};

pub fn a_star_end_buffer(start: State, end: (u16, u16), map: &MapSection, heuristic: Heuristic) -> Option<(Vec<State>, usize)> {
    let teleports = heuristic.teleports;
    astar(&start, |s| successors(s, map, teleports), |s| heuristic.h(s, end), |s| s.at_goal(&end))
}

pub fn successors(state: &State, map: &MapSection, teleports: &[(u16, u16, u8)]) -> Vec<(State, usize)> {
    let mut adjacent = Vec::with_capacity(64 + teleports.len());
    for pos in map.walk_range(state.pos_x, state.pos_y) {
        adjacent.push((state.r#move(pos.0, pos.1, pos.2).update(), 1));
    }
    if state.can_bd() {
        for pos in map.bd_range(state.pos_x, state.pos_y) {
            adjacent.push((state.bd(pos.0, pos.1, pos.2), 0));
        }
    }
    if state.can_surge() {
        if let Some(s) = state.surge(&map) { adjacent.push((s, 0)); }
    }
    if state.can_escape() {
        if let Some(s) = state.escape(&map) { adjacent.push((s, 0)); }
    }
    adjacent.push((state.update(), 1));
    for &(x, y, cost) in teleports {
        adjacent.push((state.teleport(x, y, cost), cost as usize));
    }
    adjacent
}

static HEURISTIC_DATA: OnceLock<Array6<u64>> = OnceLock::new();

pub struct Heuristic<'a> {
    data: &'static Array6<u64>,
    pub teleports: &'a [(u16, u16, u8)],
}

impl<'a> Heuristic<'a> {
    pub fn new(teleports: &'a [(u16, u16, u8)]) -> Heuristic<'a> {
        let data = HEURISTIC_DATA.get_or_init(|| {
            read_npy_compressed("HeuristicData/l_infinity_cds.npy")
        });
        Heuristic { data, teleports }
    }

    pub fn h(&self, state: &State, end: (u16, u16)) -> usize {
        let max_d = self.data.len_of(ndarray::Axis(0)) - 1;
        let h_direct = {
            let mut distance = max(state.pos_x.abs_diff(end.0), state.pos_y.abs_diff(end.1)) as usize;
            if distance > 0 { distance -= 1; }
            self.data[[distance.min(max_d), state.scd as usize, state.sscd as usize,
                       state.ecd as usize, state.secd as usize, state.bdcd as usize]] as usize
        };
        self.teleports.iter().fold(h_direct, |acc, &(tx, ty, cost)| {
            let mut dest_dist = max(tx.abs_diff(end.0), ty.abs_diff(end.1)) as usize;
            if dest_dist > 0 { dest_dist -= 1; }
            let h_dest = self.data[[dest_dist.min(max_d), 0, 0, 0, 0, 0]] as usize;
            acc.min(cost as usize + h_dest)
        })
    }
}
