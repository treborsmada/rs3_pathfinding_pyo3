use std::cmp::max;
use pyo3::prelude::*;
use crate::map_section::MapSection;

/// Represents the full A* search state: player position, facing direction, and ability cooldowns.
///
/// Cooldown fields count down from 17 each tick (via `saturating_sub`).
/// - `scd`/`sscd`: Surge primary and secondary charge cooldowns.
/// - `ecd`/`secd`: Escape primary and secondary charge cooldowns.
/// - `bdcd`: Bladed Dive cooldown.
///
/// Surge is available when `scd == 0` OR `sscd == 0`; Escape likewise; BD when `bdcd == 0`.
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct State {
    pub pos_x: u16,
    pub pos_y: u16,
    pub direction: u8,
    pub scd: u8,
    pub sscd: u8,
    pub ecd: u8,
    pub secd: u8,
    pub bdcd: u8,
}

impl IntoPy<PyObject> for State {
    fn into_py(self, py: Python<'_>) -> PyObject {
        (self.pos_x, self.pos_y, self.direction, self.scd, self.sscd, self.ecd, self.secd, self.bdcd).into_py(py)
    }
}

impl State {
    /// Advances one game tick: decrements all cooldowns by 1 (floor 0) and keeps position/direction.
    pub fn update(&self) -> State {
        let pos_x = self.pos_x;
        let pos_y = self.pos_y;
        let direction = self.direction;
        let scd = self.scd.saturating_sub(1);
        let sscd = self.sscd.saturating_sub(1);
        let ecd = self.ecd.saturating_sub(1);
        let secd = self.secd.saturating_sub(1);
        let bdcd = self.bdcd.saturating_sub(1);
        State {
            pos_x,
            pos_y,
            direction,
            scd,
            sscd,
            ecd,
            secd,
            bdcd,
        }
    }

    /// Returns a new state at (x, y) facing `direction`, cooldowns unchanged (caller applies `update`).
    pub fn r#move(&self, x: u16, y: u16, direction: u8) -> State {
        State {
            pos_x: x,
            pos_y: y,
            direction,
            scd: self.scd,
            sscd: self.sscd,
            ecd: self.ecd,
            secd: self.secd,
            bdcd: self.bdcd,
        }
    }

    /// Fires Surge in the current direction. Consumes whichever charge slot is available (`scd`
    /// first, then `sscd`) by resetting it to 17, and bumps all cross-triggered cooldowns to
    /// at least 2. Returns `None` if the destination is outside the loaded map section.
    pub fn surge(&self, section: &MapSection) -> Option<State> {
        let (new_x, new_y) = section.surge_range(self.pos_x, self.pos_y, self.direction)?;
        Some(if self.scd == 0 {
            State {
                pos_x: new_x as u16, pos_y: new_y as u16, direction: self.direction,
                scd: 17,
                sscd: max(2, self.sscd),
                ecd: max(2, self.ecd),
                secd: max(2, self.secd),
                bdcd: self.bdcd,
            }
        } else if self.sscd == 0 {
            State {
                pos_x: new_x as u16, pos_y: new_y as u16, direction: self.direction,
                scd: max(2, self.scd),
                sscd: 17,
                ecd: max(2, self.ecd),
                secd: max(2, self.secd),
                bdcd: self.bdcd,
            }
        } else { panic!() })
    }

    /// Fires Escape in the reverse of the current direction. Consumes `ecd` first, then `secd`,
    /// and cross-triggers the remaining cooldowns to at least 2. Returns `None` if out of bounds.
    pub fn escape(&self, section: &MapSection) -> Option<State> {
        let (new_x, new_y) = section.escape_range(self.pos_x, self.pos_y, self.direction)?;
        Some(if self.ecd == 0 {
            State {
                pos_x: new_x as u16, pos_y: new_y as u16, direction: self.direction,
                scd: max(2, self.scd),
                sscd: max(2, self.sscd),
                ecd: 17,
                secd: max(2, self.secd),
                bdcd: self.bdcd,
            }
        } else if self.secd == 0 {
            State {
                pos_x: new_x as u16, pos_y: new_y as u16, direction: self.direction,
                scd: max(2, self.scd),
                sscd: max(2, self.sscd),
                ecd: max(2, self.ecd),
                secd: 17,
                bdcd: self.bdcd,
            }
        } else { panic!() })
    }

    /// Fires Bladed Dive to (x, y). No cross-trigger — only `bdcd` is reset to 17.
    pub fn bd(&self, x:u16, y: u16, direction: u8) -> State{
        assert_eq!(self.bdcd, 0);
        State {
            pos_x: x,
            pos_y: y,
            direction,
            scd: self.scd,
            sscd: self.sscd,
            ecd: self.ecd,
            secd: self.secd,
            bdcd: 17,
        }
    }

    /// Teleports to (x, y) and advances all cooldowns by `cost` ticks (flooring at 0).
    /// Direction is preserved; this models fixed-cost teleport actions.
    pub fn teleport(&self, x: u16, y: u16, cost: u8) -> State {
        State {
            pos_x: x,
            pos_y: y,
            direction: self.direction,
            scd: self.scd.saturating_sub(cost),
            sscd: self.sscd.saturating_sub(cost),
            ecd: self.ecd.saturating_sub(cost),
            secd: self.secd.saturating_sub(cost),
            bdcd: self.bdcd.saturating_sub(cost),
        }
    }

    pub fn can_bd(&self) -> bool{
        self.bdcd == 0
    }

    pub fn can_surge(&self) -> bool {
        self.scd == 0 || self.sscd == 0
    }

    pub fn can_escape(&self) -> bool {
        self.ecd == 0 || self.secd == 0
    }

    /// True when the player is within the 3×3 tile region centred on `end` (i.e. adjacent or on it).
    pub fn at_goal(&self, end: &(u16, u16)) -> bool{
        end.0.saturating_sub(1) <= self.pos_x && self.pos_x <= end.0 + 1
            && end.1.saturating_sub(1) <= self.pos_y && self.pos_y <= end.1 + 1
    }
}
