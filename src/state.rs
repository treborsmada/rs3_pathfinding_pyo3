use std::cmp::max;
use crate::map_section::MapSection;

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct State {
    pub pos_x: u16,
    pub pos_y: u16,
    pub direction: u8,
    pub secd: u8,
    pub scd: u8,
    pub ecd: u8,
    pub bdcd: u8,
}

impl State {
    pub fn update(&self) -> State {
        let pos_x = self.pos_x;
        let pos_y = self.pos_y;
        let direction = self.direction;
        let secd = max(self.secd, 1) - 1;
        let scd = max(self.scd, 1) - 1;
        let ecd= max(self.ecd, 1) - 1;
        let bdcd = max(self.bdcd, 1) - 1;
        State {
            pos_x,
            pos_y,
            direction,
            secd,
            scd,
            ecd,
            bdcd,
        }
    }

    pub fn r#move(&self, x: u16, y: u16, direction: u8) -> State {
        State {
            pos_x: x,
            pos_y: y,
            direction,
            secd: self.secd,
            scd: self.scd,
            ecd: self.ecd,
            bdcd: self.bdcd,
        }
    }

    pub fn surge(&self, section: &MapSection) -> State{
        let (new_x, new_y) = section.surge_range(self.pos_x, self.pos_y, self.direction);
        if self.secd == 0 {
            State {
                pos_x: new_x as u16,
                pos_y: new_y as u16,
                direction: self.direction,
                secd: 17,
                scd: max(2, self.scd),
                ecd: 17,
                bdcd: self.bdcd,
            }
        } else if self.scd == 0 {
            State {
                pos_x: new_x as u16,
                pos_y: new_y as u16,
                direction: self.direction,
                secd: max(2, self.secd),
                scd: 17,
                ecd: max(2, self.ecd),
                bdcd: self.bdcd,
            }
        } else {
            panic!()
        }
    }

    pub fn escape(&self, section: &MapSection) -> State {
        let (new_x, new_y) = section.escape_range(self.pos_x, self.pos_y, self.direction);
        if self.secd == 0 {
            State {
                pos_x: new_x as u16,
                pos_y: new_y as u16,
                direction: self.direction,
                secd: 17,
                scd: 17,
                ecd: max(2, self.ecd),
                bdcd: self.bdcd,
            }
        } else if self.ecd == 0 {
            State {
                pos_x: new_x as u16,
                pos_y: new_y as u16,
                direction: self.direction,
                secd: max(2, self.secd),
                scd: max(2, self.scd),
                ecd: 17,
                bdcd: self.bdcd,
            }
        } else {
            panic!()
        }
    }

    pub fn bd(&self, x:u16, y: u16, direction: u8) -> State{
        assert_eq!(self.bdcd, 0);
        State {
            pos_x: x,
            pos_y: y,
            direction,
            secd: self.secd,
            scd: self.scd,
            ecd: self.ecd,
            bdcd: 17,
        }
    }

    pub fn can_bd(&self) -> bool{
        self.bdcd == 0
    }

    pub fn can_surge(&self) -> bool {
        self.secd == 0 || self.scd == 0
    }

    pub fn can_escape(&self) -> bool {
        self.secd == 0 || self.ecd == 0
    }

    pub fn at_goal(&self, end: &(u16, u16)) -> bool{
        end.0 - 1 <= self.pos_x && self.pos_x <= end.0 + 1  && end.1 - 1 <= self.pos_y && self.pos_y <= end.1 + 1
    }
}