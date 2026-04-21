use ndarray::{Array3, Axis, concatenate};
use std::cmp;
use crate::util::read_npy_compressed;

/// A loaded rectangular slice of the map, containing precomputed move/BD/SE neighbour lists.
///
/// Indices into the flat neighbour vecs use `xi * y_len + yi` where `xi = x - x_start`.
/// `se_data[[xi, yi, dir]]`: low nibble = Surge offset (tiles), high nibble = Escape offset.
#[derive(Debug)]
pub struct MapSection {
    #[allow(dead_code)]
    floor: usize,
    x_start: usize,
    x_len: usize,
    y_start: usize,
    y_len: usize,
    se_data: Array3<u8>,
    walk_vec: Vec<Vec<(u16, u16, u8)>>,
    bd_vec: Vec<Vec<(u16, u16, u8)>>,
}

impl MapSection {

    /// Returns the Surge landing tile for a player at (x, y) facing `direction`.
    /// The offset (1–10 tiles) is stored in the low nibble of `se_data`.
    pub fn surge_range(&self, x: u16, y: u16, direction: u8) -> Option<(usize, usize)> {
        let (x, y, direction) = (x as usize, y as usize, direction as usize);
        let xi = x.wrapping_sub(self.x_start);
        let yi = y.wrapping_sub(self.y_start);
        if xi >= self.x_len || yi >= self.y_len { return None; }
        let offset = (self.se_data[[xi, yi, direction]] & 15) as usize;
        Some(match direction {
            0 => (x, y + offset),
            1 => (x + offset, y + offset),
            2 => (x + offset, y),
            3 => (x + offset, y - offset),
            4 => (x, y - offset),
            5 => (x - offset, y - offset),
            6 => (x - offset, y),
            7 => (x - offset, y + offset),
            _ => panic!()
        })
    }

    /// Returns the Escape landing tile. Escape travels opposite to `direction` (up to 7 tiles).
    /// The offset is stored in the high nibble of `se_data`.
    pub fn escape_range(&self, x: u16, y: u16, direction: u8) -> Option<(usize, usize)> {
        let (x, y, direction) = (x as usize, y as usize, direction as usize);
        let xi = x.wrapping_sub(self.x_start);
        let yi = y.wrapping_sub(self.y_start);
        if xi >= self.x_len || yi >= self.y_len { return None; }
        let offset = (self.se_data[[xi, yi, direction]] >> 4) as usize;
        Some(match direction {
            0 => (x, y - offset),
            1 => (x - offset, y - offset),
            2 => (x - offset, y),
            3 => (x - offset, y + offset),
            4 => (x, y + offset),
            5 => (x + offset, y + offset),
            6 => (x + offset, y),
            7 => (x + offset, y - offset),
            _ => panic!()
        })
    }

    pub fn create_map_section(x_start: usize, x_end: usize, y_start: usize, y_end: usize, floor: usize) -> MapSection {
        let x_len = x_end - x_start + 1;
        let y_len = y_end - y_start + 1;
        let bd_data = build_bd_array(x_start, x_end, y_start, y_end, floor);
        let se_data = build_se_array(x_start, x_end, y_start, y_end, floor);
        let walk_data = build_walk_array(x_start, x_end, y_start, y_end, floor);
        let walk_vec = build_walk_vec(x_start, x_end, y_start, y_end, y_len, &walk_data);
        let bd_vec = build_bd_vec(x_start, x_end, y_start, y_end, y_len, &bd_data);
        MapSection {
            floor,
            x_start,
            x_len,
            y_start,
            y_len,
            se_data,
            walk_vec,
            bd_vec,
        }
    }

    pub fn walk_range(&self, x: u16, y: u16) -> &[(u16, u16, u8)] {
        let xi = (x as usize).wrapping_sub(self.x_start);
        let yi = (y as usize).wrapping_sub(self.y_start);
        if xi < self.x_len && yi < self.y_len {
            &self.walk_vec[xi * self.y_len + yi]
        } else {
            &[]
        }
    }

    pub fn bd_range(&self, x: u16, y: u16) -> &[(u16, u16, u8)] {
        let xi = (x as usize).wrapping_sub(self.x_start);
        let yi = (y as usize).wrapping_sub(self.y_start);
        if xi < self.x_len && yi < self.y_len {
            &self.bd_vec[xi * self.y_len + yi]
        } else {
            &[]
        }
    }
}

/// Decodes the BD bitset array into per-tile neighbour lists.
///
/// Each tile's BD reach is stored as a 441-bit bitset (21×21 window) packed into 7 × u64.
/// Bit index `k = dy*21 + dx` (with the origin at the centre, offset by 10) is set when the
/// tile at (x-10+dx, y-10+dy) is a reachable BD destination.
///
/// For each reachable destination the facing direction after BD is inferred from the vector
/// (x_diff, y_diff) using integer comparisons that approximate whether the movement is more
/// cardinal or diagonal.
fn build_bd_vec(x_start: usize, x_end: usize, y_start: usize, y_end: usize, y_len: usize, arr: &Array3<u64>) -> Vec<Vec<(u16, u16, u8)>> {
    let x_len = x_end - x_start + 1;
    let mut bd_vec = vec![Vec::new(); x_len * y_len];
    for x in x_start..=x_end {
        for y in y_start..=y_end {
            let mut tiles = Vec::new();
            for i in 0..7 {
                let bd_data = arr[[x - x_start, y - y_start, i]];
                for j in 0..64 {
                    if (bd_data >> j) & 1 == 1 {
                        // Recover (u, v) from the flat bit index k = j + 64*i.
                        let u = x - 10 + (j+64*i) % 21;
                        let v = y - 10 + (j+64*i) / 21;
                        let x_diff = (u as isize) - (x as isize);
                        let y_diff = (v as isize) - (y as isize);
                        // Infer facing direction from the BD displacement vector.
                        // Integer comparisons approximate the 22.5° sector boundaries:
                        //   if |dx|/|dy| > 7.5 → nearly horizontal (E or W)
                        //   if |dy|/|dx| > 7.5 → nearly vertical   (N or S)
                        //   otherwise → diagonal quadrant (NE, SE, SW, NW)
                        let mut direction: u8 = 0;
                        if x_diff == 0 {
                            if y_diff > 0 {
                                direction = 0;
                            } else {
                                direction = 4;
                            }
                        } else if y_diff == 0 {
                            if x_diff > 0 {
                                direction = 2;
                            } else {
                                direction = 6;
                            }
                        } else if (14 * x_diff.abs() + 7) / (2 * y_diff.abs() + 1) > 15 {
                            if x_diff > 0 {
                                direction = 2;
                            } else {
                                direction = 6;
                            }
                        } else if (14 * y_diff.abs() + 7) / (2 * x_diff.abs() + 1) > 15 {
                            if y_diff > 0 {
                                direction = 0;
                            } else {
                                direction = 4;
                            }
                        } else if x_diff > 0 {
                            if y_diff > 0 {
                                direction = 1;
                            } else {
                                direction = 3;
                            }
                        } else if x_diff < 0 {
                            if y_diff > 0 {
                                direction = 7;
                            } else {
                                direction = 5;
                            }
                        }
                        tiles.push((u as u16, v as u16, direction));
                    }
                }
            }
            bd_vec[(x - x_start) * y_len + (y - y_start)] = tiles;
        }
    }
    bd_vec
}

/// Decodes the walk-reachability array into per-tile neighbour lists.
///
/// Each tile stores reachable walk destinations within a 5×5 window packed into 2 × u64.
/// The 25 positions are laid out as 4-bit nibbles (position k at bits 4k..4k+3).
/// A nibble value 0–7 is the direction to face on arrival; value 15 (0xF) means not reachable.
fn build_walk_vec(x_start: usize, x_end: usize, y_start: usize, y_end: usize, y_len: usize, arr: &Array3<u64>) -> Vec<Vec<(u16, u16, u8)>> {
    let x_len = x_end - x_start + 1;
    let mut walk_vec = vec![Vec::new(); x_len * y_len];
    for x in x_start..=x_end {
        for y in y_start..=y_end {
            let mut tiles = Vec::new();
            for i in 0..2 {
                let walk_data = arr[[x-x_start, y - y_start, i]];
                for j in 0..16 {
                    let direction = (walk_data >> (j * 4)) & 15;
                    if direction < 8 {
                        // Flat index k = j + 16*i maps to (dx, dy) in [-2, 2]² via k%5, k/5.
                        let u = x - 2 + (j + 16 * i) % 5;
                        let v = y - 2 + (j + 16 * i) / 5;
                        tiles.push((u as u16, v as u16, direction as u8))
                    }
                }
            }
            walk_vec[(x - x_start) * y_len + (y - y_start)] = tiles;
        }
    }
    walk_vec
}

/// Loads the BD `.npy` chunks covering [x_start, x_end] × [y_start, y_end] and stitches them
/// into a single contiguous array by concatenating along x then y.
fn build_bd_array(x_start: usize, x_end: usize, y_start: usize, y_end: usize, floor: usize) -> Array3<u64> {
    let chunk_size =  1280;
    let chunk_x = (x_start/chunk_size, x_end/chunk_size);
    let chunk_y = (y_start/chunk_size, y_end/chunk_size);
    let mut rows = Vec::new();
    for j in chunk_y.0..=chunk_y.1 {
        let mut row  = Vec::new();
        for i in chunk_x.0..=chunk_x.1 {
            let path = format!("MapData/BD/bd-{i}-{j}-{floor}.npy");
            let arr: Array3<u64> = read_npy_compressed(path);
            let x_1 = cmp::max(x_start % chunk_size,(i - chunk_x.0) * chunk_size) - (i - chunk_x.0) * chunk_size;
            let x_2 = cmp::min(x_end - x_start + (x_start % chunk_size) + 1, chunk_size);
            let y_1 = cmp::max(y_start % chunk_size, (j - chunk_y.0) * chunk_size) - (j - chunk_y.0) * chunk_size;
            let y_2 = cmp::min(y_end - y_start + (y_start % chunk_size) - (j - chunk_y.0) * chunk_size + 1, chunk_size);
            let arr = arr.slice(ndarray::s![x_1..x_2, y_1..y_2, ..]).to_owned();
            row.push(arr);
        }
        let views: Vec<_> = row.iter().map(|arr| arr.view()).collect();
        rows.push(concatenate(Axis(0), &views[..]).unwrap());
    }
    let views: Vec<_> = rows.iter().map(|arr| arr.view()).collect();
    concatenate(Axis(1), &views[..]).unwrap()
}

/// Loads and stitches the SE (Surge/Escape offset) chunks. Same structure as `build_bd_array`.
fn build_se_array(x_start: usize, x_end: usize, y_start: usize, y_end: usize, floor: usize) -> Array3<u8> {
    let chunk_size =  1280;
    let chunk_x = (x_start/chunk_size, x_end/chunk_size);
    let chunk_y = (y_start/chunk_size, y_end/chunk_size);
    let mut rows = Vec::new();
    for j in chunk_y.0..=chunk_y.1 {
        let mut row  = Vec::new();
        for i in chunk_x.0..=chunk_x.1 {
            let path = format!("MapData/SE/se-{i}-{j}-{floor}.npy");
            let arr: Array3<u8> = read_npy_compressed(path);
            let x_1 = cmp::max(x_start % chunk_size,(i - chunk_x.0) * chunk_size) - (i - chunk_x.0) * chunk_size;
            let x_2 = cmp::min(x_end - x_start + (x_start % chunk_size) + 1, chunk_size);
            let y_1 = cmp::max(y_start % chunk_size, (j - chunk_y.0) * chunk_size) - (j - chunk_y.0) * chunk_size;
            let y_2 = cmp::min(y_end - y_start + (y_start % chunk_size) - (j - chunk_y.0) * chunk_size + 1, chunk_size);
            let arr = arr.slice(ndarray::s![x_1..x_2, y_1..y_2, ..]).to_owned();
            row.push(arr);
        }
        let views: Vec<_> = row.iter().map(|arr| arr.view()).collect();
        rows.push(concatenate(Axis(0), &views[..]).unwrap());
    }
    let views: Vec<_> = rows.iter().map(|arr| arr.view()).collect();
    concatenate(Axis(1), &views[..]).unwrap()
}

/// Loads and stitches the Walk reachability chunks. Same structure as `build_bd_array`.
fn build_walk_array(x_start: usize, x_end: usize, y_start: usize, y_end: usize, floor: usize) -> Array3<u64> {
    let chunk_size =  1280;
    let chunk_x = (x_start/chunk_size, x_end/chunk_size);
    let chunk_y = (y_start/chunk_size, y_end/chunk_size);
    let mut rows = Vec::new();
    for j in chunk_y.0..=chunk_y.1 {
        let mut row  = Vec::new();
        for i in chunk_x.0..=chunk_x.1 {
            let path = format!("MapData/Walk/walk-{i}-{j}-{floor}.npy");
            let arr: Array3<u64> = read_npy_compressed(path);
            let x_1 = cmp::max(x_start % chunk_size,(i - chunk_x.0) * chunk_size) - (i - chunk_x.0) * chunk_size;
            let x_2 = cmp::min(x_end - x_start + (x_start % chunk_size) + 1, chunk_size);
            let y_1 = cmp::max(y_start % chunk_size, (j - chunk_y.0) * chunk_size) - (j - chunk_y.0) * chunk_size;
            let y_2 = cmp::min(y_end - y_start + (y_start % chunk_size) - (j - chunk_y.0) * chunk_size + 1, chunk_size);
            let arr = arr.slice(ndarray::s![x_1..x_2, y_1..y_2, ..]).to_owned();
            row.push(arr);
        }
        let views: Vec<_> = row.iter().map(|arr| arr.view()).collect();
        rows.push(concatenate(Axis(0), &views[..]).unwrap());
    }
    let views: Vec<_> = rows.iter().map(|arr| arr.view()).collect();
    concatenate(Axis(1), &views[..]).unwrap()
}