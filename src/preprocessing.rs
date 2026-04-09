use rayon::prelude::*;
use std::{collections::HashMap,
          cmp::max,
          path::Path,
          fs,
          io::Write};
use flate2::{Compression, write::ZlibEncoder};
use zune_inflate::DeflateDecoder;
use ndarray::{Array2, Array3, Array6, ShapeBuilder};
use indicatif::ProgressBar;
use crate::util::{adj_positions, free_direction, read_npy_compressed, write_npy_compressed};

const RS_HEIGHT: usize = 12800;
const RS_LENGTH: usize = 6400;

struct Process {
    movement_data: HashMap<(usize, usize, usize), Array2<u8>>,
    bd_data: HashMap<(usize, usize, usize), Array3<u64>>
}

impl Process {
    fn new() -> Process {
        Process {
            movement_data: HashMap::new(),
            bd_data: HashMap::new()
        }
    }

    fn walk_range(&mut self, x: usize, y: usize, floor: usize) -> Vec<(usize, usize, usize)> {
        let mut tiles = Vec::with_capacity(25);
        let start = self.get_movement_data(x, y, floor);
        let adj = adj_positions(x, y);
        // 5x5 visited grid indexed by (dx+2, dy+2) where dx,dy in [-2,2]
        let mut visited = [[false; 5]; 5];
        visited[2][2] = true;
        // queue holds at most 8 direct neighbors (level 1 only)
        let mut queue = [(0usize, 0usize); 8];
        let mut queue_len = 0;
        for i in 0..8 {
            let j = (2*i + i/4) % 8;
            if free_direction(start, j) {
                let (nx, ny) = adj[j];
                let dx = (nx as isize - x as isize + 2) as usize;
                let dy = (ny as isize - y as isize + 2) as usize;
                if dx < 5 && dy < 5 && !visited[dx][dy] {
                    tiles.push((nx, ny, j));
                    visited[dx][dy] = true;
                    queue[queue_len] = (nx, ny);
                    queue_len += 1;
                }
            }
        }
        for qi in 0..queue_len {
            let current = queue[qi];
            let current_move_data = self.get_movement_data(current.0, current.1, floor);
            let temp_adj = adj_positions(current.0, current.1);
            for i in 0..8 {
                if free_direction(current_move_data, i) {
                    let (nx, ny) = temp_adj[i];
                    let dx = (nx as isize - x as isize + 2) as usize;
                    let dy = (ny as isize - y as isize + 2) as usize;
                    if dx < 5 && dy < 5 && !visited[dx][dy] {
                        tiles.push((nx, ny, i));
                        visited[dx][dy] = true;
                    }
                }
            }
        }
        tiles
    }

    fn bd_range(&mut self, x: usize, y: usize, floor: usize) -> Vec<(usize, usize)> {
        // 21x21 visited grid indexed by (dx+10, dy+10) where dx,dy in [-10,10]
        let mut visited = [[false; 21]; 21];
        self.bd_range_recursion(x, y, x, y, floor, 1, 2, 0, 0, 0, &mut visited);
        self.bd_range_recursion(x, y, x, y, floor, 3, 2, 4, 0, 0, &mut visited);
        self.bd_range_recursion(x, y, x, y, floor, 5, 6, 4, 0, 0, &mut visited);
        self.bd_range_recursion(x, y, x, y, floor, 7, 6, 0, 0, 0, &mut visited);
        let mut tiles = Vec::new();
        for dx in 0..21usize {
            for dy in 0..21usize {
                if visited[dx][dy] {
                    tiles.push((x + dx - 10, y + dy - 10));
                }
            }
        }
        tiles
    }

    fn bd_range_recursion(&mut self, x: usize, y: usize, ox: usize, oy: usize, floor: usize, direction: usize, horizontal: usize, vertical: usize, dist_x: usize, dist_y: usize, visited: &mut [[bool; 21]; 21]) {
        let mut dist_x = dist_x;
        let mut dist_y = dist_y;
        if dist_x > 0 || dist_y > 0 {
            let dx = (x as isize - ox as isize + 10) as usize;
            let dy = (y as isize - oy as isize + 10) as usize;
            visited[dx][dy] = true;
        }
        let curr_move = self.get_movement_data(x, y, floor);
        if dist_x < 10 && dist_y < 10 && free_direction(curr_move, direction) {
            let new_tile = adj_positions(x, y)[direction];
            self.bd_range_recursion(new_tile.0, new_tile.1, ox, oy, floor, direction, horizontal, vertical, dist_x + 1, dist_y + 1, visited);
        }
        else if dist_x < 10 && free_direction(curr_move, horizontal) {
            let new_tile = adj_positions(x, y)[horizontal];
            self.bd_range_recursion(new_tile.0, new_tile.1, ox, oy, floor, direction, horizontal, vertical, dist_x + 1, dist_y, visited);
            dist_x = 10;
        }
        else if dist_y < 10 && free_direction(curr_move, vertical) {
            let new_tile = adj_positions(x, y)[vertical];
            self.bd_range_recursion(new_tile.0, new_tile.1, ox, oy, floor, direction, horizontal, vertical, dist_x, dist_y + 1, visited);
            dist_y = 10;
        }
        let dist = [dist_x, dist_y];
        for (i, dir) in [horizontal, vertical].into_iter().enumerate() {
            let mut d = dist[i];
            let mut curr_tile = (x, y);
            let mut curr_move = self.get_movement_data(x, y, floor);
            while d < 10 && free_direction(curr_move, dir) {
                curr_tile = adj_positions(curr_tile.0, curr_tile.1)[dir];
                curr_move = self.get_movement_data(curr_tile.0, curr_tile.1, floor);
                let dx = (curr_tile.0 as isize - ox as isize + 10) as usize;
                let dy = (curr_tile.1 as isize - oy as isize + 10) as usize;
                visited[dx][dy] = true;
                d += 1;
            }
        }
    }

    fn surge_offset(&mut self, x: usize, y: usize, floor: usize, direction: usize) -> u8 {
        let bd_data = self.get_bd_data(x, y, floor);
        let (d_x, d_y): (i64, i64) = match direction {
            0 => (0, 1),
            1 => (1, 1),
            2 => (1, 0),
            3 => (1, -1),
            4 => (0, -1),
            5 => (-1, -1),
            6 => (-1, 0),
            7 => (-1, 1),
            _ => panic!()
        };
        let mut current: i64 = 220;
        let mut offset = 0;
        for i in 0..10 {
            current += d_x + d_y * 21;
            if (bd_data[current as usize / 64] >> (current as usize % 64)) & 1 == 1 {
                offset = 1 + i;
            }
        }
        offset
    }

    fn escape_offset(&mut self, x: usize, y: usize, floor: usize, direction: usize) -> u8 {
        let bd_data = self.get_bd_data(x, y, floor);
        let (d_x, d_y): (i64, i64) = match direction {
            0 => (0, -1),
            1 => (-1, -1),
            2 => (-1, 0),
            3 => (-1, 1),
            4 => (0, 1),
            5 => (1, 1),
            6 => (1, 0),
            7 => (1, -1),
            _ => panic!()
        };
        let mut current: i64 = 220;
        let mut offset = 0;
        for i in 0..7 {
            current += d_x + d_y * 21;
            if (bd_data[current as usize / 64] >> (current as usize % 64)) & 1 == 1 {
                offset = 1 + i;
            }
        }
        offset
    }

    // Returns a stack-allocated [u64; 7] instead of a heap-allocated Array1<u64>
    fn get_bd_data(&mut self, x: usize, y: usize, floor: usize) -> [u64; 7] {
        if x < RS_LENGTH && y < RS_HEIGHT {
            let chunk_size = 1280;
            let (chunk_x, chunk_y) = (x / chunk_size, y / chunk_size);
            let data = self.bd_data.entry((chunk_x, chunk_y, floor)).or_insert_with(|| {
                let path = format!("MapData/BD/bd-{chunk_x}-{chunk_y}-{floor}.npy");
                read_npy_compressed(path)
            });
            let xi = x % chunk_size;
            let yi = y % chunk_size;
            [data[[xi,yi,0]], data[[xi,yi,1]], data[[xi,yi,2]], data[[xi,yi,3]],
             data[[xi,yi,4]], data[[xi,yi,5]], data[[xi,yi,6]]]
        } else {
            [0; 7]
        }
    }

    fn get_movement_data(&mut self, x: usize, y: usize, floor: usize) -> u8 {
        if x < RS_LENGTH && y < RS_HEIGHT {
            let chunk_size = 1280;
            let (chunk_x, chunk_y) = (x / chunk_size, y / chunk_size);
            let data = self.movement_data.entry((chunk_x, chunk_y, floor)).or_insert_with(|| {
                let path = format!("MapData/Move/move-{chunk_x}-{chunk_y}-{floor}.npy");
                read_npy_compressed(path)
            });
            data[[x % chunk_size, y % chunk_size]]
        } else {
            0
        }
    }

    fn process_walk_data(&mut self, x: usize, y: usize, floor: usize) -> (u64, u64) {
        let tiles = self.walk_range(x, y, floor);
        let mut walk_data = u128::MAX;
        for tile in tiles {
            let u = x - 2;
            let v = y - 2;
            if tile.0 < RS_LENGTH && tile.1 < RS_HEIGHT {
                let temp = (15 - tile.2 as u128) << (4*(tile.0 - u + (tile.1 - v)*5));
                walk_data = walk_data - temp;
            }
        }
        (walk_data as u64, (walk_data >> 64) as u64)
    }

    fn process_bd_data(&mut self, x: usize, y: usize, floor: usize) -> [u64; 7] {
        let tiles = self.bd_range(x, y, floor);
        let mut bd_data = [0u64; 7];
        for tile in tiles {
            let u = x - 10;
            let v = y - 10;
            if tile.0 < RS_LENGTH && tile.1 < RS_HEIGHT {
                let temp = (tile.1 - v) * 21 + (tile.0 - u);
                let i = temp / 64;
                let j = temp % 64;
                bd_data[i] += 1 << j;
            }
        }
        bd_data
    }
}

fn build_movement_array(chunk_x: usize, chunk_y: usize, floor: usize) -> Array2<u8> {
    let path = format!("SourceData/collision-{chunk_x}-{chunk_y}-{floor}.bin");
    let data = fs::read(path).unwrap();
    let mut decoder = DeflateDecoder::new(&data);
    let decompressed_data = decoder.decode_zlib().unwrap();
    Array2::from_shape_vec((1280, 1280).f(), decompressed_data).unwrap()
}

fn process_movement_data(progress_bar: &ProgressBar) {
    let chunks: Vec<(usize, usize, usize)> = (0..5)
        .flat_map(|i| (0..10).flat_map(move |j| (0..4).map(move |k| (i, j, k))))
        .collect();
    chunks.into_par_iter().for_each(|(i, j, k)| {
        let arr = build_movement_array(i, j, k);
        let path = format!("MapData/Move/move-{i}-{j}-{k}.npy");
        write_npy_compressed(path, &arr);
        progress_bar.inc(1);
    });
}

fn build_walk_array(chunk_x: usize, chunk_y: usize, floor: usize) -> Array3<u64> {
    let chunk_size = 1280;
    let mut process = Process::new();
    let mut walk_array = Array3::zeros([chunk_size, chunk_size, 2]);
    let start_x = chunk_x * chunk_size;
    let start_y = chunk_y * chunk_size;
    for i in 0..chunk_size {
        for j in 0..chunk_size {
            let walk_data = process.process_walk_data(start_x + i, start_y + j, floor);
            walk_array[[i, j, 0]] = walk_data.0;
            walk_array[[i, j, 1]] = walk_data.1;
        }
    }
    walk_array
}

fn process_walk_data(progress_bar: &ProgressBar) {
    let chunks: Vec<(usize, usize, usize)> = (0..5)
        .flat_map(|i| (0..10).flat_map(move |j| (0..4).map(move |k| (i, j, k))))
        .collect();
    chunks.into_par_iter().for_each(|(i, j, k)| {
        let arr = build_walk_array(i, j, k);
        let path = format!("MapData/Walk/walk-{i}-{j}-{k}.npy");
        write_npy_compressed(path, &arr);
        progress_bar.inc(1);
    });
}

fn build_bd_array(chunk_x: usize, chunk_y: usize, floor: usize) -> Array3<u64> {
    let chunk_size = 1280;
    let mut process = Process::new();
    let mut bd_array = Array3::zeros([chunk_size, chunk_size, 7]);
    let start_x = chunk_x * chunk_size;
    let start_y = chunk_y * chunk_size;
    for i in 0..chunk_size {
        for j in 0..chunk_size {
            let bd_data = process.process_bd_data(start_x + i, start_y + j, floor);
            for k in 0..7 {
                bd_array[[i, j, k]] = bd_data[k];
            }
        }
    }
    bd_array
}

fn process_bd_data(progress_bar: &ProgressBar) {
    let chunks: Vec<(usize, usize, usize)> = (0..5)
        .flat_map(|i| (0..10).flat_map(move |j| (0..4).map(move |k| (i, j, k))))
        .collect();
    chunks.into_par_iter().for_each(|(i, j, k)| {
        let arr = build_bd_array(i, j, k);
        let path = format!("MapData/BD/bd-{i}-{j}-{k}.npy");
        write_npy_compressed(path, &arr);
        progress_bar.inc(1);
    });
}

fn build_se_array(chunk_x: usize, chunk_y: usize, floor: usize) -> Array3<u8> {
    let chunk_size = 1280;
    let mut process = Process::new();
    let mut se_array = Array3::zeros([chunk_size, chunk_size, 8]);
    let start_x = chunk_x * chunk_size;
    let start_y = chunk_y * chunk_size;
    for i in 0..chunk_size {
        for j in 0..chunk_size {
            for direction in 0..8 {
                let s_data = process.surge_offset(start_x + i, start_y + j, floor, direction);
                let e_data = process.escape_offset(start_x + i, start_y + j, floor, direction);
                se_array[[i, j, direction]] = s_data + e_data * 16;
            }
        }
    }
    se_array
}

fn process_se_data(progress_bar: &ProgressBar) {
    let chunks: Vec<(usize, usize, usize)> = (0..5)
        .flat_map(|i| (0..10).flat_map(move |j| (0..4).map(move |k| (i, j, k))))
        .collect();
    chunks.into_par_iter().for_each(|(i, j, k)| {
        let arr = build_se_array(i, j, k);
        let path = format!("MapData/SE/se-{i}-{j}-{k}.npy");
        write_npy_compressed(path, &arr);
        progress_bar.inc(1);
    });
}

// Computes the heuristic table by iterating distance in ascending order and reading
// already-computed sub-problem values directly from arr, eliminating the HashMap memo.
// Dimensions: [distance, scd, sscd, ecd, secd, bdcd]
fn process_heuristic_data(max_distance: usize) {
    let mut arr: Array6<u64> = Array6::zeros([max_distance+1, 18, 18, 18, 18, 18]);
    // distance=0 is already 0 from zeros(); start from 1
    for distance in 1..=max_distance {
        for scd in 0..=17usize {
            for sscd in 0..=17usize {
                for ecd in 0..=17usize {
                    for secd in 0..=17usize {
                        for bdcd in 0..=17usize {
                            let d = distance;
                            let mut result = usize::MAX;
                            // BD
                            if bdcd == 0 {
                                result = result.min(arr[[d.saturating_sub(10), scd, sscd, ecd, secd, 17]] as usize);
                            }
                            // Surge primary
                            if scd == 0 {
                                result = result.min(arr[[d.saturating_sub(10), 17, max(2,sscd), max(2,ecd), max(2,secd), bdcd]] as usize);
                            }
                            // Surge secondary
                            else if sscd == 0 {
                                result = result.min(arr[[d.saturating_sub(10), max(2,scd), 17, max(2,ecd), max(2,secd), bdcd]] as usize);
                            }
                            // Escape primary
                            if ecd == 0 {
                                result = result.min(arr[[d.saturating_sub(7), max(2,scd), max(2,sscd), 17, max(2,secd), bdcd]] as usize);
                            }
                            // Escape secondary
                            else if secd == 0 {
                                result = result.min(arr[[d.saturating_sub(7), max(2,scd), max(2,sscd), max(2,ecd), 17, bdcd]] as usize);
                            }
                            // Walk (no surge and no BD available)
                            if scd != 0 && sscd != 0 && bdcd != 0 {
                                let walk = arr[[d.saturating_sub(2), max(scd,1)-1, max(sscd,1)-1, max(ecd,1)-1, max(secd,1)-1, max(bdcd,1)-1]] as usize + 1;
                                result = result.min(walk);
                            }
                            arr[[distance, scd, sscd, ecd, secd, bdcd]] = result as u64;
                        }
                    }
                }
            }
        }
    }
    write_npy_compressed("HeuristicData/l_infinity_cds.npy", &arr);
}

pub fn compress_existing() {
    let mut paths: Vec<String> = Vec::new();
    paths.push("HeuristicData/l_infinity_cds.npy".to_string());
    for i in 0..5 {
        for j in 0..10 {
            for k in 0..4 {
                paths.push(format!("MapData/Move/move-{i}-{j}-{k}.npy"));
                paths.push(format!("MapData/Walk/walk-{i}-{j}-{k}.npy"));
                paths.push(format!("MapData/BD/bd-{i}-{j}-{k}.npy"));
                paths.push(format!("MapData/SE/se-{i}-{j}-{k}.npy"));
            }
        }
    }
    paths.into_par_iter().for_each(|path| {
        if Path::new(&path).try_exists().unwrap_or(false) {
            let raw = fs::read(&path).unwrap();
            let file = fs::File::create(&path).unwrap();
            let mut encoder = ZlibEncoder::new(file, Compression::default());
            encoder.write_all(&raw).unwrap();
            encoder.finish().unwrap();
        }
    });
}

pub fn setup(reset: bool) {
    let progress_bar = ProgressBar::new(801);
    fs::create_dir_all("MapData/BD").unwrap();
    fs::create_dir_all("MapData/Move").unwrap();
    fs::create_dir_all("MapData/SE").unwrap();
    fs::create_dir_all("MapData/Walk").unwrap();
    fs::create_dir_all("HeuristicData").unwrap();
    if !Path::new("HeuristicData/l_infinity_cds.npy").try_exists().unwrap() || reset {
        progress_bar.set_message("Generating heuristic data");
        process_heuristic_data(500);
    }
    progress_bar.inc(1);
    let mut moves = true;
    'a: for i in 0..5 {
        for j in 0..10 {
            for k in 0..4 {
                let path = format!("MapData/Move/move-{i}-{j}-{k}.npy");
                if !Path::new(&path).try_exists().unwrap() {
                    moves = false;
                    break 'a;
                }
            }
        }
    }
    if !moves || reset {
        process_movement_data(&progress_bar);
    } else {
        progress_bar.inc(200);
    }
    let mut walk = true;
    'a: for i in 0..5 {
        for j in 0..10 {
            for k in 0..4 {
                let path = format!("MapData/Walk/walk-{i}-{j}-{k}.npy");
                if !Path::new(&path).try_exists().unwrap() {
                    walk = false;
                    break 'a;
                }
            }
        }
    }
    if !walk || reset {
        process_walk_data(&progress_bar);
    } else {
        progress_bar.inc(200);
    }
    let mut bd = true;
    'a: for i in 0..5 {
        for j in 0..10 {
            for k in 0..4 {
                let path = format!("MapData/BD/bd-{i}-{j}-{k}.npy");
                if !Path::new(&path).try_exists().unwrap() {
                    bd = false;
                    break 'a;
                }
            }
        }
    }
    if !bd || reset {
        process_bd_data(&progress_bar);
    } else {
        progress_bar.inc(200);
    }
    let mut se = true;
    'a: for i in 0..5 {
        for j in 0..10 {
            for k in 0..4 {
                let path = format!("MapData/SE/se-{i}-{j}-{k}.npy");
                if !Path::new(&path).try_exists().unwrap() {
                    se = false;
                    break 'a;
                }
            }
        }
    }
    if !se || reset {
        process_se_data(&progress_bar);
    } else {
        progress_bar.inc(200);
    }
    progress_bar.finish();
}
