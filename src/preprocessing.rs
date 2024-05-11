use std::{collections::{HashSet, HashMap, VecDeque},
          cmp::{max, min},
          path::Path,
          fs};
use zune_inflate::DeflateDecoder;
use ndarray::{Array1, Array2, Array3, Array5, ShapeBuilder};
use ndarray_npy::{read_npy, write_npy};
use indicatif::ProgressBar;
use crate::util::{adj_positions, free_direction};

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
        let mut visited = HashSet::new();
        visited.insert((x, y));
        let mut queue = VecDeque::new();
        for i in 0..8 {
            let j = (2*i + i/4) % 8;
            if free_direction(start, j) {
                tiles.push((adj[j].0, adj[j].1, j));
                visited.insert(adj[j]);
                queue.push_back(adj[j]);
            }
        }
        while !queue.is_empty() {
            let current = queue.pop_front().unwrap();
            let current_move_data = self.get_movement_data(current.0, current.1, floor);
            let temp_adj = adj_positions(current.0, current.1);
            for i in 0..8 {
                if free_direction(current_move_data, i) {
                    if !visited.contains(&temp_adj[i]) {
                        tiles.push((temp_adj[i].0, temp_adj[i].1, i));
                        visited.insert(temp_adj[i]);
                    }
                }
            }
        }
        tiles
    }

    fn bd_range(&mut self, x: usize, y: usize, floor: usize) -> Vec<(usize, usize)> {
        let mut set = HashSet::with_capacity(441);
        self.bd_range_recursion(x, y, floor,1, 2, 0, 0, 0, &mut set);
        self.bd_range_recursion(x, y, floor,3, 2, 4, 0, 0, &mut set);
        self.bd_range_recursion(x, y, floor,5, 6, 4, 0, 0, &mut set);
        self.bd_range_recursion(x, y, floor,7, 6, 0, 0, 0, &mut set);
        set.into_iter().collect()
    }

    fn bd_range_recursion(&mut self, x: usize, y: usize, floor: usize, direction: usize, horizontal: usize, vertical: usize, dist_x: usize, dist_y: usize, tiles: &mut HashSet<(usize, usize)>) {
        let mut dist_x = dist_x;
        let mut dist_y = dist_y;
        if dist_x > 0 || dist_y > 0 {
            tiles.insert((x, y));
        }
        let curr_move = self.get_movement_data(x, y, floor);
        if dist_x < 10 && dist_y < 10 && free_direction(curr_move, direction) {
            let new_tile = adj_positions(x, y)[direction];
            self.bd_range_recursion(new_tile.0, new_tile.1, floor, direction, horizontal, vertical,dist_x + 1, dist_y + 1, tiles);
        }
        else if dist_x < 10 && free_direction(curr_move, horizontal) {
            let new_tile = adj_positions(x, y)[horizontal];
            self.bd_range_recursion(new_tile.0, new_tile.1, floor, direction, horizontal, vertical,dist_x + 1, dist_y, tiles);
            dist_x = 10;
        }
        else if dist_y < 10 && free_direction(curr_move, vertical) {
            let new_tile = adj_positions(x, y)[vertical];
            self.bd_range_recursion(new_tile.0, new_tile.1, floor, direction, horizontal, vertical,dist_x, dist_y + 1, tiles);
            dist_y = 10;
        }
        let dist = [dist_x, dist_y];
        for (i, direction) in [horizontal, vertical].into_iter().enumerate(){
            let mut d = dist[i];
            let mut curr_tile = (x, y);
            let mut curr_move = self.get_movement_data(x, y, floor);
            while d < 10 && free_direction(curr_move, direction) {
                curr_tile = adj_positions(curr_tile.0, curr_tile.1)[direction];
                curr_move = self.get_movement_data(curr_tile.0, curr_tile.1, floor);
                tiles.insert(curr_tile);
                d += 1;
            }
        }
    }

    fn surge_offset(&mut self, x: usize, y: usize, floor: usize, direction: usize) -> u8{
        let bd_data = self.get_bd_data(x, y, floor);
        let d_x;
        let d_y;
        let mut current = 220;
        let mut offset = 0;
        match direction {
            0 => {d_x = 0; d_y = 1;},
            1 => {d_x = 1; d_y = 1;},
            2 => {d_x = 1; d_y = 0;},
            3 => {d_x = 1; d_y = -1;},
            4 => {d_x = 0; d_y = -1;},
            5 => {d_x = -1; d_y = -1;},
            6 => {d_x = -1; d_y = 0;},
            7 => {d_x = -1; d_y = 1;},
            _ => panic!()
        }
        for i in 0..10 {
            current += d_x + d_y * 21;
            if (bd_data[[current as usize / 64]] >> (current % 64)) & 1 == 1 {
                offset = 1 + i;
            }
        }
        offset
    }

    fn escape_offset(&mut self, x: usize, y: usize, floor: usize, direction: usize) -> u8{
        let bd_data = self.get_bd_data(x, y, floor);
        let d_x;
        let d_y;
        let mut current = 220;
        let mut offset = 0;
        match direction {
            0 => {d_x = 0; d_y = -1;},
            1 => {d_x = -1; d_y = -1;},
            2 => {d_x = -1; d_y = 0;},
            3 => {d_x = -1; d_y = 1;},
            4 => {d_x = 0; d_y = 1;},
            5 => {d_x = 1; d_y = 1;},
            6 => {d_x = 1; d_y = 0;},
            7 => {d_x = 1; d_y = -1;},
            _ => panic!()
        }
        for i in 0..7 {
            current += d_x + d_y * 21;
            if (bd_data[[current as usize / 64]] >> (current % 64)) & 1 == 1 {
                offset = 1 + i;
            }
        }
        offset
    }

    fn get_bd_data(&mut self, x: usize, y: usize, floor: usize) -> Array1<u64> {
        if x < RS_LENGTH && y < RS_HEIGHT {
            let chunk_size = 1280;
            let (chunk_x, chunk_y) = (x / chunk_size, y / chunk_size);
            if let Some(data) = self.bd_data.get(&(chunk_x, chunk_y, floor)) {
                data.slice(ndarray::s![x % chunk_size, y % chunk_size, ..]).to_owned()
            } else {
                let path = format!("MapData/BD/bd-{chunk_x}-{chunk_y}-{floor}.npy");
                let data: Array3<u64> = read_npy(path).unwrap();
                let result = data.slice(ndarray::s![x % chunk_size, y % chunk_size, ..]).to_owned();
                self.bd_data.insert((chunk_x, chunk_y, floor), data);
                result
            }
        } else {
            Array1::from(vec![0,0,0,0,0,0,0])
        }
    }

    fn get_movement_data(&mut self, x: usize, y: usize, floor: usize) -> u8 {
        if x < RS_LENGTH && y < RS_HEIGHT {
            let chunk_size = 1280;
            let (chunk_x, chunk_y) = (x / chunk_size, y / chunk_size);
            if let Some(data) = self.movement_data.get(&(chunk_x, chunk_y, floor)) {
                data[[x % chunk_size, y % chunk_size]]
            } else {
                let path = format!("MapData/Move/move-{chunk_x}-{chunk_y}-{floor}.npy");
                let data: Array2<u8> = read_npy(path).unwrap();
                let result = data[[x % chunk_size, y % chunk_size]];
                self.movement_data.insert((chunk_x, chunk_y, floor), data);
                result
            }
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
        let mut bd_data = [0, 0, 0, 0, 0, 0, 0];
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
    for i in 0..5 {
        for j in 0..10 {
            for k in 0..4 {
                progress_bar.inc(1);
                let arr = build_movement_array(i, j ,k);
                let path = format!("MapData/Move/move-{i}-{j}-{k}.npy");
                write_npy(path, &arr).unwrap();
            }
        }
    }
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
    for i in 0..5 {
        for j in 0..10 {
            for k in 0..4 {
                progress_bar.inc(1);
                let arr = build_walk_array(i, j ,k);
                let path = format!("MapData/Walk/walk-{i}-{j}-{k}.npy");
                write_npy(path, &arr).unwrap();
            }
        }
    }
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
    for i in 0..5 {
        for j in 0..10 {
            for k in 0..4 {
                progress_bar.inc(1);
                let arr = build_bd_array(i, j ,k);
                let path = format!("MapData/BD/bd-{i}-{j}-{k}.npy");
                write_npy(path, &arr).unwrap();
            }
        }
    }
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
    for i in 0..5 {
        for j in 0..10 {
            for k in 0..4 {
                progress_bar.inc(1);
                let arr = build_se_array(i, j ,k);
                let path = format!("MapData/SE/se-{i}-{j}-{k}.npy");
                write_npy(path, &arr).unwrap();
            }
        }
    }
}

fn process_heuristic_data(max_distance: usize) {
    let mut arr : Array5<u64> = Array5::zeros([max_distance+1, 18, 18, 18, 18]);
    let mut memo = Memo::new();
    for distance in 0..=max_distance {
        for secd in 0..=17 {
            for scd in 0..=17 {
                for ecd in 0..=17 {
                    for bdcd in 0..=17 {
                        arr[[distance, secd, scd, ecd, bdcd]] = memo.distance_cds_rec(distance as isize, secd, scd, ecd, bdcd) as u64;
                    }
                }
            }
        }
    }
    write_npy("HeuristicData/l_infinity_cds.npy", &arr).unwrap();
}

struct Memo {
    data: HashMap<(isize, usize, usize, usize, usize), usize>
}

impl Memo {
    fn new() -> Memo {
        Memo {
            data: HashMap::new()
        }
    }

    fn distance_cds_rec(&mut self, distance: isize, secd: usize, scd: usize, ecd: usize, bdcd: usize) -> usize {
        if self.data.contains_key(&(distance, secd, scd, ecd, bdcd)) {
            return *self.data.get(&(distance, secd, scd, ecd, bdcd)).unwrap()
        }
        if distance <= 0 {
            return 0;
        }
        let mut bd = usize::MAX;
        let mut surge = usize::MAX;
        let mut escape = usize::MAX;
        let mut walk = usize::MAX;
        if bdcd == 0 {
            bd = self.distance_cds_rec(distance - 10, secd, scd, ecd, 17);
        }
        if secd == 0 {
            surge = self.distance_cds_rec(distance - 10, 17, max(2, scd), 17, bdcd);
        } else if scd == 0 {
            surge = self.distance_cds_rec(distance - 10, max(2, secd), 17, max(2, ecd), bdcd);
        }
        if secd == 0 {
            escape = self.distance_cds_rec(distance - 7, 17, 17, max(2, ecd), bdcd);
        } else if ecd == 0 {
            escape = self.distance_cds_rec(distance - 7, max(2, secd), max(2, scd), 17, bdcd);
        }
        if secd != 0 && bdcd != 0 {
            walk = self.distance_cds_rec(distance - 2, max(secd, 1) - 1, max(scd, 1) - 1, max(ecd, 1) - 1, max(bdcd, 1) - 1) + 1;
        }
        let result = min(min(min(bd, surge), escape), walk);
        self.data.insert((distance, secd, scd, ecd, bdcd), result);
        result
    }
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

