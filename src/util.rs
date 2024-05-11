pub fn free_direction(data: u8, direction: usize) -> bool{
    let t = [2, 32, 4, 64, 8, 128, 1, 16];
    data & t[direction] != 0
}

pub fn adj_positions(x: usize, y:usize) -> [(usize, usize); 8] {
    [(x, y + 1), (x + 1, y + 1), (x + 1, y), (x + 1, y - 1), (x, y - 1), (x - 1, y - 1), (x - 1, y), (x - 1, y + 1)]
}