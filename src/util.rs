use std::{fs::File, io::{Cursor, Read, Write}, path::Path};
use flate2::{Compression, read::ZlibDecoder, write::ZlibEncoder};
use ndarray::{Array, ArrayBase, Data, Dimension};
use ndarray_npy::{ReadableElement, ReadNpyExt, WritableElement, WriteNpyExt};

pub fn write_npy_compressed<S, D, P>(path: P, array: &ArrayBase<S, D>)
where
    P: AsRef<Path>,
    S: Data,
    S::Elem: WritableElement,
    D: Dimension,
{
    let mut npy_bytes = Vec::new();
    array.write_npy(&mut npy_bytes).unwrap();
    let file = File::create(path).unwrap();
    let mut encoder = ZlibEncoder::new(file, Compression::default());
    encoder.write_all(&npy_bytes).unwrap();
    encoder.finish().unwrap();
}

pub fn read_npy_compressed<A, D, P>(path: P) -> Array<A, D>
where
    P: AsRef<Path>,
    A: ReadableElement,
    D: Dimension,
    Array<A, D>: ReadNpyExt,
{
    let file = File::open(path).unwrap();
    let mut decoder = ZlibDecoder::new(file);
    let mut npy_bytes = Vec::new();
    decoder.read_to_end(&mut npy_bytes).unwrap();
    Array::<A, D>::read_npy(Cursor::new(npy_bytes)).unwrap()
}

/// Returns true if the collision byte `data` has movement allowed in `direction`.
/// Directions 0–7 = N, NE, E, SE, S, SW, W, NW.
/// The bitmask positions in the collision byte are non-sequential: [N=2, NE=32, E=4, SE=64, S=8, SW=128, W=1, NW=16].
pub fn free_direction(data: u8, direction: usize) -> bool{
    let t = [2, 32, 4, 64, 8, 128, 1, 16];
    data & t[direction] != 0
}

/// Returns the 8 adjacent tile coordinates in direction order N, NE, E, SE, S, SW, W, NW.
pub fn adj_positions(x: usize, y:usize) -> [(usize, usize); 8] {
    [(x, y + 1), (x + 1, y + 1), (x + 1, y), (x + 1, y - 1), (x, y - 1), (x - 1, y - 1), (x - 1, y), (x - 1, y + 1)]
}