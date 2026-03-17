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

pub fn free_direction(data: u8, direction: usize) -> bool{
    let t = [2, 32, 4, 64, 8, 128, 1, 16];
    data & t[direction] != 0
}

pub fn adj_positions(x: usize, y:usize) -> [(usize, usize); 8] {
    [(x, y + 1), (x + 1, y + 1), (x + 1, y), (x + 1, y - 1), (x, y - 1), (x - 1, y - 1), (x - 1, y), (x - 1, y + 1)]
}