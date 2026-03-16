# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Rust library (exposed to Python via PyO3/maturin) that computes optimal paths in RuneScape 3, accounting for movement abilities: **Surge**, **Escape**, and **Bladed Dive (BD)**. The A* search operates over a `State` that includes position, direction, and ability cooldowns.

## Build & Development Commands

```bash
# Build the Python extension (development, in-place)
maturin develop

# Build release wheel
maturin build --release

# Cargo check/build (Rust only, no Python)
cargo check
cargo build --release
```

There are no automated tests in this project currently.

## Architecture

### Data Pipeline (two stages)

**Stage 1 — Source data → MapData (preprocessing.rs `setup()`)**
Raw collision data lives in `SourceData/collision-{chunk_x}-{chunk_y}-{floor}.bin` (zlib-compressed). `setup()` decompresses these and generates four derived `.npy` datasets under `MapData/`:
- `Move/move-{x}-{y}-{floor}.npy` — per-tile walkability flags (u8 bitmask, 8 directions)
- `Walk/walk-{x}-{y}-{floor}.npy` — precomputed reachable walk destinations within a 5×5 window (packed into two u64s)
- `BD/bd-{x}-{y}-{floor}.npy` — reachable Bladed Dive destinations within a 21×21 window (packed into 7 u64s)
- `SE/se-{x}-{y}-{floor}.npy` — Surge/Escape offsets per direction (packed low/high nibble of u8)

`HeuristicData/l_infinity_cds.npy` is also generated: a 5D lookup table `[distance, secd, scd, ecd, bdcd]` for the A* admissible heuristic.

The world is split into chunks of 1280×1280 tiles. The map is 6400 (x) × 12800 (y) tiles across 4 floors (indices 0–3).

**Stage 2 — A* pathfinding (lib.rs → pathfinding.rs)**
`MapSection::create_map_section()` loads relevant chunks for the bounding box around the start/end (±150 tile radius) and builds in-memory hashmaps for walk and BD neighbors. The A* search then expands `State` nodes using walk moves (cost 1), BD teleport (cost 0), Surge (cost 0), or Escape (cost 0).

### State (`state.rs`)

`State` has: `pos_x`, `pos_y`, `direction` (0–7, N/NE/E/SE/S/SW/W/NW), and four cooldown fields (`secd`, `scd`, `ecd`, `bdcd`), each counting down from 17. `update()` decrements all cooldowns by 1 per tick. Ability availability:
- **Surge**: usable when `secd == 0` OR `scd == 0`
- **Escape**: usable when `secd == 0` OR `ecd == 0`
- **BD**: usable when `bdcd == 0`

Goal is a 3×3 region around the end tile (`at_goal`).

### Direction encoding

Directions 0–7 = N, NE, E, SE, S, SW, W, NW. The `free_direction` util maps these to bitmask positions `[2, 32, 4, 64, 8, 128, 1, 16]` in the collision u8.

### Python API

```python
import rs3_pathfinding
path, ticks = rs3_pathfinding.a_star((start_x, start_y), (end_x, end_y), floor)
# path: list of (pos_x, pos_y, direction, secd, scd, ecd, bdcd) tuples
# ticks: number of game ticks to reach the goal
```

`setup(reset=True)` is called automatically on each `a_star` invocation; it regenerates all MapData. Pass `reset=False` (change in `lib.rs`) to skip regeneration when data is already present.
