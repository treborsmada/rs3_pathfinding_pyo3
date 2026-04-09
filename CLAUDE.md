# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Rust library (exposed to Python via PyO3/maturin) that computes optimal paths in RuneScape 3, accounting for movement abilities: **Surge**, **Escape**, and **Bladed Dive (BD)**. The A* search operates over a `State` that includes position, direction, and ability cooldowns.

## Build & Development Commands

```bash
# Build and install the Python extension in-place (dev)
maturin develop

# Build and install optimised (use for benchmarking/running)
maturin develop --release

# Build release wheel
maturin build --release

# Cargo check/build (Rust only, no Python)
cargo check
cargo build --release
```

There are no automated tests in this project currently.

## Architecture

### Data Pipeline (two stages)

**Stage 1 — Source data → MapData (`preprocessing.rs setup()`)**
Raw collision data lives in `SourceData/collision-{chunk_x}-{chunk_y}-{floor}.bin` (zlib-compressed). `setup()` decompresses these and generates four derived `.npy` datasets under `MapData/`:
- `Move/move-{x}-{y}-{floor}.npy` — per-tile walkability flags (u8 bitmask, 8 directions)
- `Walk/walk-{x}-{y}-{floor}.npy` — reachable walk destinations within a 5×5 window, packed into two u64s. Each of the 25 grid positions occupies a 4-bit nibble; the nibble holds the direction (0–7) if that tile is reachable, or 15 if not.
- `BD/bd-{x}-{y}-{floor}.npy` — reachable Bladed Dive destinations within a 21×21 window, packed into 7 u64s as a bitset (bit set = reachable).
- `SE/se-{x}-{y}-{floor}.npy` — Surge/Escape reach per direction. Low nibble = Surge offset (up to 10), high nibble = Escape offset (up to 7).

`HeuristicData/l_infinity_cds.npy` is also generated: a 6D lookup table `[distance, scd, sscd, ecd, secd, bdcd]` (dimensions 501×18×18×18×18×18) for the A* admissible heuristic. It is loaded lazily via a `OnceLock` and cached for the process lifetime — the **first A* call per process pays a ~11 s load cost**.

The world is split into chunks of 1280×1280 tiles. The map is 6400 (x) × 12800 (y) tiles across 4 floors (indices 0–3), giving 5×10×4 = 200 chunks per dataset.

The 200 chunks per stage are processed in parallel using Rayon. Per-tile intermediate data uses stack-allocated fixed arrays (`[[bool;5];5]` for walk, `[[bool;21];21]` for BD) to avoid heap allocation in the hot loop. The heuristic table is built iteratively (distance 0→500) reading sub-problems directly from the output array rather than a HashMap memo.

**Stage 2 — A* pathfinding (`lib.rs` → `pathfinding.rs`)**
`MapSection::create_map_section()` loads the relevant `.npy` chunks for the given bounding box and builds flat `Vec<Vec<(u16, u16, u8)>>` neighbour lists (indexed by `xi * y_len + yi`) for walk and BD moves. The A* search then expands `State` nodes using walk moves (cost 1), BD (cost 0), Surge (cost 0), or Escape (cost 0).

### State (`state.rs`)

`State` has: `pos_x`, `pos_y`, `direction` (0–7), and **five** cooldown fields:

| Field | Ability |
|-------|---------|
| `scd` | Surge primary cooldown |
| `sscd` | Surge secondary cooldown |
| `ecd` | Escape primary cooldown |
| `secd` | Escape/Surge shared cooldown |
| `bdcd` | Bladed Dive cooldown |

All cooldowns count down from 17. `update()` decrements all by 1 via `saturating_sub`.

Ability availability and cooldown side-effects on use:
- **Surge**: usable when `scd == 0` OR `sscd == 0`. Using it sets the consumed slot to 17 and cross-triggers all other ability cooldowns to `max(2, current)`.
- **Escape**: usable when `ecd == 0` OR `secd == 0`. Same cross-trigger pattern.
- **BD**: usable when `bdcd == 0`. Sets `bdcd = 17`; no cross-trigger.

Goal is a 3×3 region around the end tile (`at_goal`).

### Direction encoding

Directions 0–7 = N, NE, E, SE, S, SW, W, NW. The `free_direction` util maps these to bitmask positions `[2, 32, 4, 64, 8, 128, 1, 16]` in the collision u8.

### Python API

```python
import rs3_pathfinding

# One-time preprocessing (generates MapData/ from SourceData/)
rs3_pathfinding.setup(reset=False)   # reset=True forces regeneration of all files

# One-shot pathfinding (creates a fresh MapSection each call — slow for repeated use)
path, ticks = rs3_pathfinding.a_star((start_x, start_y), (end_x, end_y), floor, setup=False,
                                     teleports=[(x, y, cost), ...])
# path: list of (pos_x, pos_y, direction, scd, sscd, ecd, secd, bdcd) tuples
# ticks: number of game ticks to reach the goal
# teleports: optional list of (x: int, y: int, cost: int) tuples — actions available from any
#            state that teleport to (x, y) and consume `cost` ticks (cost must fit in u8).
#            The map section is automatically expanded to cover all teleport destinations.

# Preloaded workflow — for repeated queries over the same area (used by pomcp.py)
rs3_pathfinding.preload_section(x_min, x_max, y_min, y_max, floor)
# Loads MapSection into a static OnceLock; subsequent calls are no-ops.

path, ticks = rs3_pathfinding.a_star_preloaded(
    (start_x, start_y), (end_x, end_y),
    direction, scd, sscd, ecd, secd, bdcd,
    teleports=[(x, y, cost), ...],
)

# Heuristic-only cost estimate (no pathfinding, reads cached OnceLock table)
cost = rs3_pathfinding.heuristic_cost(x, y, tx, ty, scd, sscd, ecd, secd, bdcd)
```

**Performance note:** The first `a_star` or `a_star_preloaded` call per process pays a ~11 s one-time cost to load `HeuristicData/l_infinity_cds.npy` into the `HEURISTIC_DATA` `OnceLock`. All subsequent calls use the cached data and are fast.

### Dependencies

| Crate | Purpose |
|---|---|
| `pyo3` | Python bindings |
| `ndarray` / `ndarray-npy` | N-dimensional arrays and `.npy` file I/O |
| `flate2` / `zune-inflate` | Zlib compression/decompression of `.npy` files |
| `pathfinding` | A* implementation |
| `rayon` | Data-parallel chunk preprocessing |
| `indicatif` | Progress bar during preprocessing |
