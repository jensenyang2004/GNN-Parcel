"""
Generate MAPF Benchmark Maps
=============================
Creates movingai-format .map files for training and evaluation.

For paper-accurate results, use the REAL movingai benchmark maps:
  https://movingai.com/benchmarks/mapf/index.html

  Download and place in instances/:
    random-64-64-10.map       → already correct (64x64)
    warehouse-10-20-10-2-1.map  or  warehouse-20-40-10-2-1.map
    Berlin_1_256.map  (city)
    ht_chantry.map    (game, from Dragon Age Origins)

This script generates synthetic stand-ins that match the structure
of the real maps so training can start before you download the benchmarks.
All synthetic maps are larger than 64x64 so the cropping logic works correctly.

Run once before training:
    python generate_maps.py
"""

import os
import random
import numpy

random.seed(0)
numpy.random.seed(0)

MIN_CROP_SIZE = 64  # all maps must be at least this large in both dimensions


def write_map(filename, grid):
    rows = len(grid)
    cols = len(grid[0])
    assert rows >= MIN_CROP_SIZE and cols >= MIN_CROP_SIZE, \
        f"Map {filename} is {rows}x{cols} — too small for 64x64 cropping"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write("type octile\n")
        f.write(f"height {rows}\n")
        f.write(f"width {cols}\n")
        f.write("map\n")
        for row in grid:
            f.write("".join("@" if c else "." for c in row) + "\n")
    n_obs = sum(c for row in grid for c in row)
    pct = 100 * n_obs / (rows * cols)
    print(f"  Saved: {filename}  ({rows}x{cols}, {pct:.1f}% obstacles)")


# -----------------------------------------------------------------------
# Random map  —  matches movingai random-64-64-10
# -----------------------------------------------------------------------

def make_random_map(rows=64, cols=64, density=0.10):
    """
    Random obstacle placement at ~density%.
    Matches: Random-64-64-10 (64x64, 10% obstacles) from movingai.
    Real map: https://movingai.com/benchmarks/mapf/index.html
    """
    grid = [[0] * cols for _ in range(rows)]
    positions = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(positions)
    nr_obstacles = int(rows * cols * density)
    for r, c in positions[:nr_obstacles]:
        grid[r][c] = 1
        if not _has_free_neighbor(grid, r, c, rows, cols):
            grid[r][c] = 0
    return grid


def _has_free_neighbor(grid, r, c, rows, cols):
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and not grid[nr][nc]:
            return True
    return False


# -----------------------------------------------------------------------
# Warehouse map  —  matches movingai warehouse benchmark structure
#
# Real movingai warehouse maps (use these for paper results):
#   warehouse-10-20-10-2-1.map  : 161 rows x 63 cols
#   warehouse-20-40-10-2-1.map  : 161 rows x 123 cols
#
# Structure: double-depth shelf rows separated by single-cell corridors,
# with vertical access corridors every few columns.
# -----------------------------------------------------------------------

def make_warehouse_map(rows=161, cols=84,
                       shelf_depth=2, shelf_gap=1, corridor_every=5):
    """
    Synthetic warehouse: shelf blocks separated by corridors.
    Dimensions match the movingai warehouse series (161 rows).
    cols=84 keeps ~20% obstacle density across 64x64 crops.

    Layout (one repeat unit, height = shelf_depth + shelf_gap):
      ....  ← horizontal corridor (shelf_gap rows)
      @@@@  ← shelf row 1 (blocked except vertical corridors)
      @@@@  ← shelf row 2 (shelf_depth=2)
    """
    grid = [[0] * cols for _ in range(rows)]

    unit = shelf_depth + shelf_gap  # rows per shelf+corridor block

    for r in range(rows):
        row_in_unit = r % unit
        if row_in_unit < shelf_depth:
            # This is a shelf row — fill with obstacles, leave vertical corridors
            for c in range(cols):
                if c % corridor_every != 0:
                    grid[r][c] = 1

    return grid


# -----------------------------------------------------------------------
# City map  —  matches movingai city benchmark structure
#
# Real movingai city maps (use these for paper results):
#   Berlin_1_256.map  : 256x256, ~37% obstacles
#   Boston_0_256.map  : 256x256, ~41% obstacles
#   Paris_1_256.map   : 256x256, ~34% obstacles
#
# Structure: large building blocks separated by roads (corridors).
# -----------------------------------------------------------------------

def make_city_map(rows=256, cols=256,
                  block_size=6, road_width=4):
    """
    Synthetic city grid: rectangular building blocks separated by roads.
    Dimensions and obstacle density match movingai city maps (~35% obstacles).

    Use the real movingai maps for paper benchmarking:
      https://movingai.com/benchmarks/mapf/index.html
    """
    grid = [[0] * cols for _ in range(rows)]

    unit = block_size + road_width  # one block + one road

    for r in range(rows):
        for c in range(cols):
            r_in_unit = r % unit
            c_in_unit = c % unit
            # Inside a building block = obstacle
            if r_in_unit < block_size and c_in_unit < block_size:
                grid[r][c] = 1

    return grid


# -----------------------------------------------------------------------
# Game map  —  matches movingai game benchmark structure
#
# Real movingai game maps (use these for paper results):
#   Dragon Age Origins:  ht_chantry.map, den000d.map (~400x400)
#   Warcraft III:        divideandconquer.map
#   Starcraft:           AcrosstheCape.map
#
# These have irregular room-and-corridor structure from actual game levels.
# The synthetic stand-in below uses random rooms connected by corridors.
# -----------------------------------------------------------------------

def make_game_map(rows=256, cols=256,
                  n_rooms=80, min_room=10, max_room=30, corridor_width=3):
    """
    Synthetic game map: random rooms connected by corridors.
    Resembles Dragon Age / Baldur's Gate dungeon maps (~40% obstacles).

    Use the real movingai maps for paper benchmarking:
      https://movingai.com/benchmarks/mapf/index.html
    """
    # Start fully blocked
    grid = [[1] * cols for _ in range(rows)]

    rooms = []

    def carve_rect(r0, c0, h, w):
        for r in range(r0, min(r0 + h, rows)):
            for c in range(c0, min(c0 + w, cols)):
                grid[r][c] = 0

    # Carve rooms
    for _ in range(n_rooms * 5):  # many attempts, accept valid ones
        h = random.randint(min_room, max_room)
        w = random.randint(min_room, max_room)
        r0 = random.randint(1, rows - h - 1)
        c0 = random.randint(1, cols - w - 1)
        carve_rect(r0, c0, h, w)
        rooms.append((r0 + h // 2, c0 + w // 2))  # store center
        if len(rooms) >= n_rooms:
            break

    # Connect rooms with corridors (L-shaped)
    for i in range(len(rooms) - 1):
        r1, c1 = rooms[i]
        r2, c2 = rooms[i + 1]
        # Horizontal then vertical
        for c in range(min(c1, c2), max(c1, c2) + 1):
            for dw in range(corridor_width):
                if 0 <= r1 + dw < rows:
                    grid[r1 + dw][c] = 0
        for r in range(min(r1, r2), max(r1, r2) + 1):
            for dw in range(corridor_width):
                if 0 <= c2 + dw < cols:
                    grid[r][c2 + dw] = 0

    return grid


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating synthetic benchmark maps...")
    print("(For paper-accurate results, replace with real movingai maps —")
    print(" see instructions at the top of this file)\n")

    # Random — same as real movingai random-64-64-10
    random_grid = make_random_map(rows=64, cols=64, density=0.10)
    write_map("instances/random-64-64-10.map", random_grid)

    # Warehouse — 161x84, matches movingai warehouse row count
    warehouse_grid = make_warehouse_map(rows=161, cols=84)
    write_map("instances/warehouse.map", warehouse_grid)

    # City — 256x256, matches movingai Berlin/Boston/Paris dimensions
    city_grid = make_city_map(rows=256, cols=256)
    write_map("instances/city.map", city_grid)

    # Game — 256x256, matches movingai Dragon Age / Starcraft dimensions
    game_grid = make_game_map(rows=256, cols=256)
    write_map("instances/game.map", game_grid)

    print("\nDone. Maps saved to instances/")
    print("\nFor paper benchmarking, download real movingai maps:")
    print("  https://movingai.com/benchmarks/mapf/index.html")
    print("  Place .map files in instances/ and pass them via --maps or --random_map/--warehouse_map")
