"""
Generate MAPF Benchmark Maps
=============================
Creates movingai-format .map files matching the paper's training maps:
  - Random-64-64-10: 64x64 grid with ~10% random obstacles
  - Warehouse: structured shelves with corridors

Run once before training:
    python generate_maps.py
"""

import os
import random
import numpy

random.seed(0)
numpy.random.seed(0)


def write_map(filename, grid):
    rows = len(grid)
    cols = len(grid[0])
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write("type octile\n")
        f.write(f"height {rows}\n")
        f.write(f"width {cols}\n")
        f.write("map\n")
        for row in grid:
            f.write("".join("@" if c else "." for c in row) + "\n")
    print(f"  Saved: {filename}  ({rows}x{cols}, "
          f"obstacles: {sum(c for row in grid for c in row)})")


def make_random_map(size=64, density=0.10):
    """
    Random map with ~density% obstacles.
    Matches: Random-64-64-10 (64x64, 10% obstacles)
    """
    grid = [[0] * size for _ in range(size)]
    positions = [(r, c) for r in range(size) for c in range(size)]
    random.shuffle(positions)
    nr_obstacles = int(size * size * density)
    for r, c in positions[:nr_obstacles]:
        # Only place obstacle if at least one neighbor stays free (avoids blocking)
        grid[r][c] = 1
        if not _has_free_neighbor(grid, r, c, size):
            grid[r][c] = 0
    return grid


def _has_free_neighbor(grid, r, c, size):
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size and not grid[nr][nc]:
            return True
    return False


def make_warehouse_map():
    """
    Warehouse map: shelves separated by corridors.
    Similar structure to standard MAPF warehouse benchmark.

    Layout:
      - Shelves (obstacles) in rows, grouped in blocks
      - Vertical corridors every 5 columns
      - Horizontal corridors every 3 rows
      - Free border for agent movement
    """
    rows, cols = 33, 57
    grid = [[0] * cols for _ in range(rows)]

    for r in range(2, rows - 1, 3):       # shelf rows every 3
        for c in range(1, cols - 1):
            if c % 5 != 0:               # leave vertical corridor every 5 cols
                grid[r][c] = 1

    return grid, rows, cols


if __name__ == "__main__":
    print("Generating benchmark maps...")

    random_grid = make_random_map(size=64, density=0.10)
    write_map("instances/random-64-64-10.map", random_grid)

    warehouse_grid, wr, wc = make_warehouse_map()
    write_map("instances/warehouse.map", warehouse_grid)

    print("\nDone. Maps saved to instances/")
    print("Now run: python train_parcel.py --nr_agents 16 --epochs 2000")