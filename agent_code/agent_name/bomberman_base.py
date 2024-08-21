"""
This file contains functinality connected to the bomberman game itself. This may include parts of the framework itself, that are, however, not exposed as part of the API.
E.g. building the arena, calculating blast coordinates, etc.
"""
from functools import cache

import numpy as np

import settings as s

# Copied from the framework and adapted
def build_arena_walls():
    WALL = -1
    arena = np.zeros((s.COLS, s.ROWS), int)

    arena[:1, :] = WALL
    arena[-1:, :] = WALL
    arena[:, :1] = WALL
    arena[:, -1:] = WALL

    for x in range(s.COLS):
        for y in range(s.ROWS):
            if (x + 1) * (y + 1) % 2 == 1:
                arena[x, y] = WALL

    return arena

arena = build_arena_walls()

# Copied from the framework and adapted
# Cache the function to avoid recalculating:
# - Relatively expensive
# - May be part of a transform e.g. (may be called up to 10^7 times in a run)
# - There are only about ~10^2 possible bomb locations (no need for lru cache)
@cache
def get_blast_coords(x, y):
    blast_coords = [(x, y)]
    power = s.BOMB_POWER

    for i in range(1, power + 1):
        if arena[x + i, y] == -1:
            break
        blast_coords.append((x + i, y))
    for i in range(1, power + 1):
        if arena[x - i, y] == -1:
            break
        blast_coords.append((x - i, y))
    for i in range(1, power + 1):
        if arena[x, y + i] == -1:
            break
        blast_coords.append((x, y + i))
    for i in range(1, power + 1):
        if arena[x, y - i] == -1:
            break
        blast_coords.append((x, y - i))

    return blast_coords