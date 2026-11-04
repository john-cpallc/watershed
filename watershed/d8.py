"""D8 flow-direction constants (ESRI encoding)."""

D8_OFFSETS: dict[int, tuple[int, int]] = {
    64: (-1, 0),   # N
    128: (-1, 1),  # NE
    1: (0, 1),     # E
    2: (1, 1),     # SE
    4: (1, 0),     # S
    8: (1, -1),    # SW
    16: (0, -1),   # W
    32: (-1, -1),  # NW
}

DIRMAP = tuple(D8_OFFSETS.keys())

NEIGHBOR_OFFSETS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

DIRECTION_CODES = {
    (-1, 0): 64,
    (-1, 1): 128,
    (0, 1): 1,
    (1, 1): 2,
    (1, 0): 4,
    (1, -1): 8,
    (0, -1): 16,
    (-1, -1): 32,
}
