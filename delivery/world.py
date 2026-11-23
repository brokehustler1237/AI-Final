"""The city maps and the map that is currently in use.
"""

# Maps.  S = restaurant, C = customer, # = closed cell (construction or a
# building), = = river, . = open road.  "bridge" is the cell the traffic
# alert is about; it lies on the shortest route.
#
# 6x6 is the map from the project brief (3 blocked cells) and the default.
# The two larger maps are extra scenarios with a river and several bridges.
MAPS = {
    "6x6": {
        "rows": [
            "...#..",
            "..#..C",
            "......",
            "S.#...",
            "......",
            "......",
        ],
        "bridge": (2, 2),
    },
    "10x10": {
        "rows": [
            "....=.....",
            ".##....##C",
            "....=.....",
            "..#.=.##..",
            "..#.=.....",
            "....=..#..",
            "S...=..#..",
            ".##.=.....",
            ".........#",
            "....=.....",
        ],
        "bridge": (1, 4),
    },
    "14x14": {
        "rows": [
            "......=.......",
            ".###..=..##...",
            ".#............",
            ".#....=....##.",
            "......=......C",
            "..##..=..#....",
            "......=..#....",
            "..##.....###..",
            "......=.......",
            "S##...=...##..",
            "......=.......",
            "..#...=..#....",
            "..#......#..#.",
            "......=.......",
        ],
        "bridge": (7, 6),
    },
}


def use_map(name):
    """Switch every global the algorithms read to another map."""
    global MAP_NAME, MAP, ROWS, COLS, START, GOAL, WALLS, WATER, ALERT_CELL
    MAP_NAME = name
    MAP = MAPS[name]["rows"]
    ROWS, COLS = len(MAP), len(MAP[0])
    cells = [(r, c) for r in range(ROWS) for c in range(COLS)]
    START = next(p for p in cells if MAP[p[0]][p[1]] == "S")
    GOAL = next(p for p in cells if MAP[p[0]][p[1]] == "C")
    WATER = {p for p in cells if MAP[p[0]][p[1]] == "="}
    WALLS = {p for p in cells if MAP[p[0]][p[1]] in "#="}
    ALERT_CELL = MAPS[name]["bridge"]


# Current map. use_map() rewrites these; every other module reads them as
# world.START, world.WALLS, ... so a map switch is seen everywhere at once.
MAP_NAME = MAP = None
ROWS = COLS = 0
START = GOAL = ALERT_CELL = None
WALLS = WATER = frozenset()

use_map("6x6")
