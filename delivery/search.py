"""Algorithm 1: A* search on the city grid (and BFS as a baseline).
"""

import heapq
from collections import deque

from . import world


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def neighbours(cell, walls):
    r, c = cell
    for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):      # up, right, down, left
        nr, nc = r + dr, c + dc
        if 0 <= nr < world.ROWS and 0 <= nc < world.COLS and (nr, nc) not in walls:
            yield (nr, nc)

def astar(start, goal, walls):
    """A* with the Manhattan distance as heuristic.

    Every move costs 1 and Manhattan distance counts the fewest moves still
    needed, so h never overestimates. That makes it admissible and A*
    returns a shortest path.

    Returns (path, expansion_log). Each log entry is (cell, g, h, f).
    """
    counter = 0
    frontier = [(manhattan(start, goal), manhattan(start, goal), counter, start)]
    g = {start: 0}
    came_from = {}
    closed = set()
    log = []

    while frontier:
        f, h, _, cell = heapq.heappop(frontier)
        if cell in closed:
            continue
        closed.add(cell)
        log.append((cell, g[cell], h, f))

        if cell == goal:
            path = [cell]
            while cell in came_from:
                cell = came_from[cell]
                path.append(cell)
            return path[::-1], log

        for nxt in neighbours(cell, walls):
            new_g = g[cell] + 1
            if new_g < g.get(nxt, float("inf")):
                g[nxt] = new_g
                came_from[nxt] = cell
                h_n = manhattan(nxt, goal)
                counter += 1
                heapq.heappush(frontier, (new_g + h_n, h_n, counter, nxt))

    return None, log

def bfs(start, goal, walls):
    """Breadth-first search. Also finds a shortest route here (all moves cost
    1) but has no heuristic, so it is used to show what A*'s h saves."""
    queue = deque([start])
    came_from = {start: None}
    expanded = []
    while queue:
        cell = queue.popleft()
        expanded.append(cell)
        if cell == goal:
            path = [cell]
            while came_from[path[-1]] is not None:
                path.append(came_from[path[-1]])
            return path[::-1], expanded
        for nxt in neighbours(cell, walls):
            if nxt not in came_from:
                came_from[nxt] = cell
                queue.append(nxt)
    return None, expanded
