"""Independent checks shared by the tests."""

from collections import deque

import delivery as fd
from delivery import world


def bfs_length(start, goal, walls):
    """Plain breadth-first search: an independent shortest-path check."""
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        cell, d = queue.popleft()
        if cell == goal:
            return d
        for nxt in fd.neighbours(cell, walls):
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, d + 1))
    return None


def valid_path(path, walls):
    return (path[0] == world.START and path[-1] == world.GOAL
            and all(c not in walls for c in path)
            and all(fd.manhattan(a, b) == 1 for a, b in zip(path, path[1:])))
