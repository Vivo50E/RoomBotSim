import heapq, math, numpy as np
from collections import deque
from scipy import ndimage

RES = 0.10
NBRS = [(1,0,1.0),(-1,0,1.0),(0,1,1.0),(0,-1,1.0),(1,1,1.4142),(1,-1,1.4142),(-1,1,1.4142),(-1,-1,1.4142)]

def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi

def inflate(occ, radius_m, res=RES):
    r = int(math.ceil(radius_m / res))
    if r <= 0: return occ.copy()
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return ndimage.binary_dilation(occ, structure=(x * x + y * y) <= r * r)

def world_to_cell(x, y, origin, res=RES): return int(math.floor((x - origin[0]) / res)), int(math.floor((y - origin[1]) / res))
def cell_to_world(i, j, origin, res=RES): return origin[0] + (i + 0.5) * res, origin[1] + (j + 0.5) * res

def nearest_free(blocked, i, j):
    ny, nx = blocked.shape
    i, j = min(max(i, 0), nx - 1), min(max(j, 0), ny - 1)
    if not blocked[j, i]: return i, j
    q = deque([(i, j)]); seen = {(i, j)}
    while q:
        ci, cj = q.popleft()
        for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
            ni, nj = ci + di, cj + dj
            if 0 <= ni < nx and 0 <= nj < ny and (ni, nj) not in seen:
                if not blocked[nj, ni]: return ni, nj
                seen.add((ni, nj)); q.append((ni, nj))
    return i, j

def astar(blocked, start, goal, max_expansions=300000):
    ny, nx = blocked.shape
    start = nearest_free(blocked, *start); goal = nearest_free(blocked, *goal)
    def h(a):
        dx, dy = abs(a[0] - goal[0]), abs(a[1] - goal[1]); return max(dx, dy) + 0.4142 * min(dx, dy)
    g = {start: 0.0}; parent = {}; pq = [(h(start), start)]; closed = set()
    while pq and max_expansions > 0:
        max_expansions -= 1
        _, cur = heapq.heappop(pq)
        if cur == goal:
            path = [cur]
            while cur in parent: cur = parent[cur]; path.append(cur)
            return path[::-1]
        if cur in closed: continue
        closed.add(cur)
        for di, dj, c in NBRS:
            ni, nj = cur[0] + di, cur[1] + dj
            if not (0 <= ni < nx and 0 <= nj < ny) or blocked[nj, ni]: continue
            if di and dj and (blocked[cur[1], ni] or blocked[nj, cur[0]]): continue
            ng = g[cur] + c
            if ng < g.get((ni, nj), 1e18):
                g[(ni, nj)] = ng; parent[(ni, nj)] = cur
                heapq.heappush(pq, (ng + h((ni, nj)), (ni, nj)))
    return None

def los(blocked, a, b):
    (x0, y0), (x1, y1) = a, b
    dx, dy = abs(x1 - x0), abs(y1 - y0); sx = 1 if x1 > x0 else -1; sy = 1 if y1 > y0 else -1; err = dx - dy
    while True:
        if blocked[y0, x0]: return False
        if (x0, y0) == (x1, y1): return True
        e2 = 2 * err
        if e2 > -dy: err -= dy; x0 += sx
        if e2 < dx:  err += dx; y0 += sy

def simplify(blocked, path):
    if not path: return path
    out = [path[0]]; k = 0
    while k < len(path) - 1:
        m = len(path) - 1
        while m > k + 1 and not los(blocked, path[k], path[m]): m -= 1
        out.append(path[m]); k = m
    return out

def reach_grid(blocked, margin_m=0.10, res=RES):
    free = ~inflate(blocked, margin_m, res)
    lab, n = ndimage.label(free)
    if n == 0: return blocked.copy()
    sizes = ndimage.sum(free, lab, range(1, n + 1))
    return ~(lab == 1 + int(np.argmax(sizes)))

def blocked_at(grid, origin, x, y, res=RES):
    ny, nx = grid.shape; i, j = world_to_cell(x, y, origin, res)
    return not (0 <= i < nx and 0 <= j < ny) or bool(grid[j, i])

def snap_free_xy(blocked, origin, x, y, res=RES):
    i, j = nearest_free(blocked, *world_to_cell(x, y, origin, res))
    return cell_to_world(i, j, origin, res)

def plan_xy(blocked, origin, start_xy, goal_xy, res=RES):
    s = world_to_cell(*start_xy, origin, res); gl = world_to_cell(*goal_xy, origin, res)
    p = astar(blocked, s, gl)
    if p is None: return None
    p = simplify(blocked, p)
    return [cell_to_world(i, j, origin, res) for (i, j) in p]

def pursuit_target(pos, path, lookahead):
    while len(path) > 1 and math.dist(pos, path[0]) < 0.25: path.pop(0)
    for p in path:
        if math.dist(pos, p) >= lookahead: return p
    return path[-1]

def social_force(pos, others, radius=0.9, k=2.0):
    fx = fy = 0.0
    for (ox, oy, orad) in others:
        dx, dy = pos[0] - ox, pos[1] - oy; d = math.hypot(dx, dy)
        if d < 1e-6: dx, dy, d = 0.01, 0.0, 0.01
        lim = radius + orad
        if d < lim:
            m = k * (lim - d) / d; fx += m * dx; fy += m * dy
    return fx, fy
