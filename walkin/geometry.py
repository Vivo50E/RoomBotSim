import numpy as np, base64, math
from scipy import ndimage
from recon import Camera, K_scaled

RES = 0.10


# ---------------------------------------------------------------- floor alignment

def floor_align(points, cameras):
    """Returns T (4x4): X_sim = T[:3,:3] @ X + T[:3,3]. Floor -> z=0, +Z up, origin at floor centroid."""
    rng = np.random.default_rng(0)
    P = points[rng.choice(len(points), min(len(points), 60000), replace=False)].astype(np.float64)
    up_prior = None
    if cameras:
        up_prior = -np.mean([c.R[:, 1] for c in cameras], 0)
        up_prior /= np.linalg.norm(up_prior)
    best_cnt, best_n, best_d = 0, None, None
    for _ in range(500):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a); nn = np.linalg.norm(n)
        if nn < 1e-6: continue
        n /= nn
        if up_prior is not None:
            if abs(n @ up_prior) < math.cos(math.radians(30)): continue
            if n @ up_prior < 0: n = -n
        d = -n @ a
        dist = P @ n + d
        if up_prior is None and (dist > 0.05).sum() < (dist < -0.05).sum():
            n, d, dist = -n, -d, -dist
        cnt = int((np.abs(dist) < 0.03).sum())
        if cnt > best_cnt: best_cnt, best_n, best_d = cnt, n, d
    if best_n is None:
        best_n, best_d = np.array([0, 0, 1.0]), -float(np.percentile(P[:, 2], 2))
    n, d = best_n, best_d
    z = np.array([0, 0, 1.0]); v = np.cross(n, z); s = np.linalg.norm(v); c = float(n @ z)
    if s < 1e-8:
        Rm = np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        Rm = np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)
    Pr = (Rm @ P.T).T
    floor_mask = np.abs(P @ n + d) < 0.03
    if not floor_mask.any(): floor_mask = Pr[:, 2] < np.percentile(Pr[:, 2], 5)
    floor_z = float(np.median(Pr[floor_mask][:, 2]))
    cx, cy = np.median(Pr[floor_mask][:, 0]), np.median(Pr[floor_mask][:, 1])
    T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = [-cx, -cy, -floor_z]
    return T


def apply_T(T, X):
    return (T[:3, :3] @ np.asarray(X, float).T).T + T[:3, 3]


def transform_camera(T, cam):
    return Camera(cam.photo, cam.width, cam.height, cam.K, T[:3, :3] @ cam.R, T[:3, :3] @ cam.t + T[:3, 3])


def camera_yaw(cam):
    f = cam.R[:, 2]
    return math.atan2(f[1], f[0])


# ---------------------------------------------------------------- depth unprojection

def unproject(u_norm, v_norm, cam, depth):
    if depth is None: return None
    H, W = depth.shape
    px = min(max(int(u_norm * W), 0), W - 1)
    py = min(max(int(v_norm * H), 0), H - 1)
    step = max(1, int(0.02 * H)); d = None
    for k in range(0, int(0.20 * H), step):
        y = max(py - k, 0)
        win = depth[max(0, y - 2):y + 3, max(0, px - 2):px + 3]
        win = win[win > 0]
        if len(win) >= 3:
            d = float(np.median(win)); py = y; break
    if d is None or d <= 0.1 or d > 30: return None
    K = K_scaled(cam.K, cam.width, cam.height, W, H)
    xc = (px - K[0, 2]) / K[0, 0] * d
    yc = (py - K[1, 2]) / K[1, 1] * d
    return cam.R @ np.array([xc, yc, d]) + cam.t


def foot_pixel(bbox):
    u0, v0, u1, v1 = bbox
    return (u0 + u1) / 2, v1 - 0.01


# ---------------------------------------------------------------- scale

DOOR_H = 2.03


def scale_from_door(door, cams_by_photo, depths):
    if not door: return 1.0
    cam = cams_by_photo.get(door["photo"]); dep = depths.get(door["photo"])
    if cam is None or dep is None: return 1.0
    u0, v0, u1, v1 = door["bbox"]; u = (u0 + u1) / 2
    top = unproject(u, v0 + 0.02, cam, dep); bot = unproject(u, v1 - 0.02, cam, dep)
    if top is None or bot is None: return 1.0
    h = abs(float(top[2] - bot[2]))
    if h < 0.8 or h > 5.0: return 1.0
    s = DOOR_H / h
    return s if abs(s - 1.0) > 0.20 else 1.0


def apply_scale(s, P, cams, depths, T):
    if s == 1.0: return P, cams, depths, T
    P = P * s
    cams = [Camera(c.photo, c.width, c.height, c.K, c.R, c.t * s) for c in cams]
    depths = {k: v * s for k, v in depths.items()}
    S = np.diag([s, s, s, 1.0])
    return P, cams, depths, S @ T


# ---------------------------------------------------------------- occupancy grid

def build_grid(P, people_xy=(), res=RES):
    lo = np.percentile(P[:, :2], 2, axis=0) - 0.5
    hi = np.percentile(P[:, :2], 98, axis=0) + 0.5
    nx = int(math.ceil((hi[0] - lo[0]) / res)); ny = int(math.ceil((hi[1] - lo[1]) / res))
    nx, ny = max(nx, 8), max(ny, 8)
    Q = P[(P[:, 2] > 0.25) & (P[:, 2] < 1.90)]
    i = np.floor((Q[:, 0] - lo[0]) / res).astype(int); j = np.floor((Q[:, 1] - lo[1]) / res).astype(int)
    ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    i, j, Q = i[ok], j[ok], Q[ok]
    count = np.zeros((ny, nx), np.int32); np.add.at(count, (j, i), 1)
    zmax = np.zeros((ny, nx), np.float32)
    if len(Q): np.maximum.at(zmax, (j, i), Q[:, 2])
    thresh = max(4, int(np.percentile(count[count > 0], 55)) if (count > 0).any() else 8)
    occ = count >= min(thresh, 8)
    nb = ndimage.convolve(occ.astype(np.int32), np.ones((3, 3), np.int32), mode="constant") - occ.astype(np.int32)
    occ &= nb >= 1
    occ = ndimage.binary_closing(occ, iterations=1)
    occ = carve_discs(occ, people_xy, 0.5, lo, res)
    free = ~occ
    lab, n = ndimage.label(free)
    if n > 0:
        sizes = ndimage.sum(free, lab, range(1, n + 1))
        walk = lab == (1 + int(np.argmax(sizes)))
    else:
        walk = free
    height_cm = np.clip(np.where(occ, zmax * 100.0, 0), 0, 250).astype(np.uint8)
    return dict(origin_xy=[float(lo[0]), float(lo[1])], size_cells=[nx, ny], occ=occ, walk=walk, height_cm=height_cm)


def _disc(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def _stamp(grid, centers_xy, radius_m, origin, res, value):
    out = grid.copy(); ny, nx = grid.shape; r = int(math.ceil(radius_m / res)); disc = _disc(r)
    for (cx, cy) in centers_xy:
        i = int(math.floor((cx - origin[0]) / res)); j = int(math.floor((cy - origin[1]) / res))
        j0, j1, i0, i1 = j - r, j + r + 1, i - r, i + r + 1
        dj0, di0 = max(0, -j0), max(0, -i0)
        j0, i0 = max(j0, 0), max(i0, 0); j1, i1 = min(j1, ny), min(i1, nx)
        if j1 <= j0 or i1 <= i0: continue
        sub = disc[dj0:dj0 + (j1 - j0), di0:di0 + (i1 - i0)]
        if value: out[j0:j1, i0:i1] |= sub
        else:     out[j0:j1, i0:i1] &= ~sub
    return out


def carve_discs(occ, centers_xy, radius_m, origin, res=RES): return _stamp(occ, centers_xy, radius_m, origin, res, False)
def stamp_discs(occ, centers_xy, radius_m, origin, res=RES): return _stamp(occ, centers_xy, radius_m, origin, res, True)


def grid_from_boxes(size_xy, boxes, people_xy=(), res=RES):
    W, H = size_xy
    lo = np.array([-W / 2 - 0.3, -H / 2 - 0.3]); nx = int(math.ceil((W + 0.6) / res)); ny = int(math.ceil((H + 0.6) / res))
    occ = np.zeros((ny, nx), bool); height_cm = np.zeros((ny, nx), np.uint8)
    bw = max(1, int(round(0.3 / res)))
    occ[:bw, :] = occ[-bw:, :] = True; occ[:, :bw] = occ[:, -bw:] = True; height_cm[occ] = 250
    for b in boxes:
        if b.get("height_m", 0) <= 0: continue
        cx, cy = b["center_xy"]; sx, sy = b["size_xy"]
        i0 = int(math.floor((cx - sx / 2 - lo[0]) / res)); i1 = int(math.ceil((cx + sx / 2 - lo[0]) / res))
        j0 = int(math.floor((cy - sy / 2 - lo[1]) / res)); j1 = int(math.ceil((cy + sy / 2 - lo[1]) / res))
        i0, j0 = max(i0, 0), max(j0, 0); i1, j1 = min(i1, nx), min(j1, ny)
        occ[j0:j1, i0:i1] = True; height_cm[j0:j1, i0:i1] = int(min(b["height_m"], 2.5) * 100)
    occ = carve_discs(occ, people_xy, 0.35, lo, res)
    free = ~occ; lab, n = ndimage.label(free)
    sizes = ndimage.sum(free, lab, range(1, n + 1)); walk = lab == (1 + int(np.argmax(sizes)))
    return dict(origin_xy=[float(lo[0]), float(lo[1])], size_cells=[nx, ny], occ=occ, walk=walk, height_cm=height_cm)


# ---------------------------------------------------------------- obstacles, walls, landmarks

def rect_decompose(occ, origin, res, height_cm, cap=400):
    ny, nx = occ.shape; rects = []; open_runs = {}
    for j in range(ny + 1):
        row = occ[j] if j < ny else np.zeros(nx, bool)
        cur = set(); i = 0
        while i < nx:
            if row[i]:
                i0 = i
                while i < nx and row[i]: i += 1
                cur.add((i0, i))
            else:
                i += 1
        for key in list(open_runs):
            if key not in cur:
                rects.append((key[0], key[1], open_runs.pop(key), j))
        for key in cur:
            open_runs.setdefault(key, j)
    out = []
    for (i0, i1, j0, j1) in rects:
        h = float(np.percentile(height_cm[j0:j1, i0:i1], 90)) / 100.0
        out.append(dict(center_xy=[origin[0] + (i0 + i1) / 2 * res, origin[1] + (j0 + j1) / 2 * res],
                        size_xy=[(i1 - i0) * res, (j1 - j0) * res], height_m=float(min(max(h, 0.3), 2.5))))
    if len(out) > cap:
        out.sort(key=lambda r: -r["size_xy"][0] * r["size_xy"][1]); out = out[:cap]
    for k, r in enumerate(out): r["id"] = f"obs_{k}"
    return out


def make_walls(origin, size_cells, res, thickness=0.10, height=2.5):
    ox, oy = origin; W = size_cells[0] * res; H = size_cells[1] * res
    return [dict(id="wall_s", center_xy=[ox + W / 2, oy], size_xy=[W, thickness], height_m=height),
            dict(id="wall_n", center_xy=[ox + W / 2, oy + H], size_xy=[W, thickness], height_m=height),
            dict(id="wall_w", center_xy=[ox, oy + H / 2], size_xy=[thickness, H], height_m=height),
            dict(id="wall_e", center_xy=[ox + W, oy + H / 2], size_xy=[thickness, H], height_m=height)]


def extract_landmarks(occ, origin, res, size_cells, min_area_m2=0.15, cap=10):
    lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
    nx, ny = size_cells; out = []
    for k in range(1, n + 1):
        js, is_ = np.where(lab == k)
        area = len(js) * res * res
        if area < min_area_m2: continue
        i0, i1, j0, j1 = is_.min(), is_.max() + 1, js.min(), js.max() + 1
        if (i1 - i0) > 0.6 * nx or (j1 - j0) > 0.6 * ny: continue
        out.append(dict(center_xy=[origin[0] + (i0 + i1) / 2 * res, origin[1] + (j0 + j1) / 2 * res],
                        size_xy=[(i1 - i0) * res, (j1 - j0) * res], _area=area))
    out.sort(key=lambda d: -d["_area"]); out = out[:cap]
    for d in out: d.pop("_area"); d["label"] = "unknown"
    return out


def nearest_walkable(walk, origin, x, y, res=RES):
    from collections import deque
    ny, nx = walk.shape
    i = min(max(int(math.floor((x - origin[0]) / res)), 0), nx - 1)
    j = min(max(int(math.floor((y - origin[1]) / res)), 0), ny - 1)
    if walk[j, i]: return origin[0] + (i + 0.5) * res, origin[1] + (j + 0.5) * res
    q = deque([(i, j)]); seen = {(i, j)}
    while q:
        ci, cj = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = ci + di, cj + dj
            if 0 <= ni < nx and 0 <= nj < ny and (ni, nj) not in seen:
                if walk[nj, ni]: return origin[0] + (ni + 0.5) * res, origin[1] + (nj + 0.5) * res
                seen.add((ni, nj)); q.append((ni, nj))
    return x, y


def place_fixture(label, bbox, cam, depth, walk, origin, room_center_xy):
    if depth is None: return None
    if label == "door":
        u, v = foot_pixel(bbox); size = [0.9, 0.3]
    else:
        u, v = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2; size = [1.5, 0.2]
    X = unproject(u, v, cam, depth)
    if X is None: return None
    x, y = float(X[0]), float(X[1])
    d = np.array(room_center_xy) - np.array([x, y]); nrm = np.linalg.norm(d)
    if nrm > 1e-6:
        x, y = (np.array([x, y]) + d / nrm * 0.6).tolist()
    x, y = nearest_walkable(walk, origin, x, y)
    return dict(label=label, center_xy=[x, y], size_xy=size)


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).astype(np.uint8).tobytes()).decode()


# ---------------------------------------------------------------- fallback frame -> sim frame

def fit_fallback_to_grid(fb_size_xy, grid):
    """Marble gives no input-camera poses, so the VLM's room-frame layout has to be dropped into the
    reconstructed grid. Both frames are metric and floor-aligned, so only a rotation by a multiple of
    90 degrees plus a translation is needed: pick the rotation whose aspect ratio matches the grid.
    Returns f(x, y) -> (x, y) in sim coordinates."""
    W, H = float(fb_size_xy[0]), float(fb_size_xy[1])
    nx, ny = grid["size_cells"]; res = RES
    gw, gh = nx * res, ny * res
    ox, oy = grid["origin_xy"]; cx, cy = ox + gw / 2, oy + gh / 2
    best, bestk = None, 0
    for k in range(4):
        w, h = (W, H) if k % 2 == 0 else (H, W)
        err = abs(w / max(h, 1e-6) - gw / max(gh, 1e-6))
        if best is None or err < best:
            best, bestk = err, k
    th = bestk * math.pi / 2
    c, s = math.cos(th), math.sin(th)
    sx = gw / max(W if bestk % 2 == 0 else H, 1e-6)
    sy = gh / max(H if bestk % 2 == 0 else W, 1e-6)
    sc = min(max(min(sx, sy), 0.5), 2.0)

    def f(x, y):
        rx, ry = x * c - y * s, x * s + y * c
        return (cx + rx * sc, cy + ry * sc)
    return f
