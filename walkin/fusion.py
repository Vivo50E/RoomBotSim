"""Cross-compare several Marble reconstructions of one room and repair the splat/mesh geometry.

Marble returns a beautiful scene per generation, but single-view worlds hallucinate geometry behind
what the camera saw, and every world carries floaters. Running one multi-image world plus one world
per photo gives independent estimates of the same room; where they agree the geometry is real.

Pipeline
  1 normalize()      Marble axes -> +Z up, metric scale, floor at z = 0, origin at the floor centroid.
  2 register()       each secondary cloud is aligned to the primary. Floor alignment already fixed
                     roll, pitch and z, so only yaw and xy remain: 72 yaw hypotheses, each scored by
                     FFT cross-correlation of top-down density images.
  3 fuse()           5 cm voxel vote across the aligned clouds. Voxels seen by < min_support scenes are
                     dropped (floaters and hallucinations); voxels the primary missed but the others
                     agree on are added back (holes behind furniture).
  4 mesh_filter()    optional second opinion from Marble's collider mesh (GLB), which is built by a
                     different process than the splats, so agreement is real evidence.
"""
import json, math, struct, logging
import numpy as np
from scipy import ndimage

log = logging.getLogger("fusion")

MARBLE_TO_ZUP = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])


# ------------------------------------------------------------------ 1. normalize

def ground_plane(P, rng=None, iters=400):
    """RANSAC the dominant near-horizontal plane. Returns (normal, d) with the room above it."""
    rng = rng or np.random.default_rng(0)
    Q = P[rng.choice(len(P), min(len(P), 60000), replace=False)]
    up = np.array([0, 0, 1.0])
    best = (0, up, -float(np.percentile(Q[:, 2], 2)))
    for _ in range(iters):
        a, b, c = Q[rng.choice(len(Q), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n = n / nn
        if abs(n @ up) < math.cos(math.radians(30)):
            continue
        if n @ up < 0:
            n = -n
        d = -n @ a
        cnt = int((np.abs(Q @ n + d) < 0.03).sum())
        if cnt > best[0]:
            best = (cnt, n, d)
    return best[1], best[2]


def normalize(points, scale=1.0, ground_offset=0.0, marble_axes=True):
    """Marble cloud -> sim-style frame. Returns (P, T) with T the 4x4 taking input to output."""
    P = np.asarray(points, np.float64)
    T = np.eye(4)
    if marble_axes:
        P = P @ MARBLE_TO_ZUP.T
        T[:3, :3] = MARBLE_TO_ZUP @ T[:3, :3]
    if scale and abs(scale - 1.0) > 1e-6:
        P = P * scale
        T = np.diag([scale, scale, scale, 1.0]) @ T
    if ground_offset:
        P = P - np.array([0, 0, ground_offset])
        S = np.eye(4); S[2, 3] = -ground_offset
        T = S @ T
    n, d = ground_plane(P)
    z = np.array([0, 0, 1.0])
    v = np.cross(n, z); s = np.linalg.norm(v); c = float(n @ z)
    if s < 1e-8:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)
    P = P @ R.T
    floor = np.abs((P @ R.T.T)[:, 2]) < 1e9  # placeholder, recomputed below
    zf = P[:, 2]
    fl = zf < (np.percentile(zf, 2) + 0.06)
    floor_z = float(np.median(zf[fl])) if fl.any() else float(np.percentile(zf, 2))
    cx = float(np.median(P[fl, 0])) if fl.any() else float(np.median(P[:, 0]))
    cy = float(np.median(P[fl, 1])) if fl.any() else float(np.median(P[:, 1]))
    off = np.array([-cx, -cy, -floor_z])
    P = P + off
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = off
    return P.astype(np.float32), M @ T


# ------------------------------------------------------------------ 2. register

def density_image(P, res, lo, size, zlo=0.25, zhi=1.9):
    """Top-down occupancy density of the wall/furniture band."""
    nx, ny = size
    Q = P[(P[:, 2] > zlo) & (P[:, 2] < zhi)]
    if len(Q) == 0:
        Q = P
    i = np.floor((Q[:, 0] - lo[0]) / res).astype(int)
    j = np.floor((Q[:, 1] - lo[1]) / res).astype(int)
    ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    img = np.zeros((ny, nx), np.float32)
    np.add.at(img, (j[ok], i[ok]), 1.0)
    m = img.max()
    return img / m if m > 0 else img


def _grid_bounds(clouds, res, pad=0.2):
    lo = np.min([c.min(0) for c in clouds], 0) - pad
    hi = np.max([c.max(0) for c in clouds], 0) + pad
    size = np.ceil((hi - lo) / res).astype(int) + 1
    return lo, tuple(int(s) for s in size)


def _occ3(P, lo, size, res, dilate=1):
    g = np.zeros(size, bool)
    idx = np.floor((np.asarray(P, np.float64) - lo) / res).astype(int)
    ok = np.all((idx >= 0) & (idx < np.array(size)), 1)
    idx = idx[ok]
    g[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    if dilate:
        g = ndimage.binary_dilation(g, iterations=dilate)
    return g


def iou(A, B, lo, size, res):
    a, b = _occ3(A, lo, size, res, 0), _occ3(B, lo, size, res, 1)
    inter = int((a & b).sum())
    return inter / (int(a.sum()) + 1e-9)


def register(primary, secondary, res=0.10, yaw_steps=72, extent=12.0, topk=4):
    """Yaw + xy that best maps `secondary` onto `primary`. Both must be floor-aligned (z up, floor 0).
    The FFT correlation proposes candidates; each is then verified by 3D voxel overlap, which rejects
    the plausible-looking 90 degree errors a top-down correlation alone accepts.
    Returns (T 4x4, score in [0,1])."""
    from scipy.signal import fftconvolve
    n = int(2 * extent / res)
    lo2 = (-extent, -extent)
    A = np.clip(density_image(primary, res, lo2, (n, n)) * 6.0, 0, 1)
    cands = []
    for k in range(yaw_steps):
        th = 2 * math.pi * k / yaw_steps
        c, s = math.cos(th), math.sin(th)
        Q = secondary.copy()
        Q[:, :2] = np.c_[Q[:, 0] * c - Q[:, 1] * s, Q[:, 0] * s + Q[:, 1] * c]
        B = np.clip(density_image(Q, res, lo2, (n, n)) * 6.0, 0, 1)
        corr = fftconvolve(A, B[::-1, ::-1], mode="same")
        idx = int(np.argmax(corr))
        jj, ii = divmod(idx, n)
        sc = float(corr[jj, ii]) / (float(np.sqrt((A ** 2).sum() * (B ** 2).sum())) + 1e-9)
        cands.append((sc, th, (ii - n // 2) * res, (jj - n // 2) * res))
    cands.sort(key=lambda c: -c[0])
    lo3, size3 = _grid_bounds([primary], 0.10)
    best = (-1.0, np.eye(4))
    for _, th, dx, dy in cands[:topk]:
        c, s = math.cos(th), math.sin(th)
        T = np.eye(4)
        T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        T[:3, 3] = [dx, dy, 0.0]
        v = iou(apply_T(T, secondary), primary, lo3, size3, 0.10)
        if v > best[0]:
            best = (v, T)
    return best[1], float(best[0])


def apply_T(T, P):
    return (np.asarray(P, np.float64) @ T[:3, :3].T + T[:3, 3]).astype(np.float32)


# ------------------------------------------------------------------ 3. consensus fuse

def fuse(clouds, res=0.05, min_support=2, coverage_res=0.20):
    """clouds[0] is the primary, already aligned. Returns (points, report).

    A primary point is culled only where the other scenes actually looked: it becomes eligible when at
    least `min_support` - 1 secondaries have geometry in its top-down column, and is then kept only if
    `min_support` scenes contain it. Voxels the secondaries agree on that the primary missed are added
    back as points, which fills holes behind furniture.
    """
    clouds = [np.asarray(c, np.float64) for c in clouds if c is not None and len(c)]
    if not clouds:
        return np.zeros((0, 3), np.float32), {"scenes": 0}
    P0 = clouds[0]
    if len(clouds) == 1 or min_support <= 1:
        return P0.astype(np.float32), {"scenes": len(clouds), "culled": 0, "added": 0,
                                       "kept": int(len(P0)), "note": "single scene, nothing to cross-check"}
    lo, size = _grid_bounds(clouds, res)
    if int(np.prod(size)) > 120_000_000:
        return P0.astype(np.float32), {"scenes": len(clouds), "note": "grid too large, fusion skipped"}
    occ = [_occ3(c, lo, size, res, dilate=1) for c in clouds]
    support = np.zeros(size, np.uint8)
    for g in occ:
        support += g.astype(np.uint8)
    cov = np.zeros(size[:2], np.uint8)
    for g in occ[1:]:
        cov += g.any(2).astype(np.uint8)
    k = int(max(1, round(coverage_res / res))) * 2 + 1
    cov = ndimage.grey_dilation(cov, size=k)

    idx = np.floor((P0 - lo) / res).astype(int)
    ok = np.all((idx >= 0) & (idx < np.array(size)), 1)
    keep = np.ones(len(P0), bool)
    ii = idx[ok]
    sup = support[ii[:, 0], ii[:, 1], ii[:, 2]]
    cvr = cov[ii[:, 0], ii[:, 1]]
    eligible = cvr >= (min_support - 1)
    keep[ok] = (~eligible) | (sup >= min_support)
    culled = int((~keep).sum())

    unanimous = max(min_support, len(clouds))          # holes are only filled where every scene agrees
    agreed = (support >= unanimous) & ~occ[0]
    added_pts = np.zeros((0, 3))
    if agreed.any():
        a = np.argwhere(agreed)
        if len(a) > 400000:
            a = a[np.random.default_rng(0).choice(len(a), 400000, replace=False)]
        added_pts = lo + (a + 0.5) * res
    P = np.vstack([P0[keep], added_pts]).astype(np.float32)
    rep = {"scenes": len(clouds), "voxel_m": res, "min_support": min_support,
           "culled": culled, "added": int(len(added_pts)), "kept": int(len(P))}
    log.info("fusion: %s", rep)
    return P, rep


# ------------------------------------------------------------------ 4. mesh second opinion

def glb_vertices(path, max_v=400000):
    """Minimal GLB reader: returns (N,3) float32 POSITION vertices from every mesh primitive."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"glTF":
        raise ValueError("not a GLB")
    n = len(data)
    off = 12
    js = bins = None
    while off + 8 <= n:
        clen, ctype = struct.unpack_from("<II", data, off)
        chunk = data[off + 8: off + 8 + clen]
        if ctype == 0x4E4F534A:
            js = json.loads(chunk.decode("utf-8"))
        elif ctype == 0x004E4942:
            bins = chunk
        off += 8 + clen + ((4 - clen % 4) % 4 if clen % 4 else 0)
    if js is None or bins is None:
        raise ValueError("GLB missing chunks")
    CT = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
    out = []
    for mesh in js.get("meshes", []):
        for prim in mesh.get("primitives", []):
            ai = prim.get("attributes", {}).get("POSITION")
            if ai is None:
                continue
            acc = js["accessors"][ai]
            if acc.get("type") != "VEC3":
                continue
            fmt, size = CT[acc["componentType"]]
            bv = js["bufferViews"][acc["bufferView"]]
            base = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
            stride = bv.get("byteStride") or size * 3
            cnt = acc["count"]
            if stride == size * 3:
                arr = np.frombuffer(bins, dtype=np.dtype(fmt), count=cnt * 3, offset=base).reshape(cnt, 3)
            else:
                arr = np.stack([np.frombuffer(bins, dtype=np.dtype(fmt), count=1, offset=base + i * stride + k * size)[0]
                                for i in range(cnt) for k in range(3)]).reshape(cnt, 3)
            out.append(np.asarray(arr, np.float32))
    if not out:
        raise ValueError("GLB has no POSITION data")
    V = np.vstack(out)
    if len(V) > max_v:
        V = V[np.random.default_rng(0).choice(len(V), max_v, replace=False)]
    return V


def mesh_filter(points, mesh_vertices, res=0.12, dilate=1):
    """Drop splat points with no collider-mesh geometry nearby. Returns (points, report)."""
    P = np.asarray(points, np.float64)
    V = np.asarray(mesh_vertices, np.float64)
    if len(V) < 100:
        return points, {"mesh": "too sparse, skipped"}
    lo = np.minimum(P.min(0), V.min(0)) - res
    size = np.ceil((np.maximum(P.max(0), V.max(0)) + res - lo) / res).astype(int) + 1
    if size.prod() > 60_000_000:
        return points, {"mesh": "grid too large, skipped"}
    grid = np.zeros(tuple(size), bool)
    vi = np.floor((V - lo) / res).astype(int)
    grid[vi[:, 0], vi[:, 1], vi[:, 2]] = True
    if dilate:
        grid = ndimage.binary_dilation(grid, iterations=dilate)
    pi = np.floor((P - lo) / res).astype(int)
    keep = grid[pi[:, 0], pi[:, 1], pi[:, 2]]
    keep |= P[:, 2] < 0.05
    rep = {"mesh_culled": int((~keep).sum()), "mesh_kept": int(keep.sum())}
    log.info("fusion: %s", rep)
    return np.asarray(points)[keep], rep


# ------------------------------------------------------------------ top level

def cross_compare(scenes, load_points, min_support=2, voxel_m=0.05, use_mesh=True):
    """scenes: [{"ply","scale","ground_offset","mesh"?}], first is the primary.
    load_points: path -> (N,3). Returns (points, T_primary, report)."""
    report = {"scenes": len(scenes), "registered": [], "warnings": []}
    if not scenes:
        raise RuntimeError("no scenes")
    P0, T0 = normalize(load_points(scenes[0]["ply"]), scenes[0].get("scale", 1.0), scenes[0].get("ground_offset", 0.0))
    clouds = [P0]
    for s in scenes[1:]:
        try:
            Pk, _ = normalize(load_points(s["ply"]), s.get("scale", 1.0), s.get("ground_offset", 0.0))
            T, score = register(P0, Pk)
            report["registered"].append({"world_id": s.get("world_id"), "overlap": round(score, 3)})
            if score < 0.30:
                report["warnings"].append(f"{s.get('world_id')} did not register (score {score:.2f}), ignored")
                continue
            clouds.append(apply_T(T, Pk))
        except Exception as e:
            report["warnings"].append(f"{s.get('world_id')}: {e}")
    P, frep = fuse(clouds, res=voxel_m, min_support=min_support)
    report["fuse"] = frep
    if use_mesh and scenes[0].get("mesh"):
        try:
            V = glb_vertices(scenes[0]["mesh"])
            Vn = apply_T(T0, V) if False else (np.asarray(V, np.float64) @ T0[:3, :3].T + T0[:3, 3])
            P, mrep = mesh_filter(P, Vn)
            report["mesh"] = mrep
        except Exception as e:
            report["warnings"].append(f"mesh filter skipped: {e}")
    return np.asarray(P, np.float32), T0, report
