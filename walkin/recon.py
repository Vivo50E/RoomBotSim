"""Scene reconstruction adapter.

reconstruct(image_paths, job_dir) -> Recon, selected by WORLD_SOURCE:
  marble    World Labs Marble (the World API). Four chest-height photos go in as one multi-image world
            with reconstruct_images=true; optionally one single-image world per photo is generated too
            and all of them are cross-compared and repaired by fusion.cross_compare. Default.
  local     pre-computed artifacts in <job_dir>/recon/ (see load_local for the layout)
  mock      synthetic room from tools/room_6x8.json (tests, /test route)
  fallback  never calls reconstruct(); the pipeline uses the VLM room-layout estimate instead

Everything downstream reads only Recon.
"""
from dataclasses import dataclass
import numpy as np, os, json, time, logging

log = logging.getLogger("recon")


@dataclass
class Camera:
    photo: str
    width: int
    height: int
    K: np.ndarray
    R: np.ndarray      # camera->world, OpenCV axes
    t: np.ndarray      # camera position in world


@dataclass
class Recon:
    points: np.ndarray          # (N,3) float32, recon frame (not floor aligned)
    cameras: list               # [Camera] may be []
    depths: dict                # {"A": (H,W) float32 meters, 0=invalid}
    splat_path: str | None      # local path to a gaussian-splat PLY or None


MOCK_ROOM_TO_ATLAS = np.eye(4)


def reconstruct(image_paths, job_dir, status=None) -> Recon:
    src = os.environ.get("WORLD_SOURCE", "marble")
    if src == "marble":
        return _marble(image_paths, job_dir, status)
    if src == "mock":
        return _mock(job_dir)
    if src == "local":
        return load_local(os.path.join(job_dir, "recon"))
    raise RuntimeError(f"WORLD_SOURCE={src} does not use reconstruct()")


# ---------------------------------------------------------------- helpers

def load_gaussian_ply(path, max_points=2_000_000):
    """Reads a 3DGS-style PLY (or a plain xyz PLY). Keeps opaque, small gaussians. Returns (N,3) float32."""
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]
    names = v.data.dtype.names
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    keep = np.isfinite(xyz).all(1)
    if "opacity" in names:
        keep &= (1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], np.float64)))) > 0.5
    if "scale_0" in names:
        s = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)).max(1)
        keep &= s < 0.5
    xyz = xyz[keep]
    if len(xyz) > max_points:
        xyz = xyz[np.random.default_rng(0).choice(len(xyz), max_points, replace=False)]
    return xyz


def to_cam_to_world(R, t, is_world_to_cam):
    R = np.asarray(R, float); t = np.asarray(t, float).reshape(3)
    if is_world_to_cam:
        return R.T, -R.T @ t
    return R, t


def gl_to_cv(R, t):
    F = np.diag([1.0, -1.0, -1.0])
    return np.asarray(R) @ F, np.asarray(t)


def K_for(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def K_scaled(K, w0, h0, w1, h1):
    S = np.diag([w1 / w0, h1 / h0, 1.0])
    return S @ K


def depth_png16_to_m(path, scale=0.001):
    from PIL import Image
    return np.asarray(Image.open(path), dtype=np.float32) * scale


# ---------------------------------------------------------------- local artifacts

def load_local(d):
    """Layout of <job>/recon/:
         points.ply | points.npy       required (any xyz PLY, 3DGS PLY, or (N,3) npy)
         scene.ply                     optional gaussian splat for the viewer (defaults to points.ply if 3DGS)
         cameras.json                  optional: [{"photo":"A","width":W,"height":H,"K":[[..]],"R":[[..]],"t":[..],
                                                  "world_to_cam":false,"opengl":false}]
         depth_A.npy / depth_A.png     optional: meters (npy float32) or png16 millimetres
    """
    if not os.path.isdir(d):
        raise RuntimeError(f"no local recon dir {d}")
    if os.path.exists(os.path.join(d, "points.npy")):
        pts = np.load(os.path.join(d, "points.npy")).astype(np.float32)
    elif os.path.exists(os.path.join(d, "points.ply")):
        pts = load_gaussian_ply(os.path.join(d, "points.ply"))
    elif os.path.exists(os.path.join(d, "scene.ply")):
        pts = load_gaussian_ply(os.path.join(d, "scene.ply"))
    else:
        raise RuntimeError("recon dir needs points.ply, points.npy or scene.ply")
    cams = []
    cj = os.path.join(d, "cameras.json")
    if os.path.exists(cj):
        for c in json.load(open(cj)):
            R, t = to_cam_to_world(c["R"], c["t"], bool(c.get("world_to_cam", False)))
            if c.get("opengl", False):
                R, t = gl_to_cv(R, t)
            cams.append(Camera(c["photo"], int(c["width"]), int(c["height"]), np.asarray(c["K"], float), R, t))
    depths = {}
    for ph in ("A", "B"):
        if os.path.exists(os.path.join(d, f"depth_{ph}.npy")):
            depths[ph] = np.load(os.path.join(d, f"depth_{ph}.npy")).astype(np.float32)
        elif os.path.exists(os.path.join(d, f"depth_{ph}.png")):
            depths[ph] = depth_png16_to_m(os.path.join(d, f"depth_{ph}.png"))
    splat = os.path.join(d, "scene.ply") if os.path.exists(os.path.join(d, "scene.ply")) else None
    return Recon(pts, cams, depths, splat)


# ---------------------------------------------------------------- marble

def _marble(image_paths, job_dir, status=None):
    """World Labs Marble. Generates the room, cross-compares every reconstruction it got back, and
    returns the repaired point cloud plus the splat for the viewer."""
    import marble, fusion
    res = marble.reconstruct_room(image_paths, job_dir, status=status)
    scenes = [res["primary"]] + list(res["extras"])
    scenes = [s for s in scenes if s.get("ply")]
    if not scenes:
        raise RuntimeError("marble returned no splat file")
    if status is not None:
        status["message"] = f"cross-checking {len(scenes)} reconstructions"
    pts, T0, report = fusion.cross_compare(scenes, load_gaussian_ply,
                                           min_support=2 if len(scenes) > 1 else 1)
    json.dump(report, open(os.path.join(job_dir, "marble", "fusion.json"), "w"), indent=1)
    log.info("marble fusion report: %s", report)
    splat = res["primary"].get("spz") or res["primary"].get("ply")
    return Recon(np.asarray(pts, np.float32), [], {}, splat)


# ---------------------------------------------------------------- mock

def _mock(job_dir):
    global MOCK_ROOM_TO_ATLAS
    here = os.path.dirname(os.path.abspath(__file__))
    room = json.load(open(os.path.join(here, "tools", "room_6x8.json")))
    rng = np.random.default_rng(0)
    W, H = room["size_xy"]; pts = []
    n = 40000
    pts.append(np.c_[rng.uniform(-W/2, W/2, n), rng.uniform(-H/2, H/2, n), rng.normal(0, 0.01, n)])
    for b in room["boxes"]:
        cx, cy = b["center_xy"]; sx, sy = b["size_xy"]; h = b["height_m"]
        m = int(3000 * max(sx * sy, 0.2)) + 400
        pts.append(np.c_[rng.uniform(cx-sx/2, cx+sx/2, m), rng.uniform(cy-sy/2, cy+sy/2, m), h + rng.normal(0, 0.01, m)])
        pts.append(np.c_[rng.choice([cx-sx/2, cx+sx/2], m), rng.uniform(cy-sy/2, cy+sy/2, m), rng.uniform(0, h, m)])
        pts.append(np.c_[rng.uniform(cx-sx/2, cx+sx/2, m), rng.choice([cy-sy/2, cy+sy/2], m), rng.uniform(0, h, m)])
    m = 20000
    pts.append(np.c_[rng.choice([-W/2, W/2], m), rng.uniform(-H/2, H/2, m), rng.uniform(0, 2.5, m)])
    pts.append(np.c_[rng.uniform(-W/2, W/2, m), rng.choice([-H/2, H/2], m), rng.uniform(0, 2.5, m)])
    P = np.vstack(pts).astype(np.float32)
    a = 0.12
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]], np.float32)
    off = np.array([0.8, -0.4, 0.3], np.float32)
    P = (Rx @ P.T).T + off
    MOCK_ROOM_TO_ATLAS = np.eye(4); MOCK_ROOM_TO_ATLAS[:3, :3] = Rx; MOCK_ROOM_TO_ATLAS[:3, 3] = off
    def cam(photo, pos, yaw):
        f = np.array([np.cos(yaw), np.sin(yaw), 0.0]); d = np.array([0, 0, -1.0]); r = np.cross(d, f)
        R = np.stack([r, d, f], 1)
        return Camera(photo, 1024, 768, K_for(900, 900, 512, 384), Rx @ R, Rx @ np.array(pos) + off)
    cams = [cam("A", [0, -H/2 + 0.4, 1.4], np.pi/2), cam("B", [W/2 - 0.4, 0, 1.4], np.pi)]
    return Recon(P, cams, {}, None)
