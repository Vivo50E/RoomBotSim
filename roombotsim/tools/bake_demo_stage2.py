#!/usr/bin/env python3
"""Stage 2 of the demo bake: repaired points + detected people -> world.json and people.json.

Runs offline against what stage 1 cached, so it is cheap to re-run while tuning the room.
Usage: python tools/bake_demo_stage2.py [job_id]
"""
import os, sys, json, math, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
for line in open(".env"):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())

import numpy as np
import geometry as G, perception
from geometry import RES, nearest_walkable, b64

JOB = sys.argv[1] if len(sys.argv) > 1 else "coffee"
COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4"]


def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    d = os.path.join("jobs", JOB); out = os.path.join(d, "marble")
    pts = np.load(os.path.join(out, "points.npy"))
    st1 = json.load(open(os.path.join(out, "stage1.json")))
    names = st1["names"]; people_raw = st1["people_raw"]; fixtures = st1["fixtures"]
    small = [os.path.join(d, f"{n}_1024.jpg") for n in names]
    if not people_raw:
        # stage 1 may have cached an empty cast (an early build mishandled pixel-space boxes).
        # A room with nobody in it is not a demo, so re-detect and write the result back.
        log("stage 1 cached no people, re-running detection")
        try:
            people_raw, fx2 = perception.detect_people(small, names)
            fixtures = fixtures or fx2
            st1["people_raw"], st1["fixtures"] = people_raw, fixtures
            with open(os.path.join(out, "stage1.json"), "w") as f:
                json.dump(st1, f, indent=1)
            log(f"re-detected {len(people_raw)} people, {len(fixtures)} fixtures")
        except Exception as e:
            log("re-detection failed:", str(e)[:200])
    log(f"{len(pts):,} points, {len(people_raw)} people, {len(fixtures)} fixtures")

    # Marble extrapolates floor and walls well past the real room; on this cafe it drew 21 x 24 m of
    # mostly empty space around ~7 x 9 m of actual furniture. Crop to the furniture core plus a margin,
    # capped near the operator's own estimate of the room, so the grid is the room and not the guess.
    band = pts[(pts[:, 2] > 0.25) & (pts[:, 2] < 1.9)]
    lo = np.percentile(band[:, :2], 4, axis=0); hi = np.percentile(band[:, :2], 96, axis=0)
    mid = (lo + hi) / 2; half = (hi - lo) / 2 + float(os.environ.get("BAKE_MARGIN", "2.2"))
    cap = np.array([float(v) for v in os.environ.get("BAKE_MAX_ROOM", "15x12").split("x")]) / 2
    half = np.minimum(half, cap)
    keep = np.all(np.abs(pts[:, :2] - mid) <= half, axis=1)
    log(f"cropping {len(pts):,} -> {keep.sum():,} points to a {2*half[0]:.1f} x {2*half[1]:.1f} m room")
    pts = pts[keep]
    grid = G.build_grid(pts)
    nx, ny = grid["size_cells"]; ox, oy = grid["origin_xy"]
    room_center = (ox + nx * RES / 2, oy + ny * RES / 2)
    log(f"grid {nx}x{ny} = {nx*RES:.1f} x {ny*RES:.1f} m, walkable {100*grid['walk'].mean():.0f}%")

    landmarks = G.extract_landmarks(grid["occ"], grid["origin_xy"], RES, grid["size_cells"])
    log(f"{len(landmarks)} landmark candidates")
    topdown = os.path.join(d, "topdown.png")
    render_topdown(grid, landmarks, topdown)
    landmarks = perception.label_landmarks(topdown, small, names, landmarks)
    log("labels:", [l["label"] for l in landmarks])

    # fixtures (door, whiteboard, tv) have no depth here, so drop them onto the nearest walkable
    # cell in the direction the grid is open; better than omitting a door the robot enters from
    for f in fixtures:
        if any(l["label"].startswith(f["label"]) for l in landmarks):
            continue
        u = (f["bbox"][0] + f["bbox"][2]) / 2
        ang = 2 * math.pi * (names.index(f["photo"]) + u) / max(len(names), 1)
        r = 0.42 * min(nx, ny) * RES
        p = (room_center[0] + r * math.cos(ang), room_center[1] + r * math.sin(ang))
        p = nearest_walkable(grid["walk"], grid["origin_xy"], *p)
        landmarks.append(dict(label=f["label"], center_xy=list(p),
                              size_xy=[0.9, 0.3] if f["label"] == "door" else [1.5, 0.2]))
        log(f'placed fixture {f["label"]} at {p[0]:.1f}, {p[1]:.1f}')

    landmarks = landmarks[:12]
    seen = {}
    for k, lm in enumerate(landmarks):
        lm["id"] = f"lm_{k}"
        c = seen.get(lm["label"], 0) + 1; seen[lm["label"]] = c
        if c > 1: lm["label"] = f'{lm["label"]} {c}'

    # people: there are no camera poses to unproject from, so place them the way the room reads:
    # seated people take stools along the long side of the main table, facing it; standing people go to
    # the landmark their own description mentions (entrance, kitchen, couch), else open floor nearby.
    counter = next((l for l in landmarks if l["label"].startswith(("counter", "cabinet"))), None)
    table = next((l for l in landmarks if l["label"].startswith(("table", "desk"))), None)
    door = next((l for l in landmarks if l["label"].startswith("door")), None)
    couch = next((l for l in landmarks if l["label"].startswith("couch")), None)
    anchor = table or counter
    def near(lm, k, n, dist=0.55):
        """k-th of n spots just outside a landmark's long edge, facing it."""
        cx, cy = lm["center_xy"]; sx, sy = lm["size_xy"]
        along = np.array([1.0, 0.0]) if sx >= sy else np.array([0.0, 1.0]); out = np.array([-along[1], along[0]])
        L = max(sx, sy); off = (k - (n - 1) / 2) * min(0.8, L / max(n, 1))
        side = 1 if k % 2 == 0 else -1
        p = np.array([cx, cy]) + along * off + out * side * (min(sx, sy) / 2 + dist)
        return nearest_walkable(grid["walk"], grid["origin_xy"], float(p[0]), float(p[1]))
    seated = [p for p in people_raw if p["posture"] == "seated"]
    standing = [p for p in people_raw if p["posture"] != "seated"]
    people = []
    for k, p in enumerate(seated):
        x, y = near(anchor, k, max(len(seated), 1), 0.45) if anchor else (room_center[0] + k, room_center[1])
        fx, fy = anchor["center_xy"] if anchor else room_center
        people.append((p, x, y, fx, fy))
    for k, p in enumerate(standing):
        desc = p["description"].lower()
        lm = door if ("entrance" in desc or "door" in desc) and door else (counter if ("kitchen" in desc or "counter" in desc) and counter else (couch if "couch" in desc and couch else None))
        if lm is not None:
            x, y = near(lm, k, max(len(standing), 1), 0.9)
        else:
            ang = 2 * math.pi * k / max(len(standing), 1) + 0.9
            x, y = nearest_walkable(grid["walk"], grid["origin_xy"], room_center[0] + 2.2 * math.cos(ang), room_center[1] + 2.2 * math.sin(ang))
        fx, fy = (lm["center_xy"] if lm is not None else room_center)
        people.append((p, x, y, fx, fy))
    people = [dict(id=p["id"], description=p["description"], posture=p["posture"], activity=p["activity"],
                   talking_to=p["talking_to"], pos_xy=[x, y], home_xy=[x, y],
                   facing_deg=math.degrees(math.atan2(fy - y, fx - x)), personality=p["personality"],
                   color=COLORS[(p["id"] - 1) % len(COLORS)], boxes=p.get("boxes", {}))
              for (p, x, y, fx, fy) in sorted(people, key=lambda t: t[0]["id"])]
    # A judge should see the room alive within seconds. Detected people skew seated and idle, and the
    # sampler leaves seated people waiting, so promote a couple of standing people to walking. They then
    # pick landmarks to wander between and the room reads as inhabited rather than staged.
    movers = [q for q in people if q["posture"] == "standing"]
    if len(movers) < 2:
        for q in people:
            if q["posture"] == "seated" and len(movers) < 2:
                q["posture"] = "standing"; movers.append(q)
    for q in movers[:2]:
        q["activity"] = "walking"; q["talking_to"] = None
    chatty = [q for q in people if q not in movers[:2]]
    if len(chatty) >= 2:                       # a pair facing each other reads better than two idlers
        chatty[0]["activity"] = chatty[1]["activity"] = "talking"
        chatty[0]["talking_to"] = chatty[1]["id"]; chatty[1]["talking_to"] = chatty[0]["id"]
    log(f"placed {len(people)} people: " + ", ".join(f'{q["id"]}:{q["posture"]}/{q["activity"]}' for q in people))

    world = dict(source="marble", resolution_m=RES, origin_xy=grid["origin_xy"], size_cells=grid["size_cells"],
                 occupancy_b64=b64(grid["occ"]), walkable_b64=b64(grid["walk"]), height_b64=b64(grid["height_cm"]),
                 obstacles=G.rect_decompose(grid["occ"], grid["origin_xy"], RES, grid["height_cm"]),
                 walls=G.make_walls(grid["origin_xy"], grid["size_cells"], RES), landmarks=landmarks,
                 cameras=[],
                 # T0 maps the raw splat into the sim frame; Spark renders PLY/SPZ in raw coordinates, so
                 # the viewer uses T0 directly. (An overhead screenshot is a bad way to check this: seeing
                 # floor tiles from above means the camera is under the floor. Check at eye height.)
                 T_atlas_to_sim=st1.get("T0", np.eye(4).tolist()),
                 splat_url=(f'/jobs/{JOB}/marble/primary_clean.ply' if os.path.exists(os.path.join(out, "primary_clean.ply"))
                            else (f'/jobs/{JOB}/marble/{st1["splat"]}' if st1.get("splat") else None)),
                 photo_a_url=f"/jobs/{JOB}/{names[0]}_1024.jpg",
                 photo_urls=[f"/jobs/{JOB}/{n}_1024.jpg" for n in names])
    with open(os.path.join(d, "world.json"), "w") as f: json.dump(world, f)
    with open(os.path.join(d, "people.json"), "w") as f: json.dump(dict(people=people), f)
    log(f'wrote {d}/world.json  ({len(world["obstacles"])} obstacles, {len(landmarks)} landmarks)')

    import sim as simmod
    sp = simmod.spawn_objects(world)
    log("coffee spawns:", {k: [round(v, 2) for v in x] for k, x in sp.items()})
    from planning import inflate, plan_xy
    b = inflate(np.frombuffer(__import__("base64").b64decode(world["occupancy_b64"]), np.uint8)
                .reshape(ny, nx).astype(bool), 0.30)
    if counter and landmarks:
        p = plan_xy(b, world["origin_xy"], landmarks[0]["center_xy"], counter["center_xy"])
        log("robot can path landmark 0 -> counter:", bool(p))
    log("STAGE 2 COMPLETE")


def render_topdown(grid, landmarks, path, px_per_m=50):
    from PIL import Image, ImageDraw
    ox, oy = grid["origin_xy"]; nx, ny = grid["size_cells"]
    W, H = int(nx * RES * px_per_m), int(ny * RES * px_per_m)
    img = Image.new("RGB", (max(W, 32), max(H, 32)), (245, 245, 245)); dr = ImageDraw.Draw(img)
    to = lambda x, y: (int((x - ox) * px_per_m), int(H - (y - oy) * px_per_m))
    js, is_ = np.where(grid["occ"])
    for j, i in zip(js, is_):
        x0, y0 = to(ox + i * RES, oy + (j + 1) * RES); x1, y1 = to(ox + (i + 1) * RES, oy + j * RES)
        dr.rectangle([x0, y0, x1, y1], fill=(70, 70, 80))
    for k, lm in enumerate(landmarks):
        cx, cy = lm["center_xy"]; sx, sy = lm["size_xy"]
        x0, y0 = to(cx - sx / 2, cy + sy / 2); x1, y1 = to(cx + sx / 2, cy - sy / 2)
        dr.rectangle([x0, y0, x1, y1], outline=(220, 40, 40), width=3)
        dr.text(((x0 + x1) / 2 - 6, (y0 + y1) / 2 - 8), str(k), fill=(220, 40, 40))
    img.save(path); return path


if __name__ == "__main__":
    main()
