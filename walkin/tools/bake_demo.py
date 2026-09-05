#!/usr/bin/env python3
"""Bake a permanent demo room from a clockwise photo sweep.

Usage: python tools/bake_demo.py <src_dir> [job_id]

The photos are a clockwise walk of the room (corner, wall, corner, ...), so their azimuths are evenly
spaced over 360 degrees. Marble takes up to 4 images with explicit direction control and up to 8 in
auto-layout, so this tries azimuths first and falls back to auto-layout on a 422.

Everything lands in jobs/<job_id>/ as world.json + people.json, which the server loads instantly.
"""
import os, sys, json, glob, math, time, threading, traceback
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
for line in open(".env"):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())

import numpy as np
import marble, fusion, recon, geometry as G, perception
from PIL import Image, ImageOps

SRC = sys.argv[1] if len(sys.argv) > 1 else "jobs/coffee_src"
JOB = sys.argv[2] if len(sys.argv) > 2 else "coffee"
N_CROSS = int(os.environ.get("BAKE_CROSS", "3"))


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    d = os.path.join("jobs", JOB); os.makedirs(d, exist_ok=True)
    out = os.path.join(d, "marble"); os.makedirs(out, exist_ok=True)
    srcs = sorted(glob.glob(os.path.join(SRC, "p*.jpg")))
    if not srcs:
        raise SystemExit(f"no p*.jpg in {SRC}")
    log(f"{len(srcs)} photos from {SRC}")

    # normalised copies the vision model and the UI both use
    names = [chr(ord("A") + i) for i in range(len(srcs))]
    small = []
    for name, src in zip(names, srcs):
        im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
        im.save(os.path.join(d, f"{name}.jpg"), quality=92)
        im.thumbnail((1024, 1024), Image.LANCZOS)
        p = os.path.join(d, f"{name}_1024.jpg"); im.save(p, quality=88); small.append(p)

    # people detection runs while Marble builds geometry
    res = {}
    def do_people():
        try:
            res["people"] = perception.detect_people(small, names)
            log("vision: %d people, %d fixtures" % (len(res["people"][0]), len(res["people"][1])))
        except Exception as e:
            log("vision failed:", str(e)[:200]); res["people"] = ([], [])
    tp = threading.Thread(target=do_people, daemon=True); tp.start()

    log("uploading to Marble")
    ids = [marble.upload_image(os.path.join(d, f"{n}.jpg")) for n in names]

    n = len(ids)
    az = [round(i * 360.0 / n, 1) for i in range(n)]        # clockwise sweep
    log(f"generating primary world, azimuths {az}")
    try:
        prim = marble.generate_multi_image(ids, azimuths=az, display_name=f"{JOB} room")
    except Exception as e:
        if "422" not in str(e):
            raise
        log("azimuth rejected for %d views, retrying in auto-layout" % n)
        body = {"world_prompt": {"type": "multi-image",
                                 "multi_image_prompt": [{"content": marble._content(asset_id=a)} for a in ids],
                                 "reconstruct_images": True},
                "model": os.environ.get("MARBLE_MODEL", "marble-1.1"),
                "display_name": f"{JOB} room", "permission": {"public": False}}
        prim = marble._req("POST", f"{marble.API}/worlds:generate", headers=marble._h(), json=body).json()

    cross = []
    for i in range(0, n, max(1, n // max(N_CROSS, 1)))[:N_CROSS]:
        try:
            cross.append((f"v{i}", marble.generate_single_image(ids[i], display_name=f"{JOB} view {i}")))
            log(f"cross-check view {i} started")
        except Exception as e:
            log(f"cross view {i} not started: {str(e)[:160]}")

    log("waiting for the room (Marble takes 5-15 min)")
    world = marble.world_of(marble.wait(prim, on_progress=lambda m: log("  ", str(m)[:90])))
    wid = world.get("world_id")
    scale, ground = marble.semantics_of(world)
    log(f"room done: {wid}  scale={scale} ground={ground}")
    primary = dict(world_id=wid, scale=scale, ground_offset=ground,
                   **marble.fetch_scene(wid, out, tag="primary"))
    log("primary splat:", primary.get("ply"))

    extras = []
    for tag, op in cross:
        try:
            w = marble.world_of(marble.wait(op, timeout_s=900))
            s, gr = marble.semantics_of(w)
            extras.append(dict(world_id=w.get("world_id"), scale=s, ground_offset=gr,
                               **marble.fetch_scene(w.get("world_id"), out, want_mesh=False, tag=tag)))
            log(f"cross view {tag} downloaded")
        except Exception as e:
            log(f"cross view {tag} failed: {str(e)[:160]}")

    scenes = [primary] + extras
    log(f"cross-comparing {len(scenes)} reconstructions")
    pts, T0, report = fusion.cross_compare(scenes, recon.load_gaussian_ply,
                                           min_support=2 if len(scenes) > 1 else 1)
    json.dump(report, open(os.path.join(out, "fusion.json"), "w"), indent=1)
    log("fusion:", json.dumps(report)[:300])
    np.save(os.path.join(out, "points.npy"), pts)
    tp.join(timeout=120)
    json.dump({"people_raw": res.get("people", ([], []))[0],
               "fixtures": res.get("people", ([], []))[1],
               "names": names, "splat": os.path.basename(primary.get("spz") or primary.get("ply") or "")},
              open(os.path.join(out, "stage1.json"), "w"), indent=1)
    log("STAGE 1 COMPLETE — geometry and people cached; run bake_demo_stage2.py next")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
