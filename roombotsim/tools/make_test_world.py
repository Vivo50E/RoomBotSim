"""Builds jobs/test/{world,people}.json from the hand-made room, with no API calls."""
import json, os, sys, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import recon
from recon import _mock
from geometry import (floor_align, apply_T, transform_camera, build_grid, rect_decompose, make_walls,
                      nearest_walkable, b64, RES)

COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4"]


def build(out_dir="jobs/test"):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "tools", "room_6x8.json")) as f:
        room = json.load(f)
    os.makedirs(out_dir, exist_ok=True)
    rec = _mock(out_dir)
    T = floor_align(rec.points, rec.cameras)
    P = apply_T(T, rec.points)
    cams = [transform_camera(T, c) for c in rec.cameras]
    T_room = T @ recon.MOCK_ROOM_TO_ATLAS
    room_xy = lambda xy: tuple(apply_T(T_room, [[xy[0], xy[1], 0.0]])[0][:2])
    people_xy = [room_xy(p["pos_xy"]) for p in room["people"]]
    grid = build_grid(P, people_xy)
    lms = []
    for b in room["boxes"]:
        if b["label"] == "chair":
            continue
        c = room_xy(b["center_xy"])
        lms.append(dict(label=b["label"], center_xy=[float(c[0]), float(c[1])], size_xy=b["size_xy"]))
    for k, lm in enumerate(lms):
        lm["id"] = f"lm_{k}"
        if lm["label"] in ("door", "whiteboard"):
            lm["center_xy"] = list(nearest_walkable(grid["walk"], grid["origin_xy"], *lm["center_xy"]))

    def cam_json(c):
        return dict(photo=c.photo, width=int(c.width), height=int(c.height),
                    K=np.asarray(c.K).tolist(), R=np.asarray(c.R).tolist(), t=np.asarray(c.t).reshape(3).tolist())

    world = dict(source="mock", resolution_m=RES, origin_xy=grid["origin_xy"], size_cells=grid["size_cells"],
                 occupancy_b64=b64(grid["occ"]), walkable_b64=b64(grid["walk"]), height_b64=b64(grid["height_cm"]),
                 obstacles=rect_decompose(grid["occ"], grid["origin_xy"], RES, grid["height_cm"]),
                 walls=make_walls(grid["origin_xy"], grid["size_cells"], RES), landmarks=lms,
                 cameras=[cam_json(c) for c in cams], T_atlas_to_sim=T.tolist(), splat_url=None,
                 photo_a_url=None, photo_urls=[])
    people = []
    for p, xy in zip(room["people"], people_xy):
        x, y = nearest_walkable(grid["walk"], grid["origin_xy"], *xy)
        table = next((l for l in lms if l["label"] == "table"), None)
        fx, fy = table["center_xy"] if table else (x + 1, y)
        people.append(dict(id=p["id"], description=p["description"], posture=p["posture"], activity=p["activity"],
                           talking_to=p["talking_to"], pos_xy=[x, y], home_xy=[x, y],
                           facing_deg=math.degrees(math.atan2(fy - y, fx - x)),
                           personality=p["personality"], color=COLORS[p["id"] - 1]))
    with open(os.path.join(out_dir, "world.json"), "w") as f:
        json.dump(world, f)
    with open(os.path.join(out_dir, "people.json"), "w") as f:
        json.dump(dict(people=people), f)
    return world, dict(people=people)


if __name__ == "__main__":
    w, p = build()
    print("obstacles", len(w["obstacles"]), "landmarks", [l["label"] for l in w["landmarks"]],
          "people", len(p["people"]), "grid", w["size_cells"])
