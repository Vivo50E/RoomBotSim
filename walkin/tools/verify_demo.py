#!/usr/bin/env python3
"""Is the baked room actually demo-ready?

Runs the checks that decide whether a judge sees a working demo or a robot stuck against a counter:
geometry sane, a counter to put coffee on, a reachable stance beside it, and the full coffee episode
completing end to end with the scripted oracle.

Usage: python tools/verify_demo.py [job_id]
"""
import os, sys, json, math, base64, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
os.environ.setdefault("WORLD_SOURCE", "marble")
os.environ.pop("OPENROUTER_API_KEY", None)          # no LLM in the check
import numpy as np

JOB = sys.argv[1] if len(sys.argv) > 1 else "coffee"
OK, BAD = [], []
def check(name, cond, info=""):
    (OK if cond else BAD).append(name)
    print(("PASS " if cond else "FAIL ") + name + (f"  {info}" if info else ""), flush=True)

d = os.path.join("jobs", JOB)
world = json.load(open(os.path.join(d, "world.json")))
people = json.load(open(os.path.join(d, "people.json")))
nx, ny = world["size_cells"]
occ = np.frombuffer(base64.b64decode(world["occupancy_b64"]), np.uint8).reshape(ny, nx).astype(bool)
walk = np.frombuffer(base64.b64decode(world["walkable_b64"]), np.uint8).reshape(ny, nx).astype(bool)
labels = [l["label"] for l in world["landmarks"]]

check("room is a sensible size", 3.0 <= nx * 0.1 <= 40 and 3.0 <= ny * 0.1 <= 40,
      f"{nx*0.1:.1f} x {ny*0.1:.1f} m")
check("enough open floor to drive in", 0.25 <= walk.mean() <= 0.95, f"{100*walk.mean():.0f}% walkable")
check("landmarks were labelled", len([l for l in labels if l != "unknown"]) >= 2, str(labels))
check("people were placed", 1 <= len(people["people"]) <= 6, f'{len(people["people"])} people')
check("people stand on walkable floor",
      all(walk[min(max(int((p["pos_xy"][1]-world["origin_xy"][1])/0.1),0),ny-1),
               min(max(int((p["pos_xy"][0]-world["origin_xy"][0])/0.1),0),nx-1)] for p in people["people"]))

import sim as simmod, server, episodes as epmod
spawns = simmod.spawn_objects(world)
surf = simmod.surface_top(world, (spawns["cup"][0], spawns["cup"][1]))
check("coffee sits on a real surface, not the floor", surf > 0.3,
      f'surface {surf:.2f} m under the cup at z={spawns["cup"][2]:.2f}')

from planning import inflate, reach_grid, snap_free_xy, plan_xy
blocked = inflate(occ, 0.30)
origin = world["origin_xy"]
stance = None
for dist in (0.6, 0.7, 0.8):
    for k in range(24):
        a = 2*math.pi*k/24
        px, py = spawns["cup"][0] + dist*math.cos(a), spawns["cup"][1] + dist*math.sin(a)
        i, j = int((px-origin[0])/0.1), int((py-origin[1])/0.1)
        if 0 <= i < nx and 0 <= j < ny and not blocked[j, i]:
            stance = (px, py); break
    if stance: break
check("a robot can stand within arm's reach of the coffee", stance is not None,
      f"stance {stance[0]:.1f}, {stance[1]:.1f}" if stance else "no free cell 0.6-0.8 m from the cup")

rt = server.Runtime(JOB, world, people, d)
rt.brain_enabled = False
k = rt.drop_robot("spot")
check("Spot can enter the room", k is not None and rt.robots[k].active)

eid = rt.start_episode({"task_type": "coffee_to_person", "requester": people["people"][0]["id"],
                        "robot_type": "spot", "policy": {"kind": "scripted"}, "seed": 3,
                        "humans": "scripted", "max_steps": 18, "max_seconds": 300})
t0 = time.time(); seen = 0
for i in range(int(320/0.01)):
    if i % 5 == 0: rt.control_tick()
    rt.sim.step(); rt.t += 0.01; rt.handle_contacts()
    if rt.ep is None: break
    if rt.ep.step > seen:
        seen = rt.ep.step; st = rt.ep.steps[-1]
        print("   %2d %-12s %-9s -> %s %s" % (st["step"], st["action"].get("action"),
              str(st["action"].get("object") or st["action"].get("target"))[:9],
              "ok" if st["result"]["ok"] else "FAIL", st["result"]["reason"]), flush=True)
rec = next((e for e in epmod.list_episodes(d) if e["episode_id"] == eid), None)
check("the coffee task completes end to end", bool(rec and rec["success"]),
      f'{rec["n_steps"]} steps, tags {rec["tags"]}' if rec else "no record")

print(f"\n{len(OK)} passed, {len(BAD)} failed")
if BAD: print("failed:", BAD)
sys.exit(1 if BAD else 0)
