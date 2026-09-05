"""Headless checks. python tests/run_all.py"""
import os, sys, json, math, time, tempfile, shutil, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("WORLD_SOURCE", "mock")
os.environ.pop("OPENROUTER_API_KEY", None)          # no network in tests
import numpy as np

PASS, FAIL = [], []
def check(name, cond, info=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name + (f"  {info}" if info else ""))

import tools.make_test_world as mtw
from geometry import RES
from planning import inflate, plan_xy, reach_grid, blocked_at, snap_free_xy
import agents as agentsmod, commands, server, episodes as epmod
from runtime_ops import load_jobs, rotate_episodes

world, people = mtw.build("jobs/test")
nx, ny = world["size_cells"]
occ = agentsmod.decode(world["occupancy_b64"], nx, ny)
origin = world["origin_xy"]
lm = {l["label"]: l for l in world["landmarks"]}

# --- grid
blocked = inflate(occ, 0.25)
p = plan_xy(blocked, origin, lm["door"]["center_xy"], lm["table"]["center_xy"])
check("grid: door -> table plans", isinstance(p, list) and len(p) > 1, f"{len(p) if p else 0} waypoints")
check("grid: obstacle count within cap", len(world["obstacles"]) <= 400, str(len(world["obstacles"])))

# --- a*
rg = reach_grid(blocked)
free = np.argwhere(~rg)
rng = np.random.default_rng(0)
ok = 0
pairs = rng.choice(len(free), (60, 2))
for a, b in pairs:
    (j0, i0), (j1, i1) = free[a], free[b]
    s = (origin[0] + (i0 + .5) * RES, origin[1] + (j0 + .5) * RES)
    g = (origin[0] + (i1 + .5) * RES, origin[1] + (j1 + .5) * RES)
    path = plan_xy(blocked, origin, s, g)
    if path and not any(blocked_at(blocked, origin, x, y) for x, y in path):
        ok += 1
check("astar: 60 random pairs all solved with clear paths", ok == 60, f"{ok}/60")

# --- commands
P = {a["id"]: a["description"] for a in people["people"]}
L = {l["id"]: l["label"] for l in world["landmarks"]}
exp = [("everyone go to the whiteboard", ["all"], "go_to", lm["whiteboard"]["id"]),
       ("person 2 walk to the door", [2], "go_to", lm["door"]["id"]),
       ("robot go to the table", ["robot_1"], "go_to", lm["table"]["id"]),
       ("3 and 4 follow the robot", [3, 4], "follow", "robot_1"),
       ("everybody sit back down", ["all"], "home", None)]
good = 0
for s, tg, it, tgt in exp:
    d = commands.keyword_parse(s, P, ["robot_1"], L)
    if d.get("ok") and d["targets"] == tg and d["intent"] == it and d["target"] == tgt:
        good += 1
    else:
        print("   mismatch:", s, d)
check("commands: 5 required sentences", good == 5, f"{good}/5")

# --- runtime, cast, robot
rt = server.Runtime("test", world, people, "jobs/test")
rt.brain_enabled = False
for a in rt.cast.agents:
    a.activity = "walking"; a.seated_orig = False; a.seated = False
viol = 0
arrivals = 0
prev_intent = {a.id: a.intent for a in rt.cast.agents}
t0 = time.time()
steps = int(120 / 0.05)
for i in range(steps):
    rt.t += 0.05
    rt.cast.tick(rt.t, 0.05, list(rt.robots.values()))
    for a in rt.cast.agents:
        if blocked_at(occ, origin, a.x, a.y): viol += 1
    for x in range(len(rt.cast.agents)):
        for y in range(x + 1, len(rt.cast.agents)):
            A, B = rt.cast.agents[x], rt.cast.agents[y]
            if math.dist((A.x, A.y), (B.x, B.y)) < 0.30: viol += 1
arrivals = rt.cast.arrivals
check("cast: 120 s wandering, no obstacle or personal-space violations", viol == 0, f"{viol} violations")
check("cast: agents keep re-tasking", arrivals >= 20, f"{arrivals} intents completed")

# --- robot navigation, through the real navigate_to skill
k = rt.drop_robot("spot")
check("robot: spot drops in", k is not None and rt.robots[k].active)
from skills import SkillRunner
run = rt.runner[k]
run.start({"action": "navigate_to", "target": lm["table"]["id"]}, rt.t)
bumps0 = sum(1 for e in rt.events if "bumped" in e["text"])
res = None
for i in range(int(60 / 0.01)):
    if i % 5 == 0:
        rt.cast.tick(rt.t, 0.05, list(rt.robots.values()))
        r = rt.robots[k]
        r.x, r.y, r.yaw = rt.sim.robot_pose(k)
        res = run.tick(rt.t)
        if res is not None:
            break
    rt.sim.step(); rt.t += 0.01; rt.handle_contacts()
bumps = sum(1 for e in rt.events if "bumped" in e["text"]) - bumps0
check("robot: navigate_to the table succeeds", bool(res and res.get("ok")), str(res))
check("robot: few contacts crossing a room of walking people", bumps <= 3, f"{bumps} bumps")

# --- full oracle episode
rt2 = server.Runtime("test", world, people, "jobs/test")
rt2.brain_enabled = False
eid = rt2.start_episode({"task_type": "coffee_to_person", "requester": 2, "robot_type": "spot",
                         "policy": {"kind": "scripted"}, "seed": 3, "humans": "scripted",
                         "max_steps": 16, "max_seconds": 260})
check("episode: started", eid is not None, str(eid))
done = False
for i in range(int(280 / 0.01)):
    if i % 5 == 0:
        rt2.control_tick()
    rt2.sim.step(); rt2.t += 0.01; rt2.handle_contacts()
    if rt2.ep is None:
        done = True; break
eps = epmod.list_episodes("jobs/test")
last = next((e for e in eps if e["episode_id"] == eid), None)
check("episode: finished", done and last is not None, f"t={rt2.t:.0f}s")
if last:
    check("episode: oracle succeeds", bool(last["success"]),
          f'steps={last["n_steps"]} tags={last["tags"]}')

# --- recording + sft
import tools.make_sft as ms
n = ms.build("jobs/test/episodes", "jobs/test/sft.jsonl", relabel="oracle", include_human=True)
check("sft: examples written", n >= 5, f"{n} examples")

# --- bad policy
class Bad:
    kind = "scripted"
    def __init__(s, seq): s.seq = list(seq)
    def act(s, obs): return s.seq.pop(0) if s.seq else {"action": "done"}
import policy as pm
rt3 = server.Runtime("test", world, people, "jobs/test")
rt3.brain_enabled = False
bad_id = rt3.start_episode({"task_type": "coffee_to_person", "requester": 2, "policy": {"kind": "scripted"},
                            "seed": 1, "max_steps": 4, "max_seconds": 90})
rt3.ep_policy = Bad([{"action": "pick", "object": "pot"}, {"action": "done"}])
for i in range(int(95 / 0.01)):
    if i % 5 == 0: rt3.control_tick()
    rt3.sim.step(); rt3.t += 0.01
    if rt3.ep is None: break
bad = next((e for e in epmod.list_episodes("jobs/test") if e["episode_id"] == bad_id), {"tags": []})
check("bad policy: picking from across the room is tagged",
      any(t.startswith("precondition:too_far") for t in bad["tags"]), str(bad["tags"]))

# --- persisted restart recovery and safe archival rotation
scratch = tempfile.mkdtemp(prefix="walkin-test-")
try:
    jd = os.path.join(scratch, "reloadable"); os.makedirs(jd)
    shutil.copy("jobs/test/world.json", os.path.join(jd, "world.json"))
    shutil.copy("jobs/test/people.json", os.path.join(jd, "people.json"))
    restored = load_jobs(scratch)
    check("restart: persisted world and people reload", "reloadable" in restored and restored["reloadable"]["runtime"] is None)
    ed = os.path.join(jd, "episodes"); os.makedirs(ed)
    for n in range(3):
        with open(os.path.join(ed, f"e{n}.jsonl"), "w") as f:
            f.write('{"type": "header", "episode_id": "e%d", "cfg": {}}\n' % n)
            f.write('{"type": "footer", "success": true, "tags": [], "n_steps": 0}\n')
        os.utime(os.path.join(ed, f"e{n}.jsonl"), (100 + n, 100 + n))
    rotate_episodes(jd, retain=1)
    listed = epmod.list_episodes(jd)
    check("episodes: rotation archives safely and listing includes archives",
          len(listed) == 3 and len(os.listdir(os.path.join(ed, "archive"))) == 2)
finally:
    shutil.rmtree(scratch)

# --- batch runner must persist an episode and exit without uvicorn
batch = subprocess.run([sys.executable, "tools/batch_run.py", "--job", "test", "--episodes", "1",
                        "--task", "go_to", "--max-seconds", "40"], capture_output=True, text=True, timeout=60)
check("batch: headless runner exits and persists", batch.returncode == 0 and '"finished": true' in batch.stdout,
      batch.stderr[-160:])

# --- realtime budget: 6 people and all 3 active robots keep up with wall clock
import copy as _copy
_perf_people = {"people": _copy.deepcopy(people["people"])}
_PAL = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4"]
while len(_perf_people["people"]) < 6:            # the demo room ships four; the target is six
    _p = _copy.deepcopy(_perf_people["people"][len(_perf_people["people"]) % 4])
    _p["id"] = len(_perf_people["people"]) + 1
    _p["pos_xy"] = [_p["pos_xy"][0] + 0.5 * _p["id"], _p["pos_xy"][1] - 0.4 * _p["id"]]
    _p["home_xy"] = list(_p["pos_xy"]); _p["color"] = _PAL[_p["id"] - 1]
    _p["posture"] = "standing"; _p["activity"] = "walking"; _p["talking_to"] = None
    _perf_people["people"].append(_p)
perf = server.Runtime("test", world, _perf_people, "jobs/test")
perf.brain_enabled = False
for _ in range(3): perf.drop_robot()
perf.start()
wall = time.perf_counter(); time.sleep(30.1); elapsed_sim = perf.t; elapsed_wall = time.perf_counter() - wall
perf.stop = True
_n_people = len(perf.cast.agents); _n_robots = sum(1 for r in perf.robots.values() if r.active)
check(f"performance: 30 s wall advances >=29 s ({_n_people} people, {_n_robots} robots)",
      elapsed_sim >= 29.0 and _n_people >= 6 and _n_robots >= 3,
      f"sim={elapsed_sim:.2f}s wall={elapsed_wall:.2f}s")

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("failed:", FAIL)
sys.exit(1 if FAIL else 0)
