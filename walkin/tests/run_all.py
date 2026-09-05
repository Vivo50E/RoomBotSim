"""Headless checks. python tests/run_all.py"""
import os, sys, json, math, time, tempfile, shutil, subprocess, copy, threading, socket, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("WORLD_SOURCE", "mock")
os.environ.pop("OPENROUTER_API_KEY", None)          # no network in tests
import numpy as np
import uvicorn, websockets

PASS, FAIL = [], []
def check(name, cond, info=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name + (f"  {info}" if info else ""))

import tools.make_test_world as mtw
from geometry import RES
from planning import inflate, plan_xy, reach_grid, blocked_at, snap_free_xy
import agents as agentsmod, commands, server, episodes as epmod
from runtime_ops import load_jobs, rotate_episodes

# Every generated artifact belongs to a disposable fixture, never jobs/test.
TEST_ROOT = tempfile.mkdtemp(prefix="walkin-suite-")
JOB_DIR = os.path.join(TEST_ROOT, "test")
world, people = mtw.build(JOB_DIR)
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
rt = server.Runtime("test", world, people, JOB_DIR)
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

# The shared cache may reuse only the people stamp; each robot still gets its own robot stamp.
cache_rt = server.Runtime("test", world, people, JOB_DIR)
for _ in range(3): cache_rt.drop_robot()
for r in cache_rt.robots.values():
    if r.active: r.goal = tuple(lm["table"]["center_xy"])
original_stamp, stamps = server.stamp_discs, []
def count_stamp(*args, **kwargs):
    stamps.append(args[1])
    return original_stamp(*args, **kwargs)
server.stamp_discs = count_stamp
try:
    for k, r in cache_rt.robots.items():
        if r.active: cache_rt.drive_robot(k, 1.0)
finally:
    server.stamp_discs = original_stamp
check("planner: shares people grid but retains robot-specific stamps", len(stamps) == 4,
      f"stamp calls={len(stamps)} (1 people + 3 robot)")

# Human teleop duration begins at this episode, rather than a preceding episode's last skill.
human_id = rt.start_episode({"task_type": "go_to", "policy": {"kind": "human"}})
check("human recording: teleop clock resets at episode start", human_id is not None and rt.ep_t_step == rt.t)
rt.end_episode("aborted")

# --- full oracle episode
rt2 = server.Runtime("test", world, people, JOB_DIR)
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
eps = epmod.list_episodes(JOB_DIR)
last = next((e for e in eps if e["episode_id"] == eid), None)
check("episode: finished", done and last is not None, f"t={rt2.t:.0f}s")
if last:
    check("episode: oracle succeeds", bool(last["success"]),
          f'steps={last["n_steps"]} tags={last["tags"]}')

# --- recording + sft
import tools.make_sft as ms
n = ms.build(os.path.join(JOB_DIR, "episodes"), os.path.join(JOB_DIR, "sft.jsonl"), relabel="oracle", include_human=True)
check("sft: examples written", n >= 5, f"{n} examples")

# --- bad policy
class Bad:
    kind = "scripted"
    def __init__(s, seq): s.seq = list(seq)
    def act(s, obs): return s.seq.pop(0) if s.seq else {"action": "done"}
import policy as pm
rt3 = server.Runtime("test", world, people, JOB_DIR)
rt3.brain_enabled = False
bad_id = rt3.start_episode({"task_type": "coffee_to_person", "requester": 2, "policy": {"kind": "scripted"},
                            "seed": 1, "max_steps": 4, "max_seconds": 90})
rt3.ep_policy = Bad([{"action": "pick", "object": "pot"}, {"action": "done"}])
for i in range(int(95 / 0.01)):
    if i % 5 == 0: rt3.control_tick()
    rt3.sim.step(); rt3.t += 0.01
    if rt3.ep is None: break
bad = next((e for e in epmod.list_episodes(JOB_DIR) if e["episode_id"] == bad_id), {"tags": []})
check("bad policy: picking from across the room is tagged",
      any(t.startswith("precondition:too_far") for t in bad["tags"]), str(bad["tags"]))

# --- persisted restart recovery and safe archival rotation
scratch = tempfile.mkdtemp(prefix="walkin-test-")
try:
    jd = os.path.join(scratch, "reloadable"); os.makedirs(jd)
    shutil.copy(os.path.join(JOB_DIR, "world.json"), os.path.join(jd, "world.json"))
    shutil.copy(os.path.join(JOB_DIR, "people.json"), os.path.join(jd, "people.json"))
    restored = load_jobs(scratch)
    check("restart: persisted world and people reload", "reloadable" in restored and restored["reloadable"]["runtime"] is None)
    # The boot-loaded record is operational, not merely visible in /status.
    server.JOBS["reloadable"] = restored["reloadable"]
    started = server.runtime_start("reloadable")
    check("restart: boot-loaded job starts a usable runtime", started["ok"] and server.JOBS["reloadable"]["runtime"] is not None)
    server.JOBS["reloadable"]["runtime"].stop = True
    del server.JOBS["reloadable"]
    ed = os.path.join(jd, "episodes"); os.makedirs(ed)
    for n in range(3):
        with open(os.path.join(ed, f"e{n}.jsonl"), "w") as f:
            f.write('{"type": "header", "episode_id": "e%d", "cfg": {"policy": {"kind": "human"}}}\n' % n)
            f.write('{"type": "step", "step": 1, "obs": {"task": "x"}, "action": {"action": "done"}, "result": {"ok": true}}\n')
            f.write('{"type": "footer", "success": true, "tags": [], "n_steps": 1}\n')
        os.utime(os.path.join(ed, f"e{n}.jsonl"), (100 + n, 100 + n))
    # A live/corrupt file containing footer text must not be mistaken for closed.
    with open(os.path.join(ed, "live.jsonl"), "w") as f:
        f.write('{"type": "header"}\n{"type": "step", "note": "footer"}\n')
    rotate_episodes(jd, retain=1)
    listed = epmod.list_episodes(jd)
    exported = os.path.join(scratch, "sft.jsonl")
    exported_n = ms.build(ed, exported, relabel="none", include_human=True)
    check("episodes: rotation detects final footer, preserves listing and SFT data",
          len(listed) == 4 and len(os.listdir(os.path.join(ed, "archive"))) == 2
          and os.path.exists(os.path.join(ed, "live.jsonl")) and exported_n == 3,
          f"listed={len(listed)} archived={len(os.listdir(os.path.join(ed, 'archive')))} sft={exported_n}")
finally:
    shutil.rmtree(scratch)

# --- batch runner must persist an episode and exit without uvicorn
batch = subprocess.run([sys.executable, "tools/batch_run.py", "--jobs-root", TEST_ROOT, "--job", "test", "--episodes", "1",
                        "--task", "go_to", "--max-seconds", "40"], capture_output=True, text=True, timeout=60)
check("batch: headless runner exits and persists", batch.returncode == 0 and '"finished": true' in batch.stdout,
      batch.stderr[-160:])
check("api: durable /api/act cancellation policy is discoverable",
      "continue after client disconnect" in server.api_schema()["cancellation"])

# --- realtime budget: exact 6-person/3-robot target over the real WebSocket route
_perf_people = {"people": copy.deepcopy(people["people"])}
while len(_perf_people["people"]) < 6:
    p = copy.deepcopy(_perf_people["people"][len(_perf_people["people"]) % 4])
    p.update(id=len(_perf_people["people"]) + 1,
             pos_xy=[-1.4, 1.4] if len(_perf_people["people"]) == 4 else [1.4, -1.4],
             home_xy=[-1.4, 1.4] if len(_perf_people["people"]) == 4 else [1.4, -1.4],
             color=server.COLORS[len(_perf_people["people"])], posture="standing", activity="walking", talking_to=None)
    _perf_people["people"].append(p)
perf_job = f"perf_ws_{os.getpid()}_{time.time_ns()}"
perf = server.Runtime(perf_job, world, _perf_people, JOB_DIR)
perf.brain_enabled = False
for _ in range(3): perf.drop_robot()
server.JOBS[perf_job] = dict(dir=JOB_DIR, photos=["A"],
                             status=dict(stage="running", qwen="skipped", atlas="done", message="performance fixture"),
                             world=world, people=_perf_people, runtime=perf)

# Reserve a loopback port, then hand that actual listening socket to uvicorn.  The
# client below uses the installed websockets package, never a FastAPI test transport.
listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("127.0.0.1", 0))
listener.listen(128)
perf_port = listener.getsockname()[1]
asgi_server = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=perf_port,
                                             log_level="warning", access_log=False, lifespan="off"))
asgi_thread = threading.Thread(target=lambda: asgi_server.run(sockets=[listener]), daemon=True)
asgi_thread.start()
server_ready = False
for _ in range(100):
    if asgi_server.started:
        server_ready = True
        break
    if not asgi_thread.is_alive():
        break
    time.sleep(.05)

client_stop = threading.Event()
client_connected = threading.Event()
state_frame = threading.Event()
state_counts = {"frames": 0, "matching": 0}
client_errors = []
def consume_network_websocket():
    async def receive_frames():
        uri = f"ws://127.0.0.1:{perf_port}/ws/{perf_job}"
        try:
            async with websockets.connect(uri, open_timeout=5, close_timeout=2) as websocket:
                client_connected.set()
                while not client_stop.is_set():
                    try:
                        frame = json.loads(await asyncio.wait_for(websocket.recv(), timeout=.25))
                    except asyncio.TimeoutError:
                        continue
                    if frame.get("type") == "state":
                        state_counts["frames"] += 1
                        if len(frame.get("people", [])) == 6 and len(frame.get("robots", [])) == 3:
                            state_counts["matching"] += 1
                            state_frame.set()
        except Exception as exc:
            if not client_stop.is_set():
                client_errors.append(str(exc))
    asyncio.run(receive_frames())

client = threading.Thread(target=consume_network_websocket, daemon=True)
elapsed_sim = elapsed_wall = 0.0
try:
    if server_ready:
        client.start()
        connected = client_connected.wait(5)
        if connected:
            perf.start()
            wall = time.perf_counter()
            time.sleep(30.1)
            elapsed_sim = perf.t
            elapsed_wall = time.perf_counter() - wall
        else:
            client_errors.append("WebSocket client did not connect")
    else:
        client_errors.append("uvicorn did not start")
finally:
    client_stop.set()
    if client.is_alive():
        client.join(3)
    perf.stop = True
    asgi_server.should_exit = True
    if asgi_thread.is_alive():
        asgi_thread.join(5)
    server.JOBS.pop(perf_job, None)
    try:
        listener.close()
    except OSError:
        pass

_n_people = len(perf.cast.agents); _n_robots = sum(1 for r in perf.robots.values() if r.active)
check("performance: real network WebSocket connects and receives 6-person/3-robot state frames",
      server_ready and client_connected.is_set() and state_frame.is_set() and state_counts["frames"] > 0
      and state_counts["matching"] > 0 and not client_errors,
      f"frames={state_counts['frames']} matching={state_counts['matching']} errors={client_errors}")
check("performance: 30 s wall advances >=29 s (6 people, 3 robots, network WebSocket client)",
      elapsed_wall >= 30.0 and elapsed_sim >= 29.0 and _n_people == 6 and _n_robots == 3,
      f"sim={elapsed_sim:.2f}s wall={elapsed_wall:.2f}s")

shutil.rmtree(TEST_ROOT)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("failed:", FAIL)
sys.exit(1 if FAIL else 0)
