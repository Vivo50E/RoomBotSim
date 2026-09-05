"""WALK-IN server: photo pipeline, sim runtime, WebSocket, and an HTTP API any model can drive."""
import os, io, json, math, time, uuid, base64, logging, threading, asyncio, traceback
import numpy as np

if os.path.exists(".env"):
    for line in open(".env"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

import recon, geometry, perception, planning, sim as simmod, agents as agentsmod, commands, policy as policymod, episodes as epmod
from runtime_ops import EVENT_LIMIT, EVENT_LOG_LIMIT, atomic_json_dump, load_jobs
from geometry import (RES, floor_align, apply_T, transform_camera, camera_yaw, build_grid, grid_from_boxes,
                      rect_decompose, make_walls, extract_landmarks, nearest_walkable, place_fixture,
                      unproject, foot_pixel, scale_from_door, apply_scale, b64, fit_fallback_to_grid)
from planning import inflate, reach_grid, blocked_at, snap_free_xy, plan_xy, pursuit_target, wrap
from geometry import stamp_discs, carve_discs
from skills import SkillRunner
from sim import OBJECTS, surface_top, on_what

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("server")

COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4"]
PHOTOS = ["A", "B", "C", "D"]
BRAIN_PERIOD_S = float(os.environ.get("BRAIN_PERIOD_S", "5"))
RECORD_FRAMES = os.environ.get("RECORD_FRAMES", "0") == "1"

app = FastAPI(title="WALK-IN")
os.makedirs("jobs", exist_ok=True)
app.mount("/jobs", StaticFiles(directory="jobs"), name="jobs")
app.mount("/static", StaticFiles(directory="static"), name="static")
JOBS = load_jobs("jobs")
if JOBS:
    log.info("reloaded %d persisted jobs", len(JOBS))


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    import openrouter, marble
    return {"ok": True, "llm_key": openrouter.has_key(), "marble_key": marble.has_key(),
            "world_source": os.environ.get("WORLD_SOURCE", "marble"), "jobs": list(JOBS)}


# ---------------------------------------------------------------- upload + pipeline

@app.post("/upload")
async def upload(photos: list[UploadFile] = File(...)):
    if not 2 <= len(photos) <= 4:
        raise HTTPException(400, "upload 2 to 4 photos (4 chest-height views works best)")
    job_id = uuid.uuid4().hex[:8]
    d = os.path.join("jobs", job_id)
    os.makedirs(d, exist_ok=True)
    names = PHOTOS[:len(photos)]
    for name, f in zip(names, photos):
        raw = await f.read()
        open(os.path.join(d, f"{name}.jpg"), "wb").write(raw)
        im = Image.open(io.BytesIO(raw))
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((1024, 1024), Image.LANCZOS)
        im.save(os.path.join(d, f"{name}_1024.jpg"), quality=88)
    JOBS[job_id] = dict(dir=d, photos=names,
                        status=dict(stage="processing", qwen="running", atlas="running", message="starting"),
                        world=None, people=None, runtime=None)
    threading.Thread(target=pipeline, args=(job_id,), daemon=True).start()
    return {"job_id": job_id, "photos": names}


@app.get("/status/{job_id}")
def status(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "no such job")
    return {**j["status"], "world": j["world"], "people": j["people"], "photos": j.get("photos", PHOTOS[:2])}


def pipeline(job_id):
    j = JOBS[job_id]
    d, st, names = j["dir"], j["status"], j["photos"]
    small = [os.path.join(d, f"{n}_1024.jpg") for n in names]
    full = [os.path.join(d, f"{n}.jpg") for n in names]
    src = os.environ.get("WORLD_SOURCE", "marble")
    res = {}

    def do_people():
        try:
            res["people"] = perception.detect_people(small, names)
            st["qwen"] = "done"
        except Exception as e:
            log.warning("people detection failed: %s", e)
            res["people"] = ([], [])
            st["qwen"] = "error"

    def do_recon():
        try:
            res["recon"] = recon.reconstruct(full, d, status=st)
            st["atlas"] = "done"
        except Exception as e:
            log.warning("reconstruction failed: %s\n%s", e, traceback.format_exc()[-800:])
            st["atlas"] = "error"
            st["message"] = f"reconstruction failed, using photo layout: {e}"[:200]

    try:
        t1 = threading.Thread(target=do_people, daemon=True)
        t2 = threading.Thread(target=do_recon, daemon=True) if src != "fallback" else None
        t1.start()
        if t2: t2.start()
        t1.join()
        if t2:
            t2.join()
        else:
            st["atlas"] = "skipped"

        people_raw, fixtures = res["people"]
        rec = res.get("recon")
        pos, fb = {}, None

        if rec is not None and len(rec.points) > 500:
            T = floor_align(rec.points, rec.cameras)
            P = apply_T(T, rec.points)
            cams = [transform_camera(T, c) for c in rec.cameras]
            depths = dict(rec.depths)
            cams_by = {c.photo: c for c in cams}
            door = next((f for f in fixtures if f["label"] == "door"), None)
            s = scale_from_door(door, cams_by, depths)
            P, cams, depths, T = apply_scale(s, P, cams, depths, T)
            cams_by = {c.photo: c for c in cams}
            for p in people_raw:
                cands = []
                for ph, bb in p["boxes"].items():
                    if depths.get(ph) is not None and ph in cams_by:
                        X = unproject(*foot_pixel(bb), cams_by[ph], depths[ph])
                        if X is not None and abs(X[2]) < 0.6:
                            cands.append((bb[3] - bb[1], (float(X[0]), float(X[1]))))
                if cands:
                    pos[p["id"]] = max(cands)[1]
            grid = build_grid(P, people_xy=list(pos.values()))
            ox, oy = grid["origin_xy"]; nx, ny = grid["size_cells"]
            room_center = (ox + nx * RES / 2, oy + ny * RES / 2)
            missing = [p for p in people_raw if p["id"] not in pos]
            if missing or not cams:
                st["message"] = "estimating room layout"
                try:
                    fb = perception.fallback_layout(small, names, people_raw)
                except Exception as e:
                    log.warning("fallback layout failed: %s", e)
                    fb = None
                mapf = fit_fallback_to_grid(fb["size_xy"], grid) if fb else None
                for p in missing:
                    if mapf and p["id"] in fb["people"]:
                        pos[p["id"]] = mapf(*fb["people"][p["id"]][0])
                    else:
                        pos[p["id"]] = room_center
                if not cams and fb and mapf:
                    W_px, H_px = Image.open(small[0]).size
                    for ph in names:
                        if ph in fb["cameras"]:
                            xy, yaw = fb["cameras"][ph]
                            cams.append(perception.synth_camera(ph, mapf(*xy), yaw, W_px, H_px))
                    cams_by = {c.photo: c for c in cams}
            landmarks = extract_landmarks(grid["occ"], grid["origin_xy"], RES, grid["size_cells"])
            topdown = render_topdown(grid, landmarks, cams, os.path.join(d, "topdown.png"))
            st["message"] = "labelling the room"
            landmarks = perception.label_landmarks(topdown, small, names, landmarks)
            for f in fixtures:
                lm = place_fixture(f["label"], f["bbox"], cams_by.get(f["photo"]), depths.get(f["photo"]),
                                   grid["walk"], grid["origin_xy"], room_center)
                if lm:
                    landmarks.append(lm)
            if fb:
                have = {l["label"] for l in landmarks}
                mapf = mapf if 'mapf' in dir() and mapf else fit_fallback_to_grid(fb["size_xy"], grid)
                for l in fb["landmarks"]:
                    if l["label"] in ("door", "whiteboard", "window", "tv") and l["label"] not in have:
                        c = mapf(*l["center_xy"])
                        c = nearest_walkable(grid["walk"], grid["origin_xy"], *c)
                        landmarks.append(dict(label=l["label"], center_xy=list(c), size_xy=l["size_xy"]))
            source = src
            splat_url = f"/jobs/{job_id}/marble/{os.path.basename(rec.splat_path)}" if rec.splat_path else None
            T_out = T
        else:
            st["message"] = "estimating room layout from the photos"
            fb = perception.fallback_layout(small, names, people_raw)
            pos = {pid: tuple(xy) for pid, (xy, face) in fb["people"].items()}
            grid = grid_from_boxes(fb["size_xy"], fb["landmarks"], people_xy=list(pos.values()))
            ox, oy = grid["origin_xy"]; nx, ny = grid["size_cells"]
            room_center = (ox + nx * RES / 2, oy + ny * RES / 2)
            landmarks = [dict(label=b["label"], center_xy=b["center_xy"], size_xy=b["size_xy"])
                         for b in fb["landmarks"] if b["label"] != "chair"]
            W_px, H_px = Image.open(small[0]).size
            cams = [perception.synth_camera(ph, *fb["cameras"][ph], W_px, H_px) for ph in names if ph in fb["cameras"]]
            for p in people_raw:
                if p["id"] not in pos:
                    pos[p["id"]] = room_center
            T_out = np.eye(4)
            source = "fallback"
            splat_url = None

        landmarks = landmarks[:12]
        seen = {}
        for k, lm in enumerate(landmarks):
            lm["id"] = f"lm_{k}"
            n = seen.get(lm["label"], 0) + 1
            seen[lm["label"]] = n
            if n > 1:
                lm["label"] = f'{lm["label"]} {n}'
        table = next((l for l in landmarks if l["label"].startswith(("table", "desk"))), None)
        people = []
        for p in people_raw:
            x, y = nearest_walkable(grid["walk"], grid["origin_xy"], *pos[p["id"]])
            if p["posture"] == "seated" and table:
                fx, fy = table["center_xy"]
            else:
                fx, fy = room_center
            facing = math.degrees(math.atan2(fy - y, fx - x))
            people.append(dict(id=p["id"], description=p["description"], posture=p["posture"],
                               activity=p["activity"], talking_to=p["talking_to"], pos_xy=[x, y],
                               home_xy=[x, y], facing_deg=facing, personality=p["personality"],
                               color=COLORS[(p["id"] - 1) % len(COLORS)]))
        world = dict(source=source, resolution_m=RES, origin_xy=grid["origin_xy"], size_cells=grid["size_cells"],
                     occupancy_b64=b64(grid["occ"]), walkable_b64=b64(grid["walk"]), height_b64=b64(grid["height_cm"]),
                     obstacles=rect_decompose(grid["occ"], grid["origin_xy"], RES, grid["height_cm"]),
                     walls=make_walls(grid["origin_xy"], grid["size_cells"], RES), landmarks=landmarks,
                     cameras=[cam_json(c) for c in cams], T_atlas_to_sim=np.asarray(T_out).tolist(),
                     splat_url=splat_url, photo_a_url=f"/jobs/{job_id}/{names[0]}_1024.jpg",
                     photo_urls=[f"/jobs/{job_id}/{n}_1024.jpg" for n in names])
        atomic_json_dump(os.path.join(d, "world.json"), world)
        atomic_json_dump(os.path.join(d, "people.json"), dict(people=people))
        j["world"] = world
        j["people"] = dict(people=people)
        st["stage"] = "ready"
        st["message"] = f'{len(people)} people, {len(landmarks)} landmarks, {len(world["obstacles"])} obstacles'
        log.info("job %s ready: %s", job_id, st["message"])
    except Exception as e:
        log.error("pipeline failed: %s\n%s", e, traceback.format_exc())
        st["stage"] = "error"
        st["message"] = str(e)[:300]


def cam_json(c):
    return dict(photo=c.photo, width=int(c.width), height=int(c.height),
                K=np.asarray(c.K).tolist(), R=np.asarray(c.R).tolist(), t=np.asarray(c.t).reshape(3).tolist())


def render_topdown(grid, landmarks, cams, path, px_per_m=60):
    from PIL import ImageDraw
    ox, oy = grid["origin_xy"]; nx, ny = grid["size_cells"]
    W, H = int(nx * RES * px_per_m), int(ny * RES * px_per_m)
    img = Image.new("RGB", (max(W, 32), max(H, 32)), (245, 245, 245))
    dr = ImageDraw.Draw(img)

    def to_px(x, y):
        return int((x - ox) * px_per_m), int(H - (y - oy) * px_per_m)

    occ = grid["occ"]
    js, is_ = np.where(occ)
    for j, i in zip(js, is_):
        x0, y0 = to_px(ox + i * RES, oy + (j + 1) * RES)
        x1, y1 = to_px(ox + (i + 1) * RES, oy + j * RES)
        dr.rectangle([x0, y0, x1, y1], fill=(70, 70, 80))
    for k, lm in enumerate(landmarks):
        cx, cy = lm["center_xy"]; sx, sy = lm["size_xy"]
        x0, y0 = to_px(cx - sx / 2, cy + sy / 2); x1, y1 = to_px(cx + sx / 2, cy - sy / 2)
        dr.rectangle([x0, y0, x1, y1], outline=(220, 40, 40), width=3)
        dr.text(((x0 + x1) / 2 - 6, (y0 + y1) / 2 - 8), str(k), fill=(220, 40, 40))
    for c in cams:
        x, y = to_px(float(c.t[0]), float(c.t[1])); yaw = camera_yaw(c)
        tip = (x + int(25 * math.cos(yaw)), y - int(25 * math.sin(yaw)))
        dr.polygon([(x - 8, y - 8), (x - 8, y + 8), tip], fill=(30, 120, 220))
        dr.text((x + 10, y + 10), c.photo, fill=(30, 120, 220))
    img.save(path)
    return path


@app.post("/runtime/start/{job_id}")
def runtime_start(job_id: str):
    """Start a persisted, already-confirmed job after a server restart."""
    j = JOBS.get(job_id)
    if not j or not j.get("world") or not j.get("people"):
        raise HTTPException(404, "job not ready")
    start_runtime(job_id)
    return {"ok": True, "job_id": job_id}


@app.post("/confirm/{job_id}")
async def confirm(job_id: str, body: dict):
    j = JOBS.get(job_id)
    if not j or not j["world"]:
        raise HTTPException(404, "job not ready")
    for e in body.get("people", []):
        p = next((x for x in j["people"]["people"] if x["id"] == e["id"]), None)
        if p and e.get("pos_xy"):
            p["pos_xy"] = e["pos_xy"]; p["home_xy"] = e["pos_xy"]
            p["facing_deg"] = e.get("facing_deg", p["facing_deg"])
    for e in body.get("landmarks", []):
        lm = next((x for x in j["world"]["landmarks"] if x["id"] == e["id"]), None)
        if lm:
            lm["label"] = e.get("label", lm["label"])
            lm["center_xy"] = e.get("center_xy", lm["center_xy"])
    atomic_json_dump(os.path.join(j["dir"], "world.json"), j["world"])
    atomic_json_dump(os.path.join(j["dir"], "people.json"), j["people"])
    start_runtime(job_id)
    return {"ok": True}


def start_runtime(job_id):
    j = JOBS[job_id]
    if j.get("runtime"):
        return j["runtime"]
    j["runtime"] = Runtime(job_id, j["world"], j["people"], j["dir"])
    j["runtime"].start()
    j["status"]["stage"] = "running"
    return j["runtime"]


@app.get("/demo")
def demo_room():
    """The pre-baked room, ready to run with no photos, no reconstruction and no waiting.
    Built by tools/bake_demo.py; DEMO_JOB names it."""
    job = os.environ.get("DEMO_JOB", "coffee")
    d = os.path.join("jobs", job)
    if job not in JOBS:
        if not os.path.exists(os.path.join(d, "world.json")):
            raise HTTPException(404, f"no baked room at {d}; run tools/bake_demo.py first")
        import runtime_ops
        loaded = runtime_ops.load_jobs("jobs")
        if job not in loaded:
            raise HTTPException(500, f"baked room at {d} did not load")
        JOBS[job] = loaded[job]
    start_runtime(job)
    j = JOBS[job]
    return {"job_id": job, "people": len(j["people"]["people"]),
            "landmarks": [l["label"] for l in j["world"]["landmarks"]]}


@app.get("/test")
def test_room():
    d = os.path.join("jobs", "test")
    if not os.path.exists(os.path.join(d, "world.json")):
        import tools.make_test_world as mtw
        mtw.build()
    world = json.load(open(os.path.join(d, "world.json")))
    people = json.load(open(os.path.join(d, "people.json")))
    JOBS["test"] = dict(dir=d, photos=["A"], status=dict(stage="ready", qwen="skipped", atlas="done",
                                                         message="demo room"),
                        world=world, people=people, runtime=JOBS.get("test", {}).get("runtime"))
    start_runtime("test")
    return {"job_id": "test"}


# ---------------------------------------------------------------- runtime

class Robot:
    def __init__(self, k, rtype):
        self.k = k
        self.id = f"robot_{k}"
        self.type = rtype
        self.active = False
        self.x = self.y = self.yaw = 0.0
        self.v = 0.0
        self.goal = None
        self.follow = None
        self.path = []
        self.last_plan = -1.0
        self.holding = None
        self.mode = "auto"
        self.teleop = (0.0, 0.0)
        self.stalled_since = -1.0


class Runtime:
    def __init__(self, job_id, world, people, job_dir):
        self.job_id = job_id
        self.world = world
        self.people = people
        self.job_dir = job_dir
        self.spawns = simmod.spawn_objects(world)
        self.sim = simmod.Sim(world, people["people"], simmod.ROBOT_TYPES, self.spawns)
        self.cast = agentsmod.Cast(people, world, self.sim)
        nx, ny = world["size_cells"]
        self.origin = world["origin_xy"]
        self.occ_static = agentsmod.decode(world["occupancy_b64"], nx, ny)
        self.blocked_robot = inflate(self.occ_static, 0.30)
        self.goal_grid_robot = reach_grid(self.blocked_robot)
        self.walk = agentsmod.decode(world["walkable_b64"], nx, ny)
        self.landmarks = {lm["id"]: lm for lm in world["landmarks"]}
        self.room_center = (self.origin[0] + nx * RES / 2, self.origin[1] + ny * RES / 2)
        self.robots = {k: Robot(k, simmod.ROBOT_TYPES.get(k, "spot")) for k in range(1, simmod.N_ROBOTS + 1)}
        self.runner = {k: SkillRunner(self, k) for k in self.robots}
        self.t = 0.0
        self.paused = False
        self.stop = False
        self.seq = 0
        self.events = []
        self.event_log = []
        # Per-planning-window stamped human grid shared by all active robots.
        self._planning_stamp_t = None
        self._planning_people_grid = None
        self.lock = threading.RLock()
        self.latest = None
        self.bump_debounce = {}
        self.yield_debounce = {}
        self.last_brain = -1e9
        self.brain_inflight = False
        self.brain_enabled = True
        self.standing_command = None
        self.held = {oid: None for oid in self.spawns}
        self.cup_filled = False
        self.spilled = False
        self.seed = 7
        self.ep = None
        self.ep_step = 0
        self.ep_phase = None
        self.ep_obs = None
        self.ep_policy = None
        self.ep_t_step = 0.0
        self.ep_events = []
        self.policy_inflight = False
        self.pending_action = None
        self.teleop_log = []
        self.demo = None                 # None | "fail" | "succeed": the scripted judge demo
        self.demo_crosser = None
        self.sim.reset_objects()

    # -------------------------------------------------- helpers
    def objects_by_id(self):
        return self.spawns

    def held_by(self, oid):
        return self.held.get(oid)

    def set_held(self, oid, who):
        self.held[oid] = who

    def push_event(self, text):
        self.seq += 1
        self.events.append({"seq": self.seq, "t": round(self.t, 2), "text": text})
        self.event_log.append(text)
        if self.ep is not None:
            self.ep_events.append(text)
        if len(self.events) > EVENT_LIMIT:
            self.events = self.events[-EVENT_LIMIT:]
        if len(self.event_log) > EVENT_LOG_LIMIT:
            self.event_log = self.event_log[-EVENT_LOG_LIMIT:]

    def label_of_xy(self, xy):
        best, bd = None, 1e9
        for lm in self.world["landmarks"]:
            d = math.dist(xy, lm["center_xy"])
            if d < bd:
                best, bd = lm, d
        return best["label"] if best and bd < 1.5 else "furniture"

    def pretty(self, geom):
        if geom.startswith("person_"):
            return f"Person {geom.split('_')[1]}"
        if geom.startswith("robot_"):
            return f"robot {geom.split('_')[1]}"
        if geom.startswith("wall"):
            return "the wall"
        if geom.startswith("obs_"):
            o = next((o for o in self.world["obstacles"] if o["id"] == geom), None)
            if o:
                return self.label_of_xy(o["center_xy"])
            return "furniture"
        return geom

    # -------------------------------------------------- loop
    def start(self):
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        dt = 0.01
        n = 0
        wall0 = time.perf_counter()
        while not self.stop:
            if not self.paused:
                if n % 5 == 0:
                    try:
                        self.control_tick()
                    except Exception:
                        log.error("control tick failed\n%s", traceback.format_exc())
                self.sim.step()
                self.t += dt
                n += 1
                self.handle_contacts()
                if (self.brain_enabled and self.t - self.last_brain >= BRAIN_PERIOD_S
                        and not self.brain_inflight and os.environ.get("OPENROUTER_API_KEY")):
                    self.fire_brain()
                if n % 5 == 0:
                    self.publish_state()
            else:
                n += 1
            target = wall0 + n * dt
            slack = target - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            elif slack < -0.5:
                wall0 = time.perf_counter() - n * dt

    def control_tick(self):
        with self.lock:
            self.cast.tick(self.t, 0.05, list(self.robots.values()))
            for k, r in self.robots.items():
                if not r.active:
                    continue
                r.x, r.y, r.yaw = self.sim.robot_pose(k)
                vx, vy = self.sim.robot_vel(k)
                r.v = math.hypot(vx, vy)
                if self.held.get("cup", "").__class__ is str and str(self.held.get("cup", "")).startswith("person_"):
                    pid = int(str(self.held["cup"]).split("_")[1])
                    a = self.cast.by_id.get(pid)
                    if a:
                        self.sim.set_object_pose("cup", a.x + 0.30 * math.cos(a.yaw), a.y + 0.30 * math.sin(a.yaw), 1.00)
                run = self.runner[k]
                if r.mode == "teleop":
                    v, w = r.teleop
                    self.sim.cmd_robot(k, v, w)
                    if self.ep is not None:
                        self.teleop_log.append([round(self.t, 2), round(v, 2), round(w, 2)])
                    continue
                if run.busy:
                    res = run.tick(self.t)
                    if res is not None:
                        self.on_skill_done(k, res)
                    continue
                if r.goal is not None:
                    self.drive_robot(k, self.t, arrive_radius=1.2 if r.follow is not None else 0.20)
                else:
                    self.sim.cmd_robot(k, 0.0, 0.0)
            self.episode_tick()

    def drive_robot(self, k, t, arrive_radius=0.20, ignore_pid=None):
        """Plans and drives one step. Returns "arrived" | "unreachable" | None."""
        r = self.robots[k]
        if r.follow is not None and isinstance(r.follow, int):
            a = self.cast.by_id.get(r.follow)
            if a:
                r.goal = (a.x, a.y)
        if r.goal is None:
            self.sim.cmd_robot(k, 0.0, 0.0)
            return None
        if math.dist((r.x, r.y), r.goal) < arrive_radius or (len(r.path) == 1 and math.dist((r.x, r.y), r.path[0]) < 0.12):
            self.sim.cmd_robot(k, 0.0, 0.0)
            r.path = []
            return "arrived"
        if t - r.last_plan >= 0.5:
            r.last_plan = t
            # Stamping people is identical for each robot at this simulation tick;
            # cache it so 3 active robots do not repeat two expensive dilations.
            stamp_key = (round(t, 2), ignore_pid)
            if self._planning_stamp_t != stamp_key:
                ppl_now = [(a.x, a.y) for a in self.cast.agents if a.id != ignore_pid]
                ppl_next = [(a.x + a.vx, a.y + a.vy) for a in self.cast.agents if a.id != ignore_pid]
                self._planning_people_grid = stamp_discs(self.blocked_robot, ppl_now + ppl_next, 0.45, self.origin)
                self._planning_stamp_t = stamp_key
            others = [(o.x, o.y) for o in self.robots.values() if o.active and o is not r]
            blocked = self._planning_people_grid
            if others:
                blocked = stamp_discs(blocked, others, 0.5, self.origin)
            if r.follow is not None:
                blocked = carve_discs(blocked, [r.goal], 0.8, self.origin)
            start = (r.x, r.y) if not blocked_at(self.goal_grid_robot, self.origin, r.x, r.y) \
                else snap_free_xy(self.goal_grid_robot, self.origin, r.x, r.y)
            r.path = plan_xy(blocked, self.origin, start, r.goal) or plan_xy(self.blocked_robot, self.origin, start, r.goal)
            if r.path is None:
                r.path = []
                self.sim.cmd_robot(k, 0.0, 0.0)
                return "unreachable"
        if not r.path:
            self.sim.cmd_robot(k, 0.0, 0.0)
            return None
        tx, ty = pursuit_target((r.x, r.y), r.path, 0.45)
        err = wrap(math.atan2(ty - r.y, tx - r.x) - r.yaw)
        vx, vy = self.sim.robot_vel(k)
        d_near, a_near = 1e9, None
        for a in self.cast.agents:
            if ignore_pid is not None and a.id == ignore_pid:
                continue
            if a.seated:
                continue        # seated people cannot step into the robot; the planner already routes around them
            d = math.dist((r.x, r.y), (a.x, a.y)) - 0.22
            if d < d_near:
                d_near, a_near = d, a
        turn = max(0.0, 1.0 - abs(err) / math.radians(60))
        v = 1.0 * max(0.0, min((d_near - 0.6) / 0.9, 1.0)) * turn
        reckless = (self.demo == "fail" and self.ep is not None and r.holding == "cup")
        if reckless:
            v = 0.8 * turn                       # the untrained policy barrels on
        if a_near is not None and d_near < 0.6 and not reckless:
            rel = (r.x - a_near.x, r.y - a_near.y)
            relv = (vx - a_near.vx, vy - a_near.vy)
            if rel[0] * relv[0] + rel[1] * relv[1] < 0:
                v = 0.0
                if t - self.yield_debounce.get((k, a_near.id), -9) > 3.0:
                    self.yield_debounce[(k, a_near.id)] = t
                    self.push_event(f"robot {k} yielded to Person {a_near.id}")
        # a robot that yields forever never finishes: after 3 s held up, ease past at walking pace
        w = 2.5 * err
        if v < 0.05:
            if r.stalled_since < 0:
                r.stalled_since = t
            elif t - r.stalled_since > (1.2 if self.demo == "succeed" else 3.0) and d_near > 0.34:
                v = 0.22 * turn                            # ease past rather than wait forever
                if a_near is not None:
                    bear = wrap(math.atan2(a_near.y - r.y, a_near.x - r.x) - r.yaw)
                    w += -1.0 * (1 if bear > 0 else -1)    # steering away from them while doing it
        else:
            r.stalled_since = -1.0
        w = max(-2.0, min(w, 2.0))
        self.sim.cmd_robot(k, v, w)
        return None

    def handle_contacts(self):
        for (k, other) in self.sim.robot_contacts():
            if self.t - self.bump_debounce.get((k, other), -9) > 2.0:
                self.bump_debounce[(k, other)] = self.t
                name = self.pretty(other)
                self.push_event(f"robot {k} bumped {name}")
                if self.ep is not None:
                    self.ep.tags.add("bumped_person" if other.startswith("person_") else "bumped_furniture")

    # -------------------------------------------------- brain
    def fire_brain(self):
        self.brain_inflight = True
        self.last_brain = self.t
        with self.lock:
            cmd = None
            if self.standing_command and self.t - self.standing_command[1] < 30:
                cmd = self.standing_command[0]
            prompt = agentsmod.build_brain_prompt(self.cast, list(self.robots.values()), self.event_log, cmd)

        def work():
            try:
                from openrouter import ask_json, brain_model
                d = ask_json(brain_model(), prompt, max_tokens=600, timeout=30)
                with self.lock:
                    for pid, intent, target, dur, thought in agentsmod.validate_intents(d, self.cast):
                        a = self.cast.by_id[pid]
                        if a.locked_until > self.t:
                            continue
                        self.cast.apply_intent(a, intent, target, dur, thought, self.t)
            except Exception as e:
                log.info("brain call skipped: %s", str(e)[:150])
            finally:
                self.brain_inflight = False

        threading.Thread(target=work, daemon=True).start()

    # -------------------------------------------------- operator
    def apply_command(self, text):
        people = {a.id: a.description for a in self.cast.agents}
        robots = [r.id for r in self.robots.values() if r.active] or ["robot_1"]
        lms = {i: lm["label"] for i, lm in self.landmarks.items()}
        parsed = commands.parse(text, people, robots, lms)
        if not parsed.get("ok"):
            self.push_event(f'Could not act on "{text[:40]}"')
            return
        targets = []
        for tg in parsed["targets"]:
            if tg == "all":
                targets += [a.id for a in self.cast.agents]
            elif tg == "robots":
                targets += [r.id for r in self.robots.values() if r.active]
            else:
                targets.append(tg)
        intent, target = parsed["intent"], parsed.get("target")
        with self.lock:
            for tg in targets:
                if isinstance(tg, int) and tg in self.cast.by_id:
                    dur = 30 if intent in ("wait", "face") else 0
                    self.cast.apply_intent(self.cast.by_id[tg], intent, target, dur, "", self.t, lock=True)
                elif isinstance(tg, str) and tg.startswith("robot_"):
                    k = int(tg.split("_")[1])
                    r = self.robots.get(k)
                    if not r or not r.active:
                        continue
                    self.runner[k] = SkillRunner(self, k)
                    if intent == "go_to" and target is not None:
                        xy = self.cast.resolve(target) or (self.landmarks.get(target) or {}).get("center_xy")
                        if xy:
                            r.goal = snap_free_xy(self.goal_grid_robot, self.origin, xy[0], xy[1])
                            r.follow = None
                    elif intent == "follow":
                        r.follow = int(target) if isinstance(target, int) else None
                    else:
                        r.goal = None; r.follow = None
            self.standing_command = (text, self.t)
        self.push_event(f"Command: {text}")

    def set_goal(self, robot_id, x, y):
        try:
            k = int(str(robot_id).split("_")[1])
        except Exception:
            k = 1
        r = self.robots.get(k)
        if not r or not r.active:
            return
        with self.lock:
            self.runner[k] = SkillRunner(self, k)
            r.goal = snap_free_xy(self.goal_grid_robot, self.origin, x, y)
            r.follow = None
            r.path = []
            r.last_plan = -1
        self.push_event(f"{r.id} sent to {self.label_of_xy(r.goal)}")

    def drop_robot(self, robot_type=None):
        with self.lock:
            k = next((k for k, r in self.robots.items()
                      if not r.active and (robot_type is None or r.type == robot_type)), None)
            if k is None:
                k = next((k for k, r in self.robots.items() if not r.active), None)
            if k is None:
                self.push_event("All robots are already in the room")
                return None
            door = next((lm for lm in self.world["landmarks"] if lm["label"].startswith("door")), None)
            if door:
                c = np.array(door["center_xy"], float)
                v = np.array(self.room_center) - c
                n = np.linalg.norm(v)
                p = c + (v / n * 0.5 if n > 1e-6 else np.array([0.5, 0.0]))
            else:
                rng = np.random.default_rng(0)
                ny, nx = self.walk.shape
                cand = np.argwhere(self.walk)
                if len(cand) == 0:
                    p = np.array(self.room_center)
                else:
                    pick = cand[rng.choice(len(cand), min(300, len(cand)), replace=False)]
                    best, bd = None, -1
                    for (j, i) in pick:
                        x = self.origin[0] + (i + 0.5) * RES; y = self.origin[1] + (j + 0.5) * RES
                        d = min([math.dist((x, y), (a.x, a.y)) for a in self.cast.agents] or [9])
                        if d > bd:
                            best, bd = (x, y), d
                    p = np.array(best)
            x, y = snap_free_xy(self.goal_grid_robot, self.origin, float(p[0]), float(p[1]))
            yaw = math.atan2(self.room_center[1] - y, self.room_center[0] - x)
            self.sim.teleport_robot(k, x, y, yaw)
            r = self.robots[k]
            r.active = True
            r.x, r.y, r.yaw = x, y, yaw
            r.goal = None; r.follow = None; r.path = []; r.holding = None; r.mode = "auto"
        self.push_event(f'{r.id} ({r.type}) entered at {"the door" if door else "the far corner"}')
        return k

    def pause(self):
        self.paused = not self.paused
        self.push_event("paused" if self.paused else "resumed")

    def reset(self):
        with self.lock:
            for a in self.cast.agents:
                a.x, a.y = a.home
                a.seated = a.seated_orig
                a.queue = []; a.path = []; a.goal = None; a.locked_until = -1
                a.yaw = math.radians(a.facing_deg)
                self.cast.apply_intent(a, "wait", None, 3, "", self.t)
            for k, r in self.robots.items():
                self.sim.teleport_robot(k, *self.sim.park[k])
                r.active = False; r.goal = None; r.follow = None; r.path = []; r.holding = None; r.mode = "auto"
                self.runner[k] = SkillRunner(self, k)
            self.sim.reset_objects()
            self.held = {oid: None for oid in self.spawns}
            self.cup_filled = False
            self.spilled = False
            self.standing_command = None
        self.push_event("reset")

    # -------------------------------------------------- episodes
    def object_state(self, oid):
        o = self.sim.object_pose(oid)
        half = OBJECTS[oid]["half_h"]
        return {"id": oid, "x": round(o["x"], 2), "y": round(o["y"], 2), "z": round(o["z"], 2),
                "on": on_what(self.world, o, half), "held_by": self.held.get(oid),
                "filled": bool(self.cup_filled) if oid == "cup" else None,
                "upright": round(o["upright"], 2)}

    def build_obs(self):
        ep = self.ep
        k = ep.robot
        r = self.robots[k]
        return {"episode_id": ep.id, "step": ep.step, "t": round(self.t, 2), "task": ep.cfg["task_text"],
                "requester": ep.cfg.get("requester"),
                "robot": {"id": r.id, "type": r.type, "x": round(r.x, 2), "y": round(r.y, 2),
                          "yaw_deg": round(math.degrees(r.yaw), 1), "holding": r.holding},
                "objects": [self.object_state(o) for o in self.spawns],
                "landmarks": [{"id": lm["id"], "label": lm["label"], "x": round(lm["center_xy"][0], 2),
                               "y": round(lm["center_xy"][1], 2)} for lm in self.world["landmarks"]],
                "people": [{"id": a.id, "description": a.description, "x": round(a.x, 2), "y": round(a.y, 2),
                            "seated": bool(a.seated)} for a in self.cast.agents],
                "last_action": ep.last_action, "last_result": ep.last_result,
                "events": self.ep_events[-6:], "spilled": bool(self.spilled),
                "actions": ["navigate_to(target)", "pick(object)", "place(object, target)", "pour(cup)",
                            "say(text)", "done"]}

    def start_episode(self, cfg):
        cfg = epmod.default_cfg(**cfg)
        if cfg.get("replay_of"):
            prev = next((e for e in epmod.list_episodes(self.job_dir) if e["episode_id"] == cfg["replay_of"]), None)
            if prev:
                for f in ("task_type", "task_text", "seed", "robot_type", "requester"):
                    if prev.get(f) is not None:
                        cfg[f] = prev[f]
        with self.lock:
            self.reset()
            self.seed = int(cfg.get("seed", 7))
            self.cast.rng = np.random.default_rng(self.seed)
            self.brain_enabled = (cfg.get("humans") == "brain")
            k = self.drop_robot(robot_type=cfg.get("robot_type"))
            if k is None:
                return None
            if cfg.get("requester") is None and self.cast.agents:
                cfg["requester"] = self.cast.agents[0].id
            if cfg["task_type"] == "go_to" and not cfg.get("target"):
                cfg["target"] = next(iter(self.landmarks), None)
            self.demo = cfg.get("demo") if cfg.get("demo") in ("fail", "succeed") else None
            self.demo_crosser = None
            if self.demo:
                # a busy room: at least two people are up and wandering at random between landmarks
                req = cfg.get("requester")
                up = [a for a in self.cast.agents if not a.seated_orig and a.id != req]
                if len(up) < 2:
                    for a in self.cast.agents:
                        if a.id != req and a not in up and len(up) < 2:
                            a.seated_orig = False; up.append(a)
                for a in up:
                    a.seated = False; a.activity = "walking"; a.talking_to = None
                    # standing up from a chair: step onto the nearest clear floor cell first
                    a.x, a.y = snap_free_xy(self.cast.goal_grid, self.origin, a.x, a.y)
                    self.cast.apply_intent(a, *self.cast.sample(a, self.t), self.t)
            self.ep = epmod.Episode(cfg, self.job_id, k, self.t, self.job_dir)
            self.ep_policy = policymod.Policy(cfg.get("policy") or {"kind": "scripted"})
            self.ep_phase = "request"
            self.ep_step = 0
            self.ep_events = []
            # A human may teleoperate immediately; its recorded segment starts now,
            # not at a previous episode's skill boundary.
            self.ep_t_step = self.t
            self.teleop_log = []
            self.policy_inflight = False
            self.pending_action = None
        self.push_event(f'episode {self.ep.id} started: {cfg["task_text"]}')
        return self.ep.id

    def demo_tick(self):
        """Act two of the judge demo. Once Spot carries the coffee, someone walks across its path.
        Untrained (\"fail\"): the robot does not yield, they collide, the coffee spills.
        Retrained (\"succeed\"): the robot's yield-and-sidestep logic is on, so it waits, steps
        around them, and delivers. Same policy both times; only the avoidance behaviour differs."""
        ep = self.ep; r = self.robots[ep.robot]
        req = ep.cfg.get("requester")
        # Act one: as soon as Spot has the pot, someone starts drifting toward the requester, so that by
        # the time the full cup is being carried they are standing right on the delivery path.
        if self.demo_crosser is None and r.holding == "pot":
            walkers = [a for a in self.cast.agents if not a.seated and a.id != req]
            if walkers:
                a = min(walkers, key=lambda a: math.dist((a.x, a.y), (r.x, r.y)))
                self.demo_crosser = a.id
                self.cast.demo_crosser = a.id
                a.queue = []
                self.cast.apply_intent(a, "go_to", req, 0, "coffee?", self.t, lock=True)
                a.intent_until = self.t + 90
                self.push_event(f"Person {a.id} heads over to Person {req}")
            return
        if r.holding != "cup" or not self.cup_filled:
            return
        a = self.cast.by_id.get(self.demo_crosser)
        if a is not None:
            if not getattr(a, "_crossing", False):
                a._crossing = True
                self.push_event(f"Person {a.id} crosses the room")
            ahead = (r.x + 1.0 * math.cos(r.yaw), r.y + 1.0 * math.sin(r.yaw))
            if self.demo == "fail":
                # untrained: they walk straight into Spot's path, re-aimed every tick, and it does not yield
                a.seated = False; a.intent = "go_to"; a.thought = "excuse me"
                a.goal = ahead; a.path = [ahead]; a.intent_until = self.t + 60; a.best_t = self.t
            elif not getattr(a, "_crossed", False):
                # retrained: they cross the robot's bow once; Spot yields and steps around them
                if math.dist((a.x, a.y), (r.x, r.y)) < 1.1:
                    a._crossed = True
                    self.cast.demo_crosser = None            # their courtesy comes back on
                    away = max(self.world["landmarks"], key=lambda l: math.dist(l["center_xy"], (r.x, r.y)))
                    self.cast.apply_intent(a, "go_to", away["id"], 0, "sorry", self.t, lock=True)
                elif a.goal is None or a.intent != "go_to":
                    far = (r.x + 3.0 * math.cos(r.yaw), r.y + 3.0 * math.sin(r.yaw))
                    a.seated = False; a.intent = "go_to"; a.thought = "excuse me"
                    a.goal = far; a.path = [ahead, far]; a.intent_until = self.t + 60; a.best_t = self.t
        if self.demo == "fail":
            for a in self.cast.agents:
                if math.dist((a.x, a.y), (r.x, r.y)) < 0.62:
                    self.cup_filled = False; self.spilled = True
                    self.sim.set_cup_filled(False)
                    self.sim.show_stain(r.x + 0.3 * math.cos(r.yaw), r.y + 0.3 * math.sin(r.yaw))
                    self.push_event(f"robot {ep.robot} bumped Person {a.id}")
                    self.push_event(f"robot {ep.robot} spilled the coffee")
                    ep.tags |= {"bumped_person", "spill"}
                    self.sim.cmd_robot(ep.robot, 0.0, 0.0)
                    self.runner[ep.robot].phase = None
                    self.end_episode("spill")
                    return

    def episode_tick(self):
        ep = self.ep
        if ep is None or ep.done:
            return
        k = ep.robot
        run = self.runner[k]
        if self.demo:
            self.demo_tick()
            if self.ep is None:
                return
        if self.ep_phase == "request" and not self.policy_inflight:
            if self.t - ep.t0 > ep.cfg["max_seconds"]:
                return self.end_episode("timeout")
            if ep.step >= ep.cfg["max_steps"]:
                return self.end_episode("max_steps")
            obs = self.build_obs()
            if RECORD_FRAMES:
                fp = os.path.join(self.job_dir, "episodes", f"{ep.id}_step{ep.step:03d}.jpg")
                if self.sim.render_eye(k, fp):
                    obs["robot_view"] = os.path.relpath(fp, self.job_dir)
            self.ep_obs = obs
            if self.ep_policy.kind == "human":
                self.ep_phase = "wait_human"
                return
            self.policy_inflight = True
            self.ep_phase = "thinking"

            def work():
                try:
                    raw = self.ep_policy.act(obs)
                    act, err = policymod.validate(raw)
                    if err:
                        self.on_policy_error(err, raw)
                    else:
                        self.pending_action = act
                except Exception as e:
                    self.on_policy_error("bad_json", {"error": str(e)[:200]})
                finally:
                    self.policy_inflight = False

            threading.Thread(target=work, daemon=True).start()
            return
        if self.pending_action is not None and not run.busy:
            act = self.pending_action
            self.pending_action = None
            self.ep_step = ep.step + 1
            self.ep_events = []
            self.ep_t_step = self.t
            res = run.start(act, self.t)
            self.ep_phase = "execute"
            if res is not None:
                self.on_skill_done(k, res)
            return
        if self.ep_phase == "execute" and not run.busy and run.result is not None:
            return

    def on_policy_error(self, kind, raw):
        ep = self.ep
        if ep is None:
            return
        ep.policy_errors += 1
        ep.tags.add(f"policy_error:{kind}")
        self.push_event(f"policy error: {kind}")
        ep.last_result = {"ok": False, "reason": f"policy_error:{kind}", "detail": {"raw": str(raw)[:200]}}
        if ep.policy_errors >= 2:
            self.end_episode("policy_error")
        else:
            self.ep_phase = "request"

    def on_skill_done(self, k, res):
        ep = self.ep
        run = self.runner[k]
        if ep is None or ep.done:
            run.phase = None
            return
        act = run.action or {}
        dur = self.t - self.ep_t_step
        teleop = self.teleop_log or None
        self.teleop_log = []
        ep.record_step(self.ep_obs or {}, act, res, list(self.ep_events), dur, teleop=teleop)
        verdict = "ok" if res.get("ok") else f'FAIL {res.get("reason")}'
        self.push_event(f'{ep.step} · {act.get("action")}({act.get("object") or act.get("target") or ""}) → {verdict}')
        run.phase = None
        if act.get("action") == "done":
            return self.end_episode("done")
        if ep.consecutive_fails >= 3:
            return self.end_episode("no_progress")
        self.ep_phase = "request"

    def end_episode(self, reason):
        ep = self.ep
        if ep is None or ep.done:
            return
        state = {"objects": {o: self.object_state(o) for o in self.spawns},
                 "robot": {"x": self.robots[ep.robot].x, "y": self.robots[ep.robot].y},
                 "people": {a.id: {"x": a.x, "y": a.y} for a in self.cast.agents},
                 "landmarks": self.landmarks}
        success, tags = epmod.evaluate(ep.cfg, state)
        ep.tags |= tags
        if reason in ("timeout", "max_steps", "no_progress", "policy_error", "aborted"):
            ep.tags.add(reason)
        self.demo = None
        self.cast.demo_crosser = None
        if reason == "done" and not success:
            ep.tags.add("done_early")
        ep.close(success, self.t - ep.t0, {"objects": state["objects"], "robot": state["robot"]})
        self.push_event(f'episode {ep.id}: {"SUCCESS" if success else "FAIL " + " ".join(sorted(ep.tags))}')
        self.ep_phase = None
        self.ep = None
        self.robots[ep.robot].mode = "auto"

    def human_skill(self, msg):
        """Human mode: record the teleop segment as a navigate_to step, then run the skill."""
        ep = self.ep
        if ep is None or ep.done:
            return
        k = ep.robot
        r = self.robots[k]
        if self.teleop_log:
            near, nd = None, 1e9
            for oid in self.spawns:
                o = self.sim.object_pose(oid)
                d = math.dist((r.x, r.y), (o["x"], o["y"]))
                if d < nd: near, nd = oid, d
            for lm in self.world["landmarks"]:
                d = math.dist((r.x, r.y), lm["center_xy"])
                if d < nd: near, nd = lm["id"], d
            for a in self.cast.agents:
                d = math.dist((r.x, r.y), (a.x, a.y))
                if d < nd: near, nd = a.id, d
            tgt = near if nd <= 1.0 else {"x": round(r.x, 2), "y": round(r.y, 2)}
            ep.record_step(self.build_obs(), {"action": "navigate_to", "target": tgt},
                           {"ok": True, "reason": "teleop", "detail": {}}, list(self.ep_events),
                           max(0.1, self.t - self.ep_t_step), teleop=list(self.teleop_log))
            self.teleop_log = []
        act, err = policymod.validate({"action": msg.get("action"), "object": msg.get("object"),
                                       "target": msg.get("target"), "text": msg.get("text")})
        if err:
            self.push_event(f"skill rejected: {err}")
            return
        self.ep_obs = self.build_obs()
        self.ep_t_step = self.t
        r.mode = "auto"
        res = self.runner[k].start(act, self.t)
        self.ep_phase = "execute"
        if res is not None:
            self.on_skill_done(k, res)

    # -------------------------------------------------- publish
    def publish_state(self):
        with self.lock:
            ep = self.ep
            st = {"type": "state", "t": round(self.t, 2),
                  "robots": [{"id": r.id, "type": r.type, "x": round(r.x, 3), "y": round(r.y, 3),
                              "yaw": round(r.yaw, 3), "v": round(r.v, 2),
                              "goal": list(r.goal) if r.goal else None,
                              "path": [[round(p[0], 2), round(p[1], 2)] for p in r.path],
                              "holding": r.holding}
                             for r in self.robots.values() if r.active],
                  "people": [{"id": a.id, "x": round(a.x, 3), "y": round(a.y, 3), "yaw": round(a.yaw, 3),
                              "seated": bool(a.seated), "activity": a.activity, "thought": a.thought}
                             for a in self.cast.agents],
                  "objects": [self.object_state(o) for o in self.spawns],
                  "episode": ({"id": ep.id, "step": ep.step, "phase": self.ep_phase, "task": ep.cfg["task_text"],
                               "policy": ep.cfg["policy"].get("kind"),
                               "last_action": ep.last_action, "last_result": ep.last_result,
                               "tags": sorted(ep.tags), "holding": self.robots[ep.robot].holding}
                              if ep is not None else None),
                  "paused": self.paused,
                  "events": self.events}
            self.latest = st

    def state_since(self, last_seq):
        st = self.latest
        if st is None:
            return None, last_seq
        ev = [e for e in st["events"] if e["seq"] > last_seq]
        new_seq = max([e["seq"] for e in ev], default=last_seq)
        out = dict(st)
        out["events"] = [{"t": e["t"], "text": e["text"]} for e in ev]
        return out, new_seq


# ---------------------------------------------------------------- websocket

@app.websocket("/ws/{job_id}")
async def ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    j = JOBS.get(job_id)
    if not j or not j.get("runtime"):
        await websocket.send_text(json.dumps({"type": "error", "message": "run not started"}))
        await websocket.close()
        return
    rt = j["runtime"]
    await websocket.send_text(json.dumps({"type": "snapshot", "world": j["world"], "people": j["people"]}))
    last = [0]

    async def sender():
        while True:
            st, last[0] = rt.state_since(last[0])
            if st:
                await websocket.send_text(json.dumps(st))
            await asyncio.sleep(0.05)

    async def receiver():
        while True:
            m = json.loads(await websocket.receive_text())
            ty = m.get("type")
            if ty == "command":
                threading.Thread(target=rt.apply_command, args=(m.get("text", ""),), daemon=True).start()
            elif ty == "goal":
                rt.set_goal(m.get("robot", "robot_1"), float(m["x"]), float(m["y"]))
            elif ty == "drop_robot":
                rt.drop_robot(m.get("robot_type"))
            elif ty == "pause":
                rt.pause()
            elif ty == "reset":
                rt.reset()
            elif ty == "teleop":
                k = int(str(m.get("robot", "robot_1")).split("_")[1])
                r = rt.robots.get(k)
                if r and r.active:
                    r.mode = "teleop" if (m.get("v") or m.get("w")) else "auto"
                    r.teleop = (float(m.get("v", 0)), float(m.get("w", 0)))
            elif ty == "skill":
                rt.human_skill(m)

    try:
        await asyncio.gather(sender(), receiver())
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception as e:
        log.info("ws closed: %s", str(e)[:120])


# ---------------------------------------------------------------- episodes + data HTTP

def _rt(job_id):
    j = JOBS.get(job_id)
    if not j or not j.get("runtime"):
        raise HTTPException(404, "run not started")
    return j["runtime"]


@app.post("/episode/start/{job_id}")
def episode_start(job_id: str, body: dict):
    rt = _rt(job_id)
    eid = rt.start_episode(body or {})
    if eid is None:
        raise HTTPException(409, "no robot available")
    return {"episode_id": eid}


@app.post("/episode/stop/{job_id}")
def episode_stop(job_id: str):
    rt = _rt(job_id)
    rt.end_episode("aborted")
    return {"ok": True}


@app.get("/episodes/{job_id}")
def episodes_list(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "no such job")
    return {"episodes": epmod.list_episodes(j["dir"])}


@app.post("/sft/{job_id}")
def build_sft(job_id: str, body: dict = None):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "no such job")
    body = body or {}
    import tools.make_sft as ms
    out = os.path.join(j["dir"], "sft.jsonl")
    n = ms.build(os.path.join(j["dir"], "episodes"), out,
                 relabel=body.get("relabel", "oracle"),
                 only_failures=bool(body.get("only_failures", False)),
                 include_human=bool(body.get("include_human", True)))
    return {"n": n, "url": f'/jobs/{job_id}/sft.jsonl'}


@app.post("/demo/retrain/{job_id}")
def demo_retrain(job_id: str):
    """Judge demo. Builds the real training set from the recorded failures; the fine-tune itself is
    simulated on the client with a progress bar, and the UI labels it as a demo."""
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "no such job")
    import tools.make_sft as ms
    out = os.path.join(j["dir"], "sft.jsonl")
    n = ms.build(os.path.join(j["dir"], "episodes"), out, relabel="oracle", only_failures=False, include_human=True)
    eps = epmod.list_episodes(j["dir"])
    fails = [e for e in eps if e.get("success") is False]
    tags = {}
    for e in fails:
        for t in e.get("tags") or []:
            tags[t] = tags.get(t, 0) + 1
    return {"examples": n, "failed_episodes": len(fails), "tags": tags, "url": f"/jobs/{job_id}/sft.jsonl",
            "note": "training set is real; the fine-tune shown in the UI is simulated for the demo"}


# ---------------------------------------------------------------- model-facing API

@app.get("/api/observe/{job_id}")
def api_observe(job_id: str):
    """The current observation, exactly what a policy is asked to act on."""
    rt = _rt(job_id)
    if rt.ep is None:
        raise HTTPException(409, "no episode running; POST /api/task/{job_id} first")
    return rt.build_obs()


@app.post("/api/task/{job_id}")
def api_task(job_id: str, body: dict):
    """Start an episode from plain text. body: {"task": "go find the coffee and bring it to me",
    "task_type": "coffee_to_person", "requester": 2, "robot_type": "spot", "seed": 7,
    "policy": {"kind": "raw"|"chat"|"scripted"|"human", ...}}"""
    rt = _rt(job_id)
    cfg = dict(body or {})
    if cfg.pop("task", None):
        cfg["task_text"] = (body or {}).get("task")
    cfg.setdefault("policy", {"kind": "human"})
    eid = rt.start_episode(cfg)
    if eid is None:
        raise HTTPException(409, "no robot available")
    return {"episode_id": eid, "observe": f"/api/observe/{job_id}", "act": f"/api/act/{job_id}"}


@app.post("/api/act/{job_id}")
def api_act(job_id: str, body: dict):
    """Execute one skill and block until it finishes.

    Accepted actions are intentionally durable across client disconnects; use
    ``POST /episode/stop`` for an explicit safe abort (see OPERATIONS.md).
    """
    rt = _rt(job_id)
    if rt.ep is None:
        raise HTTPException(409, "no episode running")
    act, err = policymod.validate(body)
    if err:
        raise HTTPException(400, {"error": err, "expected": "see GET /api/schema"})
    ep = rt.ep
    k = ep.robot
    before = ep.step
    with rt.lock:
        rt.ep_obs = rt.build_obs()
        rt.ep_t_step = rt.t
        rt.ep_events = []
        rt.robots[k].mode = "auto"
        res = rt.runner[k].start(act, rt.t)
        rt.ep_phase = "execute"
        if res is not None:
            rt.on_skill_done(k, res)
    t0 = time.time()
    while time.time() - t0 < 70:
        if rt.ep is None or rt.ep.step > before or rt.ep.done:
            break
        time.sleep(0.05)
    done = rt.ep is None or rt.ep.done
    last = (ep.last_result or {})
    return {"result": last, "step": ep.step, "episode_done": bool(done),
            "success": ep.success if done else None, "tags": sorted(ep.tags),
            "observation": None if done else rt.build_obs()}


@app.get("/api/schema")
def api_schema():
    return {
        "flow": ["POST /api/task/{job_id}", "GET /api/observe/{job_id}", "POST /api/act/{job_id} (repeat)",
                 "GET /episodes/{job_id}", "POST /sft/{job_id}"],
        "system_prompt": policymod.SKILL_SYSTEM,
        "action": {"action": "navigate_to|pick|place|pour|say|done", "target": "lm_2 | cup | 2 | null",
                   "object": "cup | pot | null", "text": "string | null"},
        "observation_keys": ["episode_id", "step", "task", "requester", "robot", "objects", "landmarks",
                             "people", "last_action", "last_result", "events", "spilled", "actions"],
        "cancellation": "Accepted /api/act requests continue after client disconnect or response timeout; POST /episode/stop explicitly aborts.",
        "note": "Or let the server call your model instead: POST /api/task with "
                "policy={'kind':'chat','url':'https://api.deepseek.com/v1/chat/completions',"
                "'model':'deepseek-chat','key':'sk-...'} and it runs the whole episode itself.",
    }
