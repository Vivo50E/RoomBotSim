"""The cast: everyone in the room who is not a robot. 20 Hz kinematic controller with A* paths and a
social-force term, an always-on scripted sampler, and an optional LLM brain that overrides intents."""
import math, base64, logging
import numpy as np
from dataclasses import dataclass, field
from planning import (RES, inflate, reach_grid, blocked_at, snap_free_xy, plan_xy,
                      pursuit_target, social_force)

log = logging.getLogger("cast")


@dataclass
class Agent:
    id: int; x: float; y: float; yaw: float
    seated_orig: bool; seated: bool
    home: tuple; activity: str; talking_to: object; personality: str; description: str
    facing_deg: float = 0.0
    intent: str = "wait"; target: object = None; intent_until: float = 0.0
    goal: tuple | None = None; path: list = field(default_factory=list)
    face_xy: tuple | None = None; follow: object = None; last_follow_plan: float = -1.0
    queue: list = field(default_factory=list)
    thought: str = ""; locked_until: float = -1.0
    vx: float = 0.0; vy: float = 0.0
    best_d: float = 1e9; best_t: float = 0.0


def decode(b64s, nx, ny):
    return np.frombuffer(base64.b64decode(b64s), np.uint8).reshape(ny, nx).astype(bool)


class Cast:
    def __init__(self, people, world, sim, seed=1):
        self.sim = sim
        self.world = world
        nx, ny = world["size_cells"]
        self.origin = world["origin_xy"]
        self.occ_static = decode(world["occupancy_b64"], nx, ny)
        self.walk = decode(world["walkable_b64"], nx, ny)
        self.blocked = inflate(self.occ_static, 0.22)
        self.goal_grid = reach_grid(self.blocked)
        self.landmarks = {lm["id"]: lm for lm in world["landmarks"]}
        self.wander_lms = [lm["id"] for lm in world["landmarks"] if not lm["label"].startswith("door")] \
            or list(self.landmarks)
        self.rng = np.random.default_rng(seed)
        self.robots = {}
        self.arrivals = 0
        self.demo_crosser = None      # this person walks into the robot on purpose; no repulsion from it
        self.agents = []
        for p in people["people"]:
            self.agents.append(Agent(
                id=p["id"], x=p["pos_xy"][0], y=p["pos_xy"][1], yaw=math.radians(p.get("facing_deg", 0)),
                seated_orig=(p["posture"] == "seated"), seated=(p["posture"] == "seated"),
                home=tuple(p.get("home_xy", p["pos_xy"])), activity=p.get("activity", "idle"),
                talking_to=p.get("talking_to"), personality=p.get("personality", ""),
                description=p.get("description", "person"), facing_deg=float(p.get("facing_deg", 0))))
        self.by_id = {a.id: a for a in self.agents}
        for a in self.agents:
            self.sim.set_person(a.id, a.x, a.y, a.yaw)

    # ---------------------------------------------------------- targets
    def resolve(self, target, jitter=False):
        if isinstance(target, str) and target in self.landmarks:
            c = list(self.landmarks[target]["center_xy"])
            if jitter:
                th = self.rng.uniform(0, 2 * math.pi); r = 0.6 * math.sqrt(self.rng.random())
                c = [c[0] + r * math.cos(th), c[1] + r * math.sin(th)]
            return snap_free_xy(self.goal_grid, self.origin, c[0], c[1])
        if isinstance(target, str) and target.startswith("robot_"):
            r = self.robots.get(target)
            return (r.x, r.y) if r is not None and r.active else None
        if isinstance(target, (int, np.integer)) and int(target) in self.by_id:
            a = self.by_id[int(target)]
            if jitter:   # walking to someone: stop on clear floor beside them, not inside their chair
                return snap_free_xy(self.goal_grid, self.origin, a.x, a.y)
            return (a.x, a.y)
        return None

    def plan_from(self, a, goal):
        start = (a.x, a.y) if not blocked_at(self.goal_grid, self.origin, a.x, a.y) \
            else snap_free_xy(self.goal_grid, self.origin, a.x, a.y)
        return plan_xy(self.blocked, self.origin, start, goal)

    # ---------------------------------------------------------- intents
    def apply_intent(self, a, intent, target, duration_s, thought, t, lock=False):
        a.intent, a.target, a.thought = intent, target, str(thought or "")[:24]
        a.intent_until = t + float(duration_s or 0)
        a.face_xy = None; a.follow = None
        if lock: a.locked_until = t + 30
        a.best_d = 1e9; a.best_t = t
        if intent == "go_to":
            g = self.resolve(target, jitter=True)
            if g is None:
                a.intent = "wait"; a.intent_until = t + 3; a.goal = None; a.path = []; return
            path = self.plan_from(a, g)
            if path is None:
                a.intent = "wait"; a.intent_until = t + 3; a.thought = "hmm"; a.goal = None; a.path = []; return
            a.goal = g; a.path = path; a.seated = False
        elif intent == "home":
            g = snap_free_xy(self.goal_grid, self.origin, *a.home)
            path = self.plan_from(a, g)
            if path is None:
                a.intent = "wait"; a.intent_until = t + 3; a.goal = None; a.path = []; return
            a.goal = g; a.path = path; a.seated = False
        elif intent == "face":
            a.face_xy = self.resolve(target); a.goal = None; a.path = []
        elif intent == "follow":
            a.follow = target; a.seated = False; a.goal = None; a.path = []
        else:
            a.intent = "wait"; a.goal = None; a.path = []

    def finish_intent(self, a, t):
        self.arrivals += 1
        if a.queue:
            self.apply_intent(a, *a.queue.pop(0), t)
        else:
            self.apply_intent(a, *self.sample(a, t), t)

    def sample(self, a, t):
        rng = self.rng
        pick = lambda xs: xs[int(rng.integers(len(xs)))]
        if a.seated_orig and a.seated:
            if a.activity == "typing" and rng.random() < 0.15:
                a.queue = [("wait", None, float(rng.uniform(4, 8)), "stretch"), ("home", None, 0, "back to it")]
                return ("go_to", pick(self.wander_lms), 0, "coffee")
            return ("wait", None, float(rng.uniform(8, 20)), pick(["focus", "typing", "hmm", "almost done"]))
        if a.activity == "talking" and a.talking_to is not None:
            return ("face", a.talking_to, float(rng.uniform(6, 15)), pick(["yeah", "right", "haha", "really"]))
        if a.activity == "idle":
            if rng.random() < 0.4:
                return ("go_to", pick(self.wander_lms), 0, pick(["stretch", "look around", "hmm"]))
            return ("wait", None, float(rng.uniform(5, 12)), pick(["hmm", "waiting", "bored"]))
        if a.activity == "walking":
            return ("go_to", pick(self.wander_lms), 0, "where next")
        return ("wait", None, float(rng.uniform(5, 12)), "hmm")

    # ---------------------------------------------------------- tick
    def tick(self, t, dt, robots):
        self.robots = {r.id: r for r in robots}
        active = [r for r in robots if r.active]
        for a in self.agents:
            if a.intent == "follow":
                f = self.resolve(a.follow)
                if f is not None and t - a.last_follow_plan >= 1.0:
                    a.last_follow_plan = t
                    if math.dist((a.x, a.y), f) > 1.2:
                        a.goal = f; a.path = self.plan_from(a, f) or []
                    else:
                        a.goal = None; a.path = []; a.face_xy = f
            v_des = (0.0, 0.0)
            if a.path and a.goal is not None:
                d = math.dist((a.x, a.y), a.goal)
                if d < a.best_d - 0.10:
                    a.best_d = d; a.best_t = t
                stuck = (t - a.best_t > 4.0)
                if d < 0.20 or (len(a.path) == 1 and math.dist((a.x, a.y), a.path[0]) < 0.12) or stuck:
                    a.path = []; a.goal = None
                    if a.intent == "home":
                        a.seated = a.seated_orig; a.yaw = math.radians(a.facing_deg)
                    self.finish_intent(a, t)
                else:
                    tx, ty = pursuit_target((a.x, a.y), a.path, 0.5)
                    dx, dy = tx - a.x, ty - a.y; L = math.hypot(dx, dy)
                    speed = 0.0 if a.seated else 1.2
                    v_des = (speed * dx / L, speed * dy / L) if L > 1e-6 else (0.0, 0.0)
            others = [(o.x, o.y, 0.22) for o in self.agents if o is not a] + ([] if a.id == self.demo_crosser else [(r.x, r.y, 0.35) for r in active])
            fx, fy = social_force((a.x, a.y), others, radius=0.9, k=2.0)
            if a.seated: fx = fy = 0.0
            vx, vy = v_des[0] + fx, v_des[1] + fy
            s = math.hypot(vx, vy)
            if s > 1.4: vx, vy = vx * 1.4 / s, vy * 1.4 / s
            nx_, ny_ = a.x + vx * dt, a.y + vy * dt
            # Someone standing up from a chair starts inside the inflated furniture zone. If every move
            # from a blocked cell were refused they could never step out, so only refuse moves that
            # go from free into blocked; a move that leaves a blocked cell is always allowed.
            inside = blocked_at(self.blocked, self.origin, a.x, a.y)
            if not inside:
                if blocked_at(self.blocked, self.origin, nx_, a.y): vx = 0.0; nx_ = a.x
                if blocked_at(self.blocked, self.origin, nx_, ny_): vy = 0.0; ny_ = a.y
            a.x, a.y, a.vx, a.vy = nx_, ny_, vx, vy
            if math.hypot(vx, vy) > 0.1:
                a.yaw = math.atan2(vy, vx)
            elif a.face_xy is not None:
                a.yaw = math.atan2(a.face_xy[1] - a.y, a.face_xy[0] - a.x)
            if t >= a.intent_until and not a.path and a.intent in ("wait", "face"):
                self.finish_intent(a, t)
            self.sim.set_person(a.id, a.x, a.y, a.yaw)


BRAIN_PROMPT = """You control the people in a room simulation. Each person is a simple agent. Decide what each one does next.

Landmarks (id: label at x, y meters):
{landmarks}

People (id: description | personality | seated/standing | activity | at x, y | current intent | thought):
{people}

Robots (id at x, y): {robots}
Recent events: {events}
Operator instruction still in force (applies to whoever it names; ignore for others): {command}

Return ONLY JSON:
{"intents": [{"id": 1, "intent": "go_to" | "wait" | "face" | "follow", "target": "lm_2" | 3 | "robot_1" | null, "duration_s": 5-30, "thought": "max 3 words"}]}

Rules: one entry per person. Seated people "wait" unless the operator instruction moves them. Standing people mostly "wait" or "face" the person they talk to; now and then "go_to" a landmark that fits their personality (never the door). "follow" trails a person or robot. thought: what they are thinking right now, 1-3 lowercase words, no punctuation."""


def build_brain_prompt(cast, robots, events, command):
    lms = "\n".join(f'{lm["id"]}: {lm["label"]} at {lm["center_xy"][0]:.1f}, {lm["center_xy"][1]:.1f}'
                    for lm in cast.world["landmarks"]) or "none"
    ppl = "\n".join(
        f'{a.id}: {a.description} | {a.personality} | {"seated" if a.seated else "standing"} | {a.activity} | '
        f'at {a.x:.1f}, {a.y:.1f} | {a.intent} | {a.thought}' for a in cast.agents) or "none"
    rb = ", ".join(f'{r.id} at {r.x:.1f}, {r.y:.1f}' for r in robots if r.active) or "none"
    ev = " · ".join(events[-6:]) or "none"
    return (BRAIN_PROMPT.replace("{landmarks}", lms).replace("{people}", ppl)
            .replace("{robots}", rb).replace("{events}", ev).replace("{command}", command or "none"))


def validate_intents(d, cast):
    out = []
    for it in (d or {}).get("intents", [])[:8]:
        try:
            pid = int(it["id"])
        except Exception:
            continue
        if pid not in cast.by_id: continue
        intent = it.get("intent")
        if intent not in ("go_to", "wait", "face", "follow"): continue
        target = it.get("target")
        if intent in ("go_to", "face", "follow") and cast.resolve(target) is None and target is not None:
            if intent == "go_to": continue
            target = None
        dur = it.get("duration_s", 10)
        try:
            dur = min(max(float(dur), 5.0), 30.0)
        except Exception:
            dur = 10.0
        thought = " ".join(str(it.get("thought", ""))[:40].lower().split()[:3])
        out.append((pid, intent, target, dur, thought))
    return out
