"""Episode config, evaluators, failure tags and the JSONL recorder."""
import os, json, math, uuid, datetime, logging

log = logging.getLogger("episodes")

DEFAULT_TASK_TEXT = {
    "coffee_to_person": "get me a cup of coffee and bring it to me",
    "bring_object": "bring me the cup",
    "go_to": "go to the table",
}


def new_id():
    return uuid.uuid4().hex[:6]


def default_cfg(**kw):
    cfg = dict(task_type="coffee_to_person", task_text=None, requester=None, object="cup", target=None,
               robot_type="spot", policy={"kind": "scripted"}, seed=7, humans="scripted",
               max_steps=14, max_seconds=300, replay_of=None)
    cfg.update({k: v for k, v in kw.items() if v is not None})
    if not cfg.get("task_text"):
        cfg["task_text"] = DEFAULT_TASK_TEXT.get(cfg["task_type"], "do the task")
    return cfg


class Episode:
    def __init__(self, cfg, job_id, robot_k, t0, job_dir):
        self.id = new_id()
        self.cfg = cfg
        self.job_id = job_id
        self.robot = robot_k
        self.t0 = t0
        self.step = 0
        self.tags = set()
        self.steps = []
        self.success = None
        self.done = False
        self.last_action = None
        self.last_result = None
        self.policy_errors = 0
        self.consecutive_fails = 0
        d = os.path.join(job_dir, "episodes")
        os.makedirs(d, exist_ok=True)
        self.path = os.path.join(d, f"{self.id}.jsonl")
        self._w = open(self.path, "a", buffering=1)
        self._write({"type": "header", "episode_id": self.id, "job_id": job_id, "cfg": _safe_cfg(cfg),
                     "t_wall": datetime.datetime.now().isoformat(timespec="seconds")})

    def _write(self, obj):
        self._w.write(json.dumps(obj) + "\n")

    def record_step(self, obs, action, result, events, duration_s, frame=None, teleop=None):
        self.step += 1
        row = {"type": "step", "step": self.step, "t": round(obs.get("t", 0.0), 2), "obs": obs,
               "action": action, "result": result, "events": events, "duration_s": round(duration_s, 2),
               "frame": frame, "teleop": teleop}
        self.steps.append(row)
        self._write(row)
        self.last_action = action
        self.last_result = result
        if result.get("ok"):
            self.consecutive_fails = 0
        else:
            self.consecutive_fails += 1
            self.tags.add(f"precondition:{result.get('reason')}" if result.get("reason") in PRECONDITIONS
                          else str(result.get("reason")))

    def close(self, success, duration_s, final):
        self.success = bool(success)
        self.done = True
        self._write({"type": "footer", "success": bool(success), "tags": sorted(self.tags),
                     "n_steps": self.step, "duration_s": round(duration_s, 2), "final": final})
        try:
            self._w.close()
        except Exception:
            pass


PRECONDITIONS = {"too_far", "too_close", "not_facing", "holding_other", "not_holding", "cup_on_floor",
                 "cup_tipped", "moving", "unknown_object", "unknown_target", "object_held",
                 "not_holding_pot", "cup_held", "already_full"}


def _safe_cfg(cfg):
    c = json.loads(json.dumps(cfg, default=str))
    if isinstance(c.get("policy"), dict) and c["policy"].get("key"):
        c["policy"]["key"] = "***"
    return c


def evaluate(cfg, state):
    """state: {"objects": {...}, "robot": {...}, "people": {...}, "landmarks": {...}, "held": {...}}"""
    tt = cfg["task_type"]
    objs = state["objects"]
    if tt == "coffee_to_person":
        cup = objs.get("cup")
        if not cup or not cup.get("filled"):
            return False, ({"delivered_empty"} if cup and str(cup.get("held_by") or "").startswith("person_") else set())
        if cup.get("held_by") == f"person_{cfg.get('requester')}":
            return True, set()
        if str(cup.get("held_by") or "").startswith("person_"):
            return False, {"wrong_person"}
        req = state["people"].get(cfg.get("requester"))
        if (req and cup.get("upright", 1) > 0.9 and not cup.get("held_by")
                and math.dist((cup["x"], cup["y"]), (req["x"], req["y"])) <= 1.0):
            return True, set()
        return False, set()
    if tt == "bring_object":
        oid = cfg.get("object", "cup"); o = objs.get(oid)
        if not o:
            return False, set()
        tgt = cfg.get("target")
        if o.get("held_by") == f"person_{tgt}":
            return True, set()
        if o.get("on") == tgt and o.get("upright", 1) > 0.9:
            return True, set()
        return False, set()
    if tt == "go_to":
        r = state["robot"]; tgt = cfg.get("target")
        lm = state["landmarks"].get(tgt)
        if lm:
            dx = max(abs(r["x"] - lm["center_xy"][0]) - lm["size_xy"][0] / 2, 0.0)
            dy = max(abs(r["y"] - lm["center_xy"][1]) - lm["size_xy"][1] / 2, 0.0)
            return math.hypot(dx, dy) <= 0.9, set()
        p = state["people"].get(tgt)
        if p:
            return math.dist((r["x"], r["y"]), (p["x"], p["y"])) <= 0.9, set()
    return False, set()


def list_episodes(job_dir):
    d = os.path.join(job_dir, "episodes")
    out = []
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".jsonl"):
            continue
        try:
            lines = [json.loads(l) for l in open(os.path.join(d, fn)) if l.strip()]
        except Exception:
            continue
        if not lines:
            continue
        head = lines[0]
        foot = next((l for l in lines if l.get("type") == "footer"), None)
        cfg = head.get("cfg", {})
        out.append({"episode_id": head.get("episode_id"), "task_type": cfg.get("task_type"),
                    "task_text": cfg.get("task_text"),
                    "policy_kind": (cfg.get("policy") or {}).get("kind"),
                    "robot_type": cfg.get("robot_type"), "seed": cfg.get("seed"),
                    "replay_of": cfg.get("replay_of"),
                    "success": foot.get("success") if foot else None,
                    "tags": foot.get("tags") if foot else [],
                    "n_steps": foot.get("n_steps") if foot else sum(1 for l in lines if l.get("type") == "step"),
                    "t_wall": head.get("t_wall")})
    out.sort(key=lambda e: e.get("t_wall") or "", reverse=True)
    return out
