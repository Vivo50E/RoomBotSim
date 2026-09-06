"""The decision policy. Skill level, sequential: observation JSON in, one action JSON out.

kinds:
  scripted  built-in oracle (also the relabeler for SFT)
  chat      any OpenAI-compatible /chat/completions endpoint (OpenRouter qwen3.7-flash, DeepSeek,
            vLLM, your fine-tuned model). Set url/model/key.
  raw       your own endpoint: POST observation -> action JSON
  human     the runtime never calls act(); the operator drives
"""
import os, json, math, logging, requests
from openrouter import parse_json

log = logging.getLogger("policy")

SKILL_SYSTEM = """You are the decision policy of a home robot in a simulated room. You receive a JSON observation and must reply with ONE action as JSON only, no prose.

Actions:
- {"action":"navigate_to","target":<landmark id | object id | person id>}   move next to the target and face it
- {"action":"pick","object":<object id>}       requires: robot next to and facing the object (within 0.95 m), hands free
- {"action":"place","object":<object id>,"target":<landmark id | person id>}   put a held object on a surface or hand it to a person you are next to (<= 1 m)
- {"action":"pour","target":"cup"}             requires: holding the pot, cup on a surface within 0.95 m, facing it
- {"action":"say","text":"..."}
- {"action":"done"}                             when the task is complete

Always navigate to something before picking, pouring, or placing. Check "last_result": if the last action failed, fix the cause (usually: move closer, face the object, or put something down first). Reply with the JSON object only."""

VALID = {"navigate_to", "pick", "place", "pour", "say", "done"}

# ------------------------------------------------------------------ pixel-in policy
# kind "vlm": the model sees the robot's camera frame plus a state JSON with every coordinate removed.
# It must decide from pixels whether the target is in reach, which side the person is on, whether the
# cup is upright. The ids are kept so navigate_to/pick/place can name things; the geometry is not.
VLM_ADDENDUM = """

You are a PIXEL-IN policy: the attached image is the robot's forward camera at this instant. The JSON has NO positions or distances. Use the image to decide: is the target object visible and within arm's reach (large, centred, lower half of the frame)? Is a person blocking the way? If the target is not visible or looks far, navigate_to it first. Never guess a pick/pour when the object is not clearly close and centred. Reply with the JSON object only."""

STRIP = {"x", "y", "z", "yaw", "center_xy", "size_xy", "pos_xy", "dist", "distance", "d", "height_m", "path", "goal"}


def pixel_obs(obs):
    """Observation with geometry removed. What a robot with a camera and a symbolic state tracker knows."""
    def scrub(v):
        if isinstance(v, dict):
            return {k: scrub(x) for k, x in v.items() if k not in STRIP}
        if isinstance(v, list):
            return [scrub(x) for x in v]
        return v
    o = scrub({k: v for k, v in obs.items() if k not in ("robot_view", "_frame_path")})
    return o


def encode_image(path):
    import base64
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


def validate(a):
    """-> (action, error_kind|None)"""
    if not isinstance(a, dict):
        return None, "bad_json"
    act = a.get("action")
    if act not in VALID:
        return None, "bad_action"
    out = {"action": act, "target": a.get("target"), "object": a.get("object"), "text": a.get("text")}
    if act == "pick" and out["object"] in (None, ""):
        if isinstance(out["target"], str):
            out["object"] = out["target"]
        else:
            return None, "missing_field"
    if act == "place":
        if out["object"] in (None, "") or out["target"] in (None, ""):
            return None, "missing_field"
    if act == "pour" and out["target"] in (None, ""):
        out["target"] = "cup"
    if act == "navigate_to" and out["target"] in (None, ""):
        return None, "missing_field"
    if act == "say" and not out["text"]:
        out["text"] = "ok"
    return out, None


class Policy:
    def __init__(self, cfg):
        self.cfg = dict(cfg or {})
        self.cfg.setdefault("kind", os.environ.get("POLICY_KIND", "scripted"))

    @property
    def kind(self):
        return self.cfg["kind"]

    def act(self, obs):
        k = self.cfg["kind"]
        if k == "scripted":
            return oracle(obs)
        key = self.cfg.get("key") or os.environ.get("POLICY_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
        url = self.cfg.get("url") or os.environ.get("POLICY_URL", "https://openrouter.ai/api/v1/chat/completions")
        h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        if k == "chat":
            body = {"model": self.cfg.get("model") or os.environ.get("POLICY_MODEL", "qwen/qwen3.7-flash"),
                    "temperature": 0, "max_tokens": 200,
                    "messages": [{"role": "system", "content": SKILL_SYSTEM},
                                 {"role": "user", "content": json.dumps(obs)}],
                    "response_format": {"type": "json_object"}}
            if "openrouter.ai" in url:
                body["reasoning"] = {"enabled": False}
            r = requests.post(url, headers=h, json=body, timeout=30)
            if r.status_code == 400:
                body.pop("response_format", None); body.pop("reasoning", None)
                r = requests.post(url, headers=h, json=body, timeout=30)
            r.raise_for_status()
            return parse_json(r.json()["choices"][0]["message"]["content"])
        if k == "vlm":
            frame = obs.get("_frame_path")
            content = [{"type": "text", "text": json.dumps(pixel_obs(obs))}]
            if frame and os.path.exists(frame):
                content.append({"type": "image_url", "image_url": {"url": encode_image(frame)}})
            else:
                log.warning("vlm policy: no frame for this step, deciding from state only")
            body = {"model": self.cfg.get("model") or os.environ.get("VLM_MODEL", "qwen/qwen3.7-flash"),
                    "temperature": 0, "max_tokens": 200,
                    "messages": [{"role": "system", "content": SKILL_SYSTEM + VLM_ADDENDUM},
                                 {"role": "user", "content": content}]}
            if "openrouter.ai" in url:
                body["reasoning"] = {"enabled": False}
            import time as _t
            r = None
            for attempt in range(4):                      # 429/5xx from the provider are not policy mistakes
                r = requests.post(url, headers=h, json=body, timeout=60)
                if r.status_code == 400:
                    body.pop("reasoning", None)
                    r = requests.post(url, headers=h, json=body, timeout=60)
                if r.status_code < 400:
                    break
                log.warning("vlm policy: HTTP %s %s (attempt %d)", r.status_code, r.text[:160], attempt + 1)
                _t.sleep(1.5 * (attempt + 1))
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
            log.info("vlm policy said: %s", txt[:200])
            return parse_json(txt)
        if k == "raw":
            r = requests.post(url, headers=h, json=obs, timeout=30)
            r.raise_for_status()
            return r.json()
        raise ValueError(k)


# ------------------------------------------------------------------ scripted oracle

def _near(obs, xy, r=0.9):
    return math.hypot(obs["robot"]["x"] - xy[0], obs["robot"]["y"] - xy[1]) <= r


def oracle(obs):
    """Stateless plan for 'coffee to requester'. Works from any state, so it can relabel any step."""
    objs = {o["id"]: o for o in obs.get("objects", [])}
    if "cup" not in objs or "pot" not in objs:
        return {"action": "done"}
    cup, pot = objs["cup"], objs["pot"]
    hold = obs["robot"].get("holding")
    req_id = obs.get("requester")
    req = next((p for p in obs.get("people", []) if p["id"] == req_id), None)
    lms = obs.get("landmarks") or []
    counter = next((l for l in lms if str(l.get("label", "")).startswith("counter")), lms[0] if lms else None)
    lr = obs.get("last_result") or {}
    la = obs.get("last_action") or {}

    def go(t): return {"action": "navigate_to", "target": t}
    arrived = lambda tgt: la.get("action") == "navigate_to" and la.get("target") == tgt and lr.get("ok")

    if cup.get("held_by") == f"person_{req_id}" and cup.get("filled"):
        return {"action": "done"}
    if not cup.get("filled") and not obs.get("spilled"):
        if hold != "pot":
            if hold:
                if counter and _near(obs, (counter["x"], counter["y"]), 1.2):
                    return {"action": "place", "object": hold, "target": counter["id"]}
                return go(counter["id"] if counter else "pot")
            if arrived("pot"):
                return {"action": "pick", "object": "pot"}
            return go("pot")
        if arrived("cup"):
            return {"action": "pour", "target": "cup"}
        return go("cup")
    if obs.get("spilled") and not cup.get("filled"):
        return {"action": "say", "text": "I spilled it, sorry."} if la.get("action") != "say" else {"action": "done"}
    if hold == "pot":
        if counter and arrived(counter["id"]):
            return {"action": "place", "object": "pot", "target": counter["id"]}
        return go(counter["id"] if counter else "cup")
    if hold != "cup":
        if arrived("cup"):
            return {"action": "pick", "object": "cup"}
        return go("cup")
    if req and arrived(req["id"]):
        return {"action": "place", "object": "cup", "target": req["id"]}
    return go(req["id"] if req else (counter["id"] if counter else "cup"))
