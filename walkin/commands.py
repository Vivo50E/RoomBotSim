"""Operator sentence -> one command. LLM parse first, deterministic keyword parse as the fallback."""
import re, logging
from openrouter import ask_json, brain_model, has_key

log = logging.getLogger("commands")

PARSE_PROMPT = """Convert the operator's sentence into one command for a room simulation.

People: {people}
Robots: {robots}. "the robot" with no number means robot_1.
Landmarks: {landmarks}

Sentence: "{text}"

Return ONLY JSON:
{"ok": true, "targets": ["all"] or ["robots"] or ["robot_1"] or [2, 4], "intent": "go_to" | "wait" | "face" | "follow" | "home", "target": "lm_3" or 2 or "robot_1" or null}
"everyone"/"everybody" means all people (not robots) -> targets ["all"]. "sit back down" / "go back" / "reset" / "return" -> intent "home", target null.
If the sentence cannot be mapped, return {"ok": false}."""


def llm_parse(text, people, robots, landmarks):
    if not has_key():
        return None
    try:
        p = "; ".join(f"{i}: {d}" for i, d in people.items()) or "none"
        r = ", ".join(robots) or "none"
        l = "; ".join(f"{i}: {lab}" for i, lab in landmarks.items()) or "none"
        prompt = (PARSE_PROMPT.replace("{people}", p).replace("{robots}", r)
                  .replace("{landmarks}", l).replace("{text}", text.replace('"', "'")))
        d = ask_json(brain_model(), prompt, max_tokens=160, timeout=15)
        if not d.get("ok"):
            return None
        if d.get("intent") not in ("go_to", "wait", "face", "follow", "home"):
            return None
        if not isinstance(d.get("targets"), list) or not d["targets"]:
            return None
        return d
    except Exception as e:
        log.info("llm command parse failed: %s", e)
        return None


def keyword_parse(text, people, robots, landmarks):
    s = text.lower()
    targets = []
    if re.search(r"\b(everyone|everybody|all)\b", s):
        targets = ["all"]
    m = re.findall(r"\brobot ?(\d)\b", s)
    if m:
        targets += [f"robot_{k}" for k in m]
    elif re.search(r"\brobots?\b", s) and not re.search(r"follow the robot|to the robot|at the robot", s):
        targets += ["robot_1"]
    nums = [int(n) for n in re.findall(r"\b(\d)\b", s) if int(n) in people]
    if not targets:
        targets = nums or []
    if re.search(r"\b(sit|back down|go back|home|reset|return)\b", s):
        intent, target = "home", None
    elif re.search(r"\bfollow\b", s):
        intent = "follow"
        tail = s.split("follow", 1)[1]
        target = "robot_1" if "robot" in tail else (nums[-1] if nums else None)
    elif re.search(r"\b(wait|stop|stay|freeze)\b", s):
        intent, target = "wait", None
    elif re.search(r"\b(face|look at|turn to)\b", s):
        intent, target = "face", ("robot_1" if "robot" in s else (nums[-1] if nums else None))
    else:
        intent = "go_to"
        best = None
        for lm_id, label in landmarks.items():
            if label != "unknown" and label in s and (best is None or len(label) > len(landmarks[best])):
                best = lm_id
        target = best
        if target is None and ("robot" in s and targets != ["robot_1"]):
            target = "robot_1"
    if not targets or (intent == "go_to" and target is None) or (intent in ("follow", "face") and target is None):
        return {"ok": False}
    return {"ok": True, "targets": targets, "intent": intent, "target": target}


def parse(text, people, robots, landmarks):
    return llm_parse(text, people, robots, landmarks) or keyword_parse(text, people, robots, landmarks)
