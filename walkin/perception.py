"""VLM calls: who is in the room, what the landmarks are, and a layout estimate when reconstruction
gives no camera poses. Every call degrades to a documented fallback instead of raising."""
import os, math, logging
from openrouter import ask_json, vlm_model, has_key

log = logging.getLogger("perception")

PEOPLE_PROMPT = """You are given {n} photos of the same room, taken from {n} different spots at chest height. They are photo {names} in that order. Some people are in the room.

Return ONLY a JSON object, no prose, exactly this shape:
{
  "people": [
    {
      "id": 1,
      "boxes": {"A": [u0, v0, u1, v1], "B": null},
      "description": "short non-identifying description, e.g. 'person in a grey hoodie with a laptop'",
      "posture": "seated" or "standing",
      "activity": "typing" or "talking" or "idle" or "walking",
      "talking_to": id of the person they are talking to, or null,
      "personality": "one short line, e.g. 'restless, curious about gadgets'"
    }
  ],
  "fixtures": [
    {"label": "door" or "whiteboard" or "window" or "tv", "photo": "A", "bbox": [u0, v0, u1, v1]}
  ]
}

Rules:
- Coordinates are normalized to [0, 1]: u is horizontal from the left edge, v is vertical from the top edge. u0 < u1, v0 < v1. Box the whole body from head to feet, or head to seat if the legs are hidden.
- "boxes" has one key per photo the person is visible in, using the photo letters above. Omit photos where they are not visible.
- The same person appears in several photos. Match them by clothing and position. List each person exactly once.
- At most 6 people. Ignore reflections, posters, and people on screens.
- description must not include names, ethnicity, age, or anything identifying beyond clothing and objects held.
- fixtures: every door, whiteboard, window and TV you can see, each listed once, in whichever photo shows it best. Empty list if none.
"""

LABELS_PROMPT = """Image 1 is a top-down occupancy map of a room. Dark cells are obstacles. Red numbered rectangles are candidate landmarks. The remaining images are the photos of the same room, in order {names}.

For each numbered rectangle decide what object it is. Return ONLY JSON:
{"labels": {"0": "table", "1": "couch"}}
Allowed labels: table, desk, chair, couch, counter, shelf, cabinet, plant, wall, unknown.
Use "wall" for long thin rectangles along the room edge. Use "unknown" if unsure. One entry per number.
"""

FALLBACK_LAYOUT_PROMPT = """{n} photos of the same room from {n} spots at chest height, in order {names}. Estimate the room layout in meters.

Return ONLY JSON:
{
  "room_size_xy": [width_m, depth_m],
  "cameras": [{"photo": "A", "pos_xy": [x, y], "yaw_deg": d}],
  "landmarks": [{"label": "table", "center_xy": [x, y], "size_xy": [sx, sy], "height_m": h}],
  "people": [{"id": 1, "pos_xy": [x, y], "facing_deg": d}]
}
Coordinate frame: origin at the center of the room floor. +x is to the right and +y is forward as seen from photo A. yaw_deg is the direction faced, 0 = +x, counter-clockwise positive. Give one camera entry per photo.
Reference sizes: door 0.9 wide x 2.0 tall, table 1.8 x 0.9 x 0.75, desk 1.4 x 0.7 x 0.75, chair 0.5 x 0.5 x 0.9, couch 2.0 x 0.9 x 0.8, counter 0.6 deep x 0.9 tall, ceiling 2.7, standing adult 1.7.
Allowed landmark labels: table, desk, chair, couch, counter, shelf, cabinet, plant, door, whiteboard, window, tv. Give door, whiteboard, window and tv as thin boxes against a wall: door 0.9 x 0.2 with height_m 0; whiteboard/window/tv 1.5 x 0.2 with height_m 0.
The people are, in this order with these ids: {people_list}
Every listed person must appear once in "people".
"""


def _valid_bbox(b, wh=None):
    """The prompt asks for normalized [0,1] boxes, but vision models routinely answer in pixels.
    Clamping those to [0,1] collapses every box to zero width and silently empties the room, so detect
    pixel space and divide by the raster instead."""
    try:
        u0, v0, u1, v1 = [float(x) for x in b]
    except Exception:
        return None
    mx = max(abs(u0), abs(v0), abs(u1), abs(v1))
    if mx > 1.5:
        # Qwen-VL family answers on a 0-1000 grid regardless of raster; other models answer in pixels.
        # Dividing a 1000-grid box by a 1024x768 raster shifted every box onto the person's feet.
        if mx <= 1000.0:
            w = h = 1000.0
        else:
            w, h = wh if wh else (1024.0, 1024.0)
        u0, u1 = u0 / w, u1 / w
        v0, v1 = v0 / h, v1 / h
    u0, v0, u1, v1 = [min(max(v, 0.0), 1.0) for v in (u0, v0, u1, v1)]
    if u1 < u0: u0, u1 = u1, u0
    if v1 < v0: v0, v1 = v1, v0
    if u1 - u0 < 0.02 or v1 - v0 < 0.02: return None
    return [u0, v0, u1, v1]


def _names(photos):
    return ", ".join(photos)


def detect_people(photo_paths, photos):
    """photo_paths in the same order as `photos` (e.g. ["A","B","C","D"]). Returns (people, fixtures)."""
    from PIL import Image
    sizes = {}
    for name, path in zip(photos, photo_paths):
        try:
            with Image.open(path) as im:
                sizes[name] = im.size
        except Exception:
            sizes[name] = (1024.0, 1024.0)
    prompt = PEOPLE_PROMPT.replace("{n}", str(len(photos))).replace("{names}", _names(photos))
    # seven photos x six people x a per-photo box dict is ~4k tokens of JSON; 1600 truncated it mid-object
    try:
        d = ask_json(vlm_model(), prompt, photo_paths, max_tokens=4000)
    except Exception as e:
        log.warning("people call failed once (%s), retrying", str(e)[:80])
        d = ask_json(vlm_model(), prompt, photo_paths, max_tokens=4000)
    raw = d.get("people", [])[:6]
    old_to_new = {}
    people = []
    for k, p in enumerate(raw):
        boxes = {}
        src = p.get("boxes") if isinstance(p.get("boxes"), dict) else {}
        for ph in photos:
            wh = sizes.get(ph)
            b = _valid_bbox(src.get(ph), wh) if src else None
            if b is None:
                b = _valid_bbox(p.get(f"bbox_{ph}"), wh)
            if b: boxes[ph] = b
        if not boxes: continue
        old_to_new[p.get("id")] = len(people) + 1
        people.append(dict(id=len(people) + 1, boxes=boxes,
                           description=str(p.get("description", "person"))[:80],
                           posture="seated" if p.get("posture") == "seated" else "standing",
                           activity=p.get("activity") if p.get("activity") in ("typing", "talking", "idle", "walking") else "idle",
                           talking_to=p.get("talking_to"),
                           personality=str(p.get("personality", "easygoing"))[:80]))
    ids = {p["id"] for p in people}
    for p in people:
        t = old_to_new.get(p["talking_to"], p["talking_to"])
        p["talking_to"] = t if isinstance(t, int) and t in ids and t != p["id"] else None
    fixtures = []
    for f in d.get("fixtures", []):
        b = _valid_bbox(f.get("bbox"), sizes.get(f.get("photo")))
        if b and f.get("label") in ("door", "whiteboard", "window", "tv") and f.get("photo") in photos:
            fixtures.append(dict(label=f["label"], photo=f["photo"], bbox=b))
    return people, fixtures


def label_landmarks(topdown_png, photo_paths, photos, landmarks):
    labels = {}
    if has_key():
        try:
            prompt = LABELS_PROMPT.replace("{names}", _names(photos))
            d = ask_json(vlm_model(), prompt, [topdown_png] + list(photo_paths), max_tokens=400)
            labels = d.get("labels", {}) or {}
        except Exception as e:
            log.warning("label call failed: %s", e)
    allowed = {"table", "desk", "chair", "couch", "counter", "shelf", "cabinet", "plant", "wall", "unknown"}
    for k, lm in enumerate(landmarks):
        lab = str(labels.get(str(k), "unknown")).lower().strip()
        lm["label"] = lab if lab in allowed else "unknown"
    return [lm for lm in landmarks if lm["label"] != "wall"]


def fallback_layout(photo_paths, photos, people):
    plist = "; ".join(f'{p["id"]}: {p["description"]} ({p["posture"]})' for p in people) or "none"
    prompt = (FALLBACK_LAYOUT_PROMPT.replace("{n}", str(len(photos))).replace("{names}", _names(photos))
              .replace("{people_list}", plist))
    d = ask_json(vlm_model(), prompt, photo_paths, max_tokens=1400)
    W, H = (d.get("room_size_xy") or [6, 8])[:2]
    W = min(max(float(W), 2.5), 20); H = min(max(float(H), 2.5), 20)
    lms = []
    for lm in d.get("landmarks", [])[:14]:
        try:
            cx, cy = lm["center_xy"]; sx, sy = lm["size_xy"]
            lms.append(dict(label=str(lm.get("label", "unknown")), center_xy=[float(cx), float(cy)],
                            size_xy=[max(float(sx), 0.2), max(float(sy), 0.2)],
                            height_m=max(float(lm.get("height_m", 0.75)), 0.0)))
        except Exception:
            continue
    ppl = {}
    for p in d.get("people", []):
        try:
            ppl[int(p["id"])] = ([float(p["pos_xy"][0]), float(p["pos_xy"][1])], float(p.get("facing_deg", 0)))
        except Exception:
            continue
    cams = {}
    for c in d.get("cameras", []):
        if c.get("photo") in photos:
            try:
                cams[c["photo"]] = ([float(c["pos_xy"][0]), float(c["pos_xy"][1])], float(c.get("yaw_deg", 90)))
            except Exception:
                continue
    return dict(size_xy=[W, H], landmarks=lms, people=ppl, cameras=cams)


def synth_camera(photo, pos_xy, yaw_deg, w_px, h_px, z=1.4):
    """Fallback camera -> recon.Camera. 70 degree horizontal FOV."""
    import numpy as np
    from recon import Camera, K_for
    yaw = math.radians(yaw_deg)
    f = np.array([math.cos(yaw), math.sin(yaw), 0.0]); d = np.array([0, 0, -1.0]); r = np.cross(d, f)
    R = np.stack([r, d, f], 1)
    fx = w_px / (2 * math.tan(math.radians(35)))
    return Camera(photo, w_px, h_px, K_for(fx, fx, w_px / 2, h_px / 2), R, np.array([pos_xy[0], pos_xy[1], z]))
