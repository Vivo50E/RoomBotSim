"""World Labs Marble client — the World API.

Docs: https://docs.worldlabs.ai/api/reference/
  Base            https://api.worldlabs.ai
  Auth header     WLT-Api-Key: <key>
  Upload          POST /marble/v1/media-assets:prepare_upload  -> {media_asset:{media_asset_id},
                                                                   upload_info:{upload_url, required_headers, upload_method}}
                  then PUT the bytes to upload_url with required_headers
  Generate        POST /marble/v1/worlds:generate              -> {operation_id, done, ...}
  Poll            GET  /marble/v1/operations/{operation_id}    -> {done, response:<World>, error}
  Get world       GET  /marble/v1/worlds/{world_id}
  Export          POST /marble/v1/worlds/{world_id}:export     {asset_type:"splats"|"mesh", format:"ply"|"glb",
                                                                resolution, mesh_variant}
                  PLY splat export returns a completed operation synchronously.

Four chest-height photos go in as a single `multi-image` world prompt with per-image azimuths and
`reconstruct_images: true`, which asks Marble to reconstruct the input views rather than invent a scene.

Env:
  WORLDLABS_API_KEY   (or MARBLE_API_KEY)
  MARBLE_MODEL        marble-1.1 (default) | marble-1.1-plus | marble-1.0 | marble-1.0-draft
  MARBLE_CROSS_CHECK  1 (default) = also generate one single-image world per photo and fuse by consensus
  MARBLE_CROSS_MODEL  marble-1.0-draft (default) — cheap model for the cross-check reconstructions
  MARBLE_TIMEOUT_S    900
"""
import os, time, json, base64, logging, threading
import numpy as np
import requests

log = logging.getLogger("marble")

BASE = os.environ.get("MARBLE_BASE_URL", "https://api.worldlabs.ai").rstrip("/")
API = BASE + "/marble/v1"
# Marble exports use OpenCV axes (+y down, +z forward). Sim wants +Z up, so pre-rotate
# (x, y, z)_marble -> (x, z, -y). floor_align refines whatever is left.
MARBLE_TO_ZUP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)


def api_key():
    return os.environ.get("WORLDLABS_API_KEY") or os.environ.get("MARBLE_API_KEY") or ""


def has_key():
    return bool(api_key())


def _h(extra=None):
    h = {"WLT-Api-Key": api_key(), "Content-Type": "application/json"}
    if extra:
        h.update(extra)
    return h


def _req(method, url, **kw):
    kw.setdefault("timeout", 60)
    r = requests.request(method, url, **kw)
    if r.status_code >= 400:
        raise RuntimeError(f"marble {method} {url.split('/marble')[-1]} -> {r.status_code}: {r.text[:400]}")
    return r


# ------------------------------------------------------------------ media assets

def upload_image(path):
    """Upload one JPEG, return its media_asset_id."""
    name = os.path.basename(path)
    ext = os.path.splitext(name)[1].lstrip(".").lower() or "jpg"
    body = {"file_name": name, "kind": "image", "extension": ext}
    j = _req("POST", f"{API}/media-assets:prepare_upload", headers=_h(), json=body).json()
    aid = j["media_asset"]["media_asset_id"]
    up = j["upload_info"]
    hdr = dict(up.get("required_headers") or {})
    hdr.setdefault("Content-Type", "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}")
    method = (up.get("upload_method") or "PUT").upper()
    with open(path, "rb") as f:
        r = requests.request(method, up["upload_url"], data=f.read(), headers=hdr, timeout=180)
    if r.status_code >= 400:
        raise RuntimeError(f"marble upload failed {r.status_code}: {r.text[:200]}")
    log.info("marble: uploaded %s -> %s", name, aid)
    return aid


def _content(asset_id=None, path=None):
    if asset_id:
        return {"source": "media_asset", "media_asset_id": asset_id}
    ext = os.path.splitext(path)[1].lstrip(".").lower() or "jpg"
    return {"source": "data_base64", "data_base64": base64.b64encode(open(path, "rb").read()).decode(), "extension": ext}


# ------------------------------------------------------------------ generate / poll

def generate_multi_image(asset_ids, azimuths=None, model=None, seed=None, display_name="RoomBotSim room", text_prompt=None):
    """One world from up to 4 photos, reconstructing the input views."""
    n = len(asset_ids)
    if azimuths is None:
        azimuths = [round(i * 360.0 / max(n, 1), 1) for i in range(n)]
    prompt = {
        "type": "multi-image",
        "multi_image_prompt": [{"azimuth": float(az), "content": _content(asset_id=a)} for a, az in zip(asset_ids, azimuths)],
        "reconstruct_images": True,
    }
    if text_prompt:
        prompt["text_prompt"] = text_prompt
    body = {"world_prompt": prompt, "model": model or os.environ.get("MARBLE_MODEL", "marble-1.1"),
            "display_name": display_name[:64], "permission": {"public": False}}
    if seed is not None:
        body["seed"] = int(seed)
    return _req("POST", f"{API}/worlds:generate", headers=_h(), json=body).json()


def generate_single_image(asset_id, model=None, seed=None, display_name="RoomBotSim view"):
    # is_pano takes the JSON literals 'auto', true or false. Sending the string "false" is a 422.
    body = {"world_prompt": {"type": "image", "image_prompt": _content(asset_id=asset_id), "is_pano": False},
            "model": model or os.environ.get("MARBLE_CROSS_MODEL", "marble-1.0-draft"),
            "display_name": display_name[:64], "permission": {"public": False}}
    if seed is not None:
        body["seed"] = int(seed)
    return _req("POST", f"{API}/worlds:generate", headers=_h(), json=body).json()


def wait(op, timeout_s=None, poll_s=5.0, on_progress=None):
    """Poll an operation to completion; returns its `response` payload."""
    timeout_s = timeout_s or float(os.environ.get("MARBLE_TIMEOUT_S", "900"))
    oid = op["operation_id"]
    t0 = time.time()
    while not op.get("done"):
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"marble operation {oid} timed out after {timeout_s:.0f}s")
        time.sleep(poll_s)
        op = _req("GET", f"{API}/operations/{oid}", headers=_h()).json()
        if on_progress:
            try:
                on_progress(op.get("metadata") or {})
            except Exception:
                pass
    if op.get("error"):
        e = op["error"]
        raise RuntimeError(f"marble operation failed [{e.get('code')}]: {e.get('message')}")
    return op.get("response") or {}


def get_world(world_id):
    return _req("GET", f"{API}/worlds/{world_id}", headers=_h()).json()


def export(world_id, asset_type, fmt, resolution=None, mesh_variant=None, timeout_s=600):
    """Export and return a download URL. PLY splats come back already done."""
    body = {"asset_type": asset_type, "format": fmt}
    if resolution:
        body["resolution"] = resolution
    if mesh_variant:
        body["mesh_variant"] = mesh_variant
    op = _req("POST", f"{API}/worlds/{world_id}:export", headers=_h(), json=body).json()
    res = wait(op, timeout_s=timeout_s, poll_s=3.0)
    url = res.get("url")
    if not url:
        raise RuntimeError(f"export returned no url: {json.dumps(res)[:300]}")
    return url


def download(url, path):
    r = requests.get(url, timeout=600, stream=True)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    return path


def world_of(response):
    """The generate operation's response payload -> the World dict."""
    if not isinstance(response, dict):
        raise RuntimeError("marble: unexpected operation response")
    return response.get("world") or response


def splat_url_of(world):
    sp = ((world.get("assets") or {}).get("splats") or {}).get("spz_urls") or {}
    if isinstance(sp, dict) and sp:
        for k in ("full_res", "high", "default"):
            if k in sp:
                return sp[k]
        return list(sp.values())[0]
    return None


def semantics_of(world):
    sm = ((world.get("assets") or {}).get("splats") or {}).get("semantics_metadata") or {}
    return float(sm.get("metric_scale_factor") or 1.0), float(sm.get("ground_plane_offset") or 0.0)


def fetch_scene(world_id, out_dir, want_mesh=None, tag=""):
    """Export + download PLY splats (and optionally the collider mesh) for one world.

    Splat PLY export is synchronous and returns in a few seconds. The GLB mesh export is a different,
    much slower service: measured 5 September, a mesh export on a completed draft world was still
    IN_PROGRESS after 12 minutes with its updated_at never moving off creation. It is only used as an
    optional second opinion in fusion.mesh_filter, so it is off unless MARBLE_MESH=1, and capped when on.
    """
    if want_mesh is None:
        want_mesh = os.environ.get("MARBLE_MESH", "0") == "1"
    os.makedirs(out_dir, exist_ok=True)
    pre = f"{tag}_" if tag else ""
    out = {}
    url = export(world_id, "splats", "ply")
    out["ply"] = download(url, os.path.join(out_dir, f"{pre}scene.ply"))
    try:
        spz = splat_url_of(get_world(world_id))
        if spz:
            out["spz"] = download(spz, os.path.join(out_dir, f"{pre}scene.spz"))
    except Exception as e:
        log.warning("marble: spz download skipped (%s)", e)
    if want_mesh:
        try:
            murl = export(world_id, "mesh", "glb", resolution="150k", mesh_variant="vertex_colored",
                          timeout_s=float(os.environ.get("MARBLE_MESH_TIMEOUT_S", "180")))
            out["mesh"] = download(murl, os.path.join(out_dir, f"{pre}collider.glb"))
        except Exception as e:
            log.warning("marble: mesh export skipped (%s)", e)
    return out


# ------------------------------------------------------------------ whole-room reconstruction

def reconstruct_room(image_paths, job_dir, status=None, cross_check=None):
    """Photos -> {"primary": {...}, "extras": [{...}], "world": <World>}.

    primary/extras entries: {"world_id", "ply", "spz"?, "mesh"?, "scale", "ground_offset"}
    The primary comes from one multi-image world over all photos. When cross-checking is on, one
    single-image world per photo is generated in parallel and fused by consensus (see fusion.py).
    """
    if not has_key():
        raise RuntimeError("WORLDLABS_API_KEY not set")
    out = os.path.join(job_dir, "marble")
    os.makedirs(out, exist_ok=True)
    if cross_check is None:
        cross_check = os.environ.get("MARBLE_CROSS_CHECK", "1") == "1"

    def note(msg):
        log.info("marble: %s", msg)
        if status is not None:
            status["message"] = msg

    note(f"uploading {len(image_paths)} photos")
    ids = [upload_image(p) for p in image_paths]

    note("generating room from all views")
    prim_op = generate_multi_image(ids, display_name=os.path.basename(job_dir))
    cross_ops = []
    if cross_check and len(ids) > 1:
        for i, a in enumerate(ids):
            try:
                cross_ops.append((f"v{i}", generate_single_image(a, display_name=f"view {i}")))
            except Exception as e:
                log.warning("marble: cross view %d not started (%s)", i, e)

    world = world_of(wait(prim_op, on_progress=lambda m: note(f"building room {m.get('progress', '')}".strip())))
    wid = world.get("world_id")
    note("downloading splats")
    scale, ground = semantics_of(world)
    primary = dict(world_id=wid, scale=scale, ground_offset=ground, **fetch_scene(wid, out, tag="primary"))

    extras = []
    if cross_ops:
        note("downloading cross-check views")
        results = {}

        def grab(tag, op):
            try:
                w = world_of(wait(op, timeout_s=600))
                s, g = semantics_of(w)
                results[tag] = dict(world_id=w.get("world_id"), scale=s, ground_offset=g,
                                    **fetch_scene(w.get("world_id"), out, want_mesh=False, tag=tag))
            except Exception as e:
                log.warning("marble: cross view %s failed (%s)", tag, e)

        ths = [threading.Thread(target=grab, args=(t, o), daemon=True) for t, o in cross_ops]
        for t in ths:
            t.start()
        for t in ths:
            t.join(timeout=700)
        extras = [results[k] for k in sorted(results)]
    json.dump(dict(primary=primary, extras=extras, world=world), open(os.path.join(out, "marble.json"), "w"), indent=1)
    return dict(primary=primary, extras=extras, world=world)
