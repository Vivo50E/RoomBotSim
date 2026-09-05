"""Runtime persistence and bounded-memory helpers.

Episode files are append-only while an episode runs.  On close, older JSONL files are
compressed into ``episodes/archive``; archives are never deleted by this module.
"""
import gzip
import json
import os
import shutil
import tempfile


def env_int(name, default, minimum=1):
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default

EVENT_LIMIT = env_int("RUNTIME_EVENT_LIMIT", 400)
EVENT_LOG_LIMIT = env_int("RUNTIME_EVENT_LOG_LIMIT", 120)
EPISODE_RETAIN = env_int("EPISODE_RETAIN", 200, 0)


def atomic_json_dump(path, value):
    """Write JSON atomically so a restart never consumes a partial metadata file."""
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try: os.unlink(tmp)
        except OSError: pass
        raise


def _closed_episode(path):
    """True only when the final JSONL record is a valid episode footer.

    A byte search for ``type: footer`` can rotate a live/corrupt log whose step
    payload merely contains that text.  Episode.close writes the footer last, so
    requiring a parseable final record is the durable closed-file contract.
    """
    last = None
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last = json.loads(line)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(last, dict) and last.get("type") == "footer"


def rotate_episodes(job_dir, retain=EPISODE_RETAIN):
    """Compress closed older episodes, retaining newest ``retain`` as JSONL.

    The current/open file (no final valid footer) is never rotated.  ``os.replace``
    makes final archive publication atomic; source data is removed only after the
    archive was completely written. Existing archives are retained indefinitely.
    """
    src = os.path.join(job_dir, "episodes")
    if not os.path.isdir(src):
        return []
    closed = []
    for name in os.listdir(src):
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(src, name)
        try:
            if not _closed_episode(path):
                continue
            closed.append((os.path.getmtime(path), path))
        except OSError:
            continue
    closed.sort(reverse=True)
    moved = []
    archive = os.path.join(src, "archive")
    for _, path in closed[retain:]:
        os.makedirs(archive, exist_ok=True)
        dst = os.path.join(archive, os.path.basename(path) + ".gz")
        if os.path.exists(dst):
            continue
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=archive)
        try:
            with os.fdopen(fd, "wb") as raw:
                with gzip.GzipFile(fileobj=raw, mode="wb") as out, open(path, "rb") as inp:
                    shutil.copyfileobj(inp, out)
                raw.flush(); os.fsync(raw.fileno())
            os.replace(tmp, dst)
            os.unlink(path)
            moved.append(dst)
        except Exception:
            try: os.unlink(tmp)
            except OSError: pass
            raise
    return moved


def load_jobs(root="jobs"):
    """Return valid persisted jobs without starting simulation threads."""
    jobs = {}
    if not os.path.isdir(root):
        return jobs
    for job_id in sorted(os.listdir(root)):
        d = os.path.join(root, job_id)
        if not os.path.isdir(d):
            continue
        try:
            with open(os.path.join(d, "world.json")) as f: world = json.load(f)
            with open(os.path.join(d, "people.json")) as f: people = json.load(f)
            if not isinstance(world, dict) or not isinstance(people, dict) or not isinstance(people.get("people"), list):
                raise ValueError("invalid world or people schema")
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        photos = [p[:-9] for p in ("A_1024.jpg", "B_1024.jpg", "C_1024.jpg", "D_1024.jpg")
                  if os.path.exists(os.path.join(d, p))]
        jobs[job_id] = dict(dir=d, photos=photos or ["A"],
                            status=dict(stage="ready", qwen="persisted", atlas="persisted",
                                        message="reloaded persisted job"),
                            world=world, people=people, runtime=None)
    return jobs
