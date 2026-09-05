#!/usr/bin/env python3
"""Run persisted WALK-IN episodes without uvicorn.

Example: python tools/batch_run.py --job test --episodes 10 --task coffee_to_person
"""
import argparse
import json
import os
import sys
import math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
os.environ.setdefault("WORLD_SOURCE", "mock")

import server
import tools.make_test_world as make_test_world


def step(rt, seconds):
    """Advance physics at its normal 100 Hz without a wall-clock server thread."""
    for n in range(int(seconds / .01)):
        if n % 5 == 0:
            rt.control_tick()
        rt.sim.step()
        rt.t += .01
        rt.handle_contacts()
        if rt.ep is None:
            return True
    return rt.ep is None


def main():
    p = argparse.ArgumentParser(description="Headless persistent WALK-IN episode runner")
    p.add_argument("--job", default="test")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--task", default="coffee_to_person", choices=("coffee_to_person", "bring_object", "go_to"))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-seconds", type=float, default=280)
    p.add_argument("--jobs-root", default="jobs", help="persisted jobs root (default: jobs)")
    args = p.parse_args()
    if args.episodes < 1:
        p.error("--episodes must be positive")
    if args.max_seconds <= 0:
        p.error("--max-seconds must be positive")
    jobs_root = os.path.abspath(args.jobs_root)
    loaded = server.load_jobs(jobs_root)
    if args.job not in loaded:
        if args.job != "test":
            p.error("job is not persisted (needs <jobs-root>/<id>/world.json and people.json)")
        make_test_world.build(os.path.join(jobs_root, "test"))
        loaded = server.load_jobs(jobs_root)
    job = loaded[args.job]
    rt = server.Runtime(args.job, job["world"], job["people"], job["dir"])
    rt.brain_enabled = False
    results = []
    for i in range(args.episodes):
        eid = rt.start_episode({"task_type": args.task, "policy": {"kind": "scripted"},
                                "seed": args.seed + i, "humans": "scripted",
                                "max_seconds": args.max_seconds})
        if not eid:
            raise RuntimeError("no robot available")
        finished = step(rt, math.ceil(args.max_seconds + 2))
        # A deadline is still a completed, recorded episode; never leave an open
        # JSONL behind merely because a synchronous batch budget was exhausted.
        if not finished and rt.ep is not None:
            rt.end_episode("timeout")
            finished = True
        recorded = next((e for e in __import__("episodes").list_episodes(job["dir"])
                         if e["episode_id"] == eid), None)
        results.append({"episode_id": eid, "finished": bool(finished),
                        "success": recorded.get("success") if recorded else None,
                        "tags": recorded.get("tags") if recorded else ["record_missing"]})
    report = {"job": args.job, "episodes": results}
    print(json.dumps(report, indent=2))
    # A missing persisted footer is an operational failure even if physics stopped.
    if any("record_missing" in r["tags"] for r in results):
        print("batch failed: one or more episode records were not written", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
