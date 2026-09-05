#!/usr/bin/env python3
"""Run persisted WALK-IN episodes without uvicorn.

Example: python tools/batch_run.py --job test --episodes 10 --task coffee_to_person
"""
import argparse
import json
import os
import sys

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
    args = p.parse_args()
    if args.episodes < 1:
        p.error("--episodes must be positive")
    if args.job not in server.JOBS:
        if args.job != "test":
            p.error("job is not persisted (needs jobs/<id>/world.json and people.json)")
        make_test_world.build("jobs/test")
        server.JOBS.update(server.load_jobs("jobs"))
    job = server.JOBS[args.job]
    rt = server.Runtime(args.job, job["world"], job["people"], job["dir"])
    rt.brain_enabled = False
    results = []
    for i in range(args.episodes):
        eid = rt.start_episode({"task_type": args.task, "policy": {"kind": "scripted"},
                                "seed": args.seed + i, "humans": "scripted",
                                "max_seconds": args.max_seconds})
        if not eid:
            raise RuntimeError("no robot available")
        finished = step(rt, args.max_seconds + 2)
        results.append({"episode_id": eid, "finished": finished})
    print(json.dumps({"job": args.job, "episodes": results}, indent=2))


if __name__ == "__main__":
    main()
