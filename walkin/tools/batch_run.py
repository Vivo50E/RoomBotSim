#!/usr/bin/env python3
"""Run persisted WALK-IN episodes without uvicorn.

Example: python tools/batch_run.py --job test --episodes 10 --task coffee_to_person
"""
import argparse
import json
import os
import sys
import time
import math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
os.environ.setdefault("WORLD_SOURCE", "mock")

import server
import episodes as epmod
import tools.make_test_world as make_test_world


def step(rt, seconds, policy_wait_s=120.0):
    """Advance physics at its normal 100 Hz without a wall-clock server thread.

    A remote policy answers on wall-clock time while this loop runs roughly 30x faster than real
    time, so advancing the simulation during that call spends the episode's whole budget before the
    first action ever arrives. Measured: a two-episode qwen batch recorded 0 and 1 steps and both
    ended `timeout`. Hold simulated time still while a policy call is in flight.
    """
    for n in range(int(seconds / .01)):
        if rt.policy_inflight:
            waited = 0.0
            while rt.policy_inflight and waited < policy_wait_s:
                time.sleep(0.02); waited += 0.02
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
    p.add_argument("--policy", default="scripted", choices=("scripted", "chat", "raw", "vlm"),
                   help="scripted runs the built-in oracle; chat and raw call your model; vlm sends the robot camera frame too")
    p.add_argument("--policy-url", default=None, help="OpenAI-compatible /chat/completions, or your raw endpoint")
    p.add_argument("--policy-model", default=None)
    p.add_argument("--requester", type=int, default=None, help="person id who asked; defaults to the first")
    p.add_argument("--robot-type", default="spot", choices=("spot", "humanoid"))
    p.add_argument("--max-seconds", type=float, default=280)
    p.add_argument("--jobs-root", default="jobs", help="persisted jobs root (default: jobs)")
    args = p.parse_args()
    if args.episodes < 1:
        p.error("--episodes must be positive")
    if args.max_seconds <= 0:
        p.error("--max-seconds must be positive")
    # The key comes from the environment, never the command line, so it stays out of shell history.
    policy_key = os.environ.get("POLICY_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
    policy = {"kind": args.policy}
    if args.policy != "scripted":
        policy.update(url=args.policy_url or os.environ.get("POLICY_URL") or "https://openrouter.ai/api/v1/chat/completions",
                      model=args.policy_model or os.environ.get("POLICY_MODEL"), key=policy_key)
        if not policy["url"] or not policy["key"]:
            p.error("--policy %s needs --policy-url (or POLICY_URL) and POLICY_KEY in the environment"
                    % args.policy)
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
        eid = rt.start_episode({"task_type": args.task, "policy": policy,
                                "seed": args.seed + i, "humans": "scripted",
                                "requester": args.requester, "robot_type": args.robot_type,
                                "max_seconds": args.max_seconds})
        if not eid:
            raise RuntimeError("no robot available")
        finished = step(rt, math.ceil(args.max_seconds + 2))
        # A deadline is still a completed, recorded episode; never leave an open
        # JSONL behind merely because a synchronous batch budget was exhausted.
        if not finished and rt.ep is not None:
            rt.end_episode("timeout")
            finished = True
        recorded = next((e for e in epmod.list_episodes(job["dir"])
                         if e["episode_id"] == eid), None)
        results.append({"episode_id": eid, "finished": bool(finished),
                        "success": recorded.get("success") if recorded else None,
                        "tags": recorded.get("tags") if recorded else ["record_missing"]})
    ok = sum(1 for r in results if r["success"])
    tags = {}
    for r in results:
        for t in (r["tags"] or []):
            tags[t] = tags.get(t, 0) + 1
    report = {"job": args.job, "policy": args.policy, "task": args.task,
              "success_rate": round(ok / len(results), 3) if results else None,
              "succeeded": ok, "of": len(results),
              "tags": dict(sorted(tags.items(), key=lambda kv: -kv[1])),
              "episodes": results}
    print(json.dumps(report, indent=2))
    # A missing persisted footer is an operational failure even if physics stopped.
    if any("record_missing" in r["tags"] for r in results):
        print("batch failed: one or more episode records were not written", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
