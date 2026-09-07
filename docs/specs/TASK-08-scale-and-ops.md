# Task 08 — Make it survive a room full of people using it

**Owner:** one agent. **Depends on:** nothing.

## Known limits, measured

- The runtime loop is 100 Hz physics, 20 Hz control and publish, and resyncs its wall clock if it falls
  more than 0.5 s behind. It has been run with 4 people and 1 robot. It has **not** been profiled with 6
  people and 3 active robots plus a websocket client, which is the spec's own performance target.
- `JOBS` is an in-process dict. Restart the server and every job is gone, though `world.json`,
  `people.json` and the episode JSONL files survive on disk. There is no route that reloads a job from
  disk except `/test`.
- One `Runtime` per job, each with its own thread. Nothing limits how many jobs run at once.
- A crowd that presses against a robot used to deadlock it: the yield rule stopped the robot whenever
  anyone came within 0.6 m and closing, forever. It now eases past at 0.22 m/s after 3 s held up, steering
  away from whoever is closest. That trade buys liveness at the cost of occasional contacts, and contacts
  are recorded as `bumped_person` tags rather than hidden.

## Do this

1. **Profile — but the answer is already in.** Measured on this machine, 5 September: 6 people, 3 active
   robots, publishing state every tick, no websocket client. It ran **20.0 s of sim in 0.67 s of wall
   clock, 30x real time**. The target is 1x. There is nothing to optimise here.

   So: write the perf case as a regression guard, not as an optimisation project. Do **not** cache the
   stamped planning grids across robots — that was my suggestion before measuring and it is premature.
   If your harness reports anything near 1x, suspect the harness: the most likely cause is calling
   `control_tick` every step instead of every fifth, or leaving the LLM brain enabled so every tick blocks
   on a network call. Set `brain_enabled = False` for the test.

   The one thing worth measuring that this run did not cover is a **real websocket client attached**, since
   `publish_state` serialises the whole world 20 times a second and `state_since` copies it per client.
2. **Reload jobs from disk on boot.** Scan `jobs/*/world.json` into `JOBS` at startup so a restart does not
   lose a room someone spent five minutes photographing.
3. **Bound the queues.** `Runtime.events` is trimmed to 400; `event_log` is not. Episode JSONL files grow
   without limit. Add rotation.
4. **Headless batch mode.** A runner that starts a Runtime, runs N episodes, and exits without uvicorn.
   Task 05 needs this and should not have to invent it.
5. **Failure surface.** `POST /api/act` blocks up to 70 s waiting for a skill. If a caller disconnects the
   skill keeps running. Decide whether that is right, and either document it or add cancellation.

## Defect found by review, 5 September

**Rotation silently removes episodes from the training data.** `runtime_ops.rotate_episodes` gzips closed
episodes into `episodes/archive/` and unlinks the original. `episodes.list_episodes` was correctly updated
to read both, so the UI still shows them — but `tools/make_sft.py` line 18 still globs only
`os.path.join(episodes_dir, "*.jsonl")`, so every archived episode disappears from the fine-tuning export
with no warning.

Reproduced: three recorded episodes, `retain=1`, exporter returned examples from one of them. At the
default `EPISODE_RETAIN=200` this bites exactly when Task 05 starts running episodes in volume, which is
the one moment the data matters most.

Fix `make_sft.build` to walk `archive/*.jsonl.gz` as well, opening with `gzip.open(path, "rt")`, the same
way `list_episodes` now does. Add a test that records more episodes than `EPISODE_RETAIN`, rotates, and
asserts the exporter still returns every step.

## Second defect, same review

**The performance test does not test what it claims.** The new case is labelled
`"performance: 30 s wall advances >=29 s (6 people, 3 robots)"` but it builds its Runtime from the demo
room, which has **four** people, not six. Six is the stated target, and the label asserts six, so the test
passes while leaving the requirement unmeasured.

Grow the cast to six before constructing the Runtime — copy two existing entries, give them fresh ids,
distinct `pos_xy`/`home_xy`, colours 5 and 6 from the palette, `posture: standing` and `activity: walking`
so they actually move. For reference, six people with three robots measured 30x real time on this machine,
so the assertion will still pass comfortably; the point is that it should be measuring the real thing.

Also note the case adds a hard `time.sleep(30.1)` to every run, taking the suite from 2 s to 32 s. That is
a fair price for a real-time guard, but consider gating it behind a flag so the fast feedback loop stays
fast.

## Third defect, same review

**The batch runner can only run the oracle.** `tools/batch_run.py` is otherwise exactly right — it loads a
persisted job, runs N episodes at 100 Hz with no uvicorn and no wall-clock thread, converts an exhausted
deadline into a properly closed `timeout` episode rather than leaving an open JSONL behind, and exits
non-zero when a record is missing. But the policy is hardcoded to `{"kind": "scripted"}`.

Task 05 exists to measure a model against itself before and after fine-tuning, and its first step is
"N seeds x the chat policy, headless". With the policy fixed to the built-in oracle, this runner cannot
produce that number at all — the oracle succeeds every time by construction.

Add `--policy scripted|chat|raw`, plus `--policy-url`, `--policy-model` and a key read from the
environment rather than the command line so it does not land in shell history. While you are there,
`--requester` and `--robot-type` are worth exposing for the same reason. Everything else about the file
can stay as it is.

## Done when

`tests/run_all.py` includes a performance case that asserts 30 s of wall clock advances sim time by at
least 29 s with 6 people and 3 robots, and it passes on the demo laptop.
