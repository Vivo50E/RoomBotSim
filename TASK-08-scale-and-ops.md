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

1. **Profile.** 6 people, 3 robots, a websocket client, 60 s. If `t` falls behind wall clock, the cheapest
   wins are in `Cast.tick` (the social-force loop is O(n²) over a handful of agents, fine) and in
   `drive_robot`'s replan, which rebuilds two stamped grids every 0.5 s per robot. Cache the stamped grid
   across robots in the same tick.
2. **Reload jobs from disk on boot.** Scan `jobs/*/world.json` into `JOBS` at startup so a restart does not
   lose a room someone spent five minutes photographing.
3. **Bound the queues.** `Runtime.events` is trimmed to 400; `event_log` is not. Episode JSONL files grow
   without limit. Add rotation.
4. **Headless batch mode.** A runner that starts a Runtime, runs N episodes, and exits without uvicorn.
   Task 05 needs this and should not have to invent it.
5. **Failure surface.** `POST /api/act` blocks up to 70 s waiting for a skill. If a caller disconnects the
   skill keeps running. Decide whether that is right, and either document it or add cancellation.

## Done when

`tests/run_all.py` includes a performance case that asserts 30 s of wall clock advances sim time by at
least 29 s with 6 people and 3 robots, and it passes on the demo laptop.
