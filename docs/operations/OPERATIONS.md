# RoomBotSim runtime operations

## Restart and recovery
At import/server boot, valid `jobs/<id>/world.json` plus `people.json` pairs are loaded into `server.JOBS`. They are ready for `/status` and episode listing; simulation threads are **not** auto-started. Start an already-confirmed persisted room with `POST /runtime/start/{job_id}` before using its WebSocket or episode APIs. Invalid/partial job directories are skipped, not modified. World and people updates use atomic replacement, so an interrupted write retains the previous valid version.

## Resource limits
`RUNTIME_EVENT_LIMIT` (default `400`) bounds WebSocket-visible runtime events. `RUNTIME_EVENT_LOG_LIMIT` (default `120`) bounds prompt/event history. `BRAIN_PERIOD_S` defaults to `5` seconds. The planner shares the stamped people grid among robots replanning in the same simulation tick; robot-specific collision stamps remain separate.

Episode JSONL is append-only while live. Once closed, `EPISODE_RETAIN` (default `200`, `0` allowed) newest closed files remain in `episodes/`; older files are atomically gzip-compressed to `episodes/archive/`. Archives are retained indefinitely and are included by `/episodes`; no episode data or job directory is deleted by this policy.

## Headless batches
Run without uvicorn from `roombotsim/`:

```sh
python tools/batch_run.py --job test --episodes 10 --task coffee_to_person --seed 7
# use an isolated persisted root in CI:
python tools/batch_run.py --jobs-root /tmp/roombotsim-jobs --job test --episodes 1 --task go_to
```

The runner reloads an existing persisted job (or creates only the built-in `test` fixture), advances normal 100 Hz physics synchronously, writes a closed episode JSONL, prints a JSON report, and exits. It never starts an HTTP server. Invalid arguments and a missing episode record exit non-zero.

## `/api/act` cancellation
An accepted `POST /api/act/{job}` is durable: disconnecting the HTTP client **does not cancel** its robot skill or episode step. This avoids an abandoned request leaving a robot half through a manipulation. Clients should reconnect with `GET /api/observe/{job}` and use `POST /episode/stop/{job}` for explicit safe abort; a timed-out API response likewise does not imply cancellation.
