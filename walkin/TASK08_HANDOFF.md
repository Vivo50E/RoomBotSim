# TASK-08 handoff

## Changed paths
- `runtime_ops.py`: rotation now requires a parseable final JSONL footer, rather than matching footer text anywhere in a file.
- `server.py`: boot-reloaded rooms can be started with `POST /runtime/start/{job_id}`; human episode teleop timing resets at episode start.
- `tools/batch_run.py`: isolated `--jobs-root`, input validation, closed-record verification, JSON result fields, and clean exit status.
- `tests/run_all.py`: disposable fixtures; reload-to-runtime, strict rotation/SFT preservation, planner cache isolation, human timer, durable API docs, and an exact 30 s/29 s six-person/three-robot regression through a real loopback uvicorn `/ws/{job}` connection.
- `WEBSOCKET_PERF_FIX.md`: exact commands, passing result, and network-client proof for the performance regression.
- `OPERATIONS.md`: documents runtime restart and batch behavior.

`episodes.py` and `tools/make_sft.py` already contained the partial worker's archive-aware listing/export changes; the suite now proves them through rotation.

## Verification
- `python -m py_compile runtime_ops.py episodes.py server.py tools/batch_run.py tests/run_all.py`
- `python tests/run_all.py` — **23 passed, 0 failed** (including real network WebSocket performance assertions).
- `python static/verify_task07.py` — static console regression verifier passed; no static files were edited by this task.
- isolated batch smoke: `python tools/batch_run.py --jobs-root <tmp> --job test --episodes 1 --task go_to --max-seconds 8` exited 0 with a closed timeout record; invalid `--max-seconds 0` exited 2.

## Measured performance
The enforced full-suite case ran **30.10 s wall** and advanced **30.11 s simulated** with exactly **6 people**, **3 active robots**, and a real `websockets.connect` client continuously decoding frames from the actual loopback uvicorn `/ws/{job}` route (threshold: 30 s wall, >=29 s sim). The test received 588 matching state frames. See `WEBSOCKET_PERF_FIX.md`.
