# TASK-08 handoff

## Changed paths
- `runtime_ops.py`: rotation now requires a parseable final JSONL footer, rather than matching footer text anywhere in a file.
- `server.py`: boot-reloaded rooms can be started with `POST /runtime/start/{job_id}`; human episode teleop timing resets at episode start.
- `tools/batch_run.py`: isolated `--jobs-root`, input validation, closed-record verification, JSON result fields, and clean exit status.
- `tests/run_all.py`: disposable fixtures; reload-to-runtime, strict rotation/SFT preservation, planner cache isolation, human timer, durable API docs, and exact 30 s/29 s six-person/three-robot state-stream-client regression checks.
- `OPERATIONS.md`: documents runtime restart and batch behavior.

`episodes.py` and `tools/make_sft.py` already contained the partial worker's archive-aware listing/export changes; the suite now proves them through rotation.

## Verification
- `python -m py_compile runtime_ops.py episodes.py server.py tools/batch_run.py tests/run_all.py`
- `python tests/run_all.py` — **22 passed, 0 failed**.
- `python static/verify_task07.py` — static console regression verifier passed; no static files were edited by this task.
- isolated batch smoke: `python tools/batch_run.py --jobs-root <tmp> --job test --episodes 1 --task go_to --max-seconds 8` exited 0 with a closed timeout record; invalid `--max-seconds 0` exited 2.

## Measured performance
The enforced full-suite case ran **30.10 s wall** and advanced **30.11 s simulated** with exactly **6 people**, **3 active robots**, and a 20 Hz state-stream/WebSocket-sender-equivalent JSON consumer (threshold: 30 s wall, >=29 s sim).

## Limitation
The state-stream client is in-process (it calls the same `state_since` copy and JSON serialization used by the WebSocket sender), rather than a network WebSocket, to avoid an unavailable/incompatible FastAPI test client dependency. The server WebSocket payload contract is unchanged.
