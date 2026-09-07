> **Status: superseded, 5 September.** Two corrections to this audit.
>
> 1. Its headline FAIL was correct and useful: the performance case really did bypass the WebSocket
>    route. That has since been fixed — the suite now reserves a loopback port, serves the real app from
>    a uvicorn thread, and connects with the `websockets` client. Measured 589 state frames over 30.1 s,
>    every one carrying six people and three robots, no client errors. The suite reports 23 passed.
> 2. Its evidence section claims "NumPy divide/overflow/invalid-value warnings from `recon.py:179` and
>    `geometry.py:28,42,43,52`". Those do not reproduce. Running the whole suite under
>    `-W always::RuntimeWarning` emits no NumPy RuntimeWarning at all, and none of the cited lines
>    performs a division. The warnings that *were* present were `ResourceWarning` for unclosed file
>    handles, which is a different thing and has now been fixed at the source.
>
> Treat the verdict as historical.

# TASK-07 / TASK-08 independent acceptance audit

Audit date: 2026-09-05. Product source was not edited by this audit.

## Verdict

**FAIL — one TASK-08 acceptance requirement is not met by the claimed performance regression.** The test does assert the required >=29 s simulated-time threshold and did pass, but it does **not** attach a real WebSocket client. It calls `Runtime.state_since()` and `json.dumps()` in-process, bypassing the server WebSocket route, async send, socket transport, and receiver/task behaviour. The task explicitly requires a WebSocket client and the request expressly disallows a watered-down proxy.

All other inspected TASK-07/TASK-08 requirements passed or are evidenced below. Physical phone/LAN operation was not performed; that portion is static/browser-verified only.

## Commands run and outputs

```sh
cd roombotsim && python tests/run_all.py
```

Output summary: **22 passed, 0 failed**. The final performance line was:

```
PASS performance: 30 s wall advances >=29 s (6 people, 3 robots, state-stream client)  sim=30.10s wall=30.10s
```

The run emitted pre-existing NumPy divide/overflow/invalid-value warnings from `recon.py:179` and `geometry.py:28,42,43,52`; no assertion failed.

```sh
cd roombotsim && python -m uvicorn server:app --host 127.0.0.1 --port 8765
cd roombotsim && python tests/verify_ui.py --url http://127.0.0.1:8765 --exercise-human
```

The app was locally served and the verifier output:

```
JS syntax: passed
QR: http://10.104.4.240:8000/
map fit: 7×9, 20 m room, and corridor extents passed
browser: desktop canvas, live demo, and phone uploader passed
TASK-07 static checks passed
```

The optional Playwright Human branch was enabled. Its assertions confirmed the local JSONL has exactly one teleop-backed `navigate_to` immediately before `pick` when the record exists (it did for this local server), and no duplicate visible navigation row.

```sh
cd roombotsim && curl -sS http://127.0.0.1:8765/status/dbg2
curl -sS -X POST http://127.0.0.1:8765/runtime/start/dbg2
curl -sS http://127.0.0.1:8765/status/dbg2
```

Output summary: persisted `dbg2` was `ready`, had a world and four people; start returned `{"ok":true,"job_id":"dbg2"}`; status became `running`.

```sh
cd roombotsim && python -m py_compile runtime_ops.py episodes.py server.py tools/batch_run.py tools/generate_qr.py tests/verify_ui.py
cd roombotsim && python tools/batch_run.py --jobs-root /tmp/audit-roombotsim-batch --job test --episodes 1 --task go_to --max-seconds 8
```

`py_compile: PASS`. The isolated batch completed with a closed timeout record, exited 0, and printed `"finished": true` (the short 8 s budget predictably reported `success_rate: 0.0`, tag `timeout`).

```sh
cd roombotsim && python tools/generate_qr.py http://192.168.50.42:8000/
# Pillow inspection and OpenCV decode
cd roombotsim && python tools/generate_qr.py http://127.0.0.1:8000/
```

Generated QR output was PNG `(264, 264)` and decoded to the supplied LAN URL. The restored checked-in QR decoded to `http://10.104.4.240:8000/`. Loopback regeneration correctly exited 1 with `Refusing a loopback QR`.

## Requirement evidence

| Requirement | Status | Evidence |
|---|---|---|
| 30-second 6-person/3-active-robot performance, >=29 s sim | **FAIL** | `tests/run_all.py:222-232` genuinely builds six people, disables the brain, and drops three robots; lines 243-248 sleep 30.1 s and assert `elapsed_sim >= 29.0`. Actual run: 30.10 sim seconds. But client requirement fails as detailed below. |
| Actual WebSocket client on performance test | **FAIL** | `tests/run_all.py:233-241` explicitly labels its thread “Equivalent to a websocket sender” and calls `perf.state_since()` directly. It never connects `/ws/{job_id}` (`server.py:994-1035`). This is an in-process proxy, not a WebSocket client. |
| Boot reload and persisted runtime start | PASS | `server.py:39` loads persisted jobs at import; `runtime_ops.py:91-113` validates and loads world/people; `server.py:286-324` starts a loaded job. Full suite and real `dbg2` HTTP start both passed. |
| Bounded runtime queues | PASS | `server.py:424-433` limits visible events using `EVENT_LIMIT` (400 default) and prompt history using `EVENT_LOG_LIMIT` (120 default); defaults are `runtime_ops.py:20-22`. |
| Episode rotation/archive visibility/SFT preservation | PASS | `runtime_ops.py:49-89` only rotates closed JSONL with a parseable final footer and writes gzip atomically. `episodes.py:129-158` lists normal plus archived records; `tools/make_sft.py:17-28` exports both. Full suite created 3 records with retain=1 and confirmed 4 listed including live, 2 archived, and 3 SFT examples. |
| Headless batch and policy options | PASS | `tools/batch_run.py:31-104` is synchronous/no uvicorn, closes deadline records, returns nonzero for missing record, accepts scripted/chat/raw plus URL/model, environment key, requester and robot type. Isolated real command exited 0. |
| `/api/act` durable-disconnect semantics documented | PASS | `server.py:1119-1153` documents/block-waits behaviour and `server.py:1156-1169` exposes cancellation text. `OPERATIONS.md` also documents reconnect/explicit stop. `/api/schema` returned the same policy from the served app. |
| Map proportions, initial canvas paint, confirm drag | PASS | `static/index.html:410-430` DPR-sizes first paint and fits room extent; `:492-523` uses screen-size-aware hit tests, pointer capture/cancel, and clamped confirm dragging. Static regression covers 7x9, 20x20, and corridor extents; Playwright confirmed desktop bitmap dimensions after first paint. |
| Human teleop recording/export | PASS | `server.py:805-809` resets teleop time at episode start; `server.py:920-955` records one `navigate_to` with samples before the requested skill. Browser Human test drove W then pick and checked the local JSONL sequence; UI `/sft` request includes `include_human:true` at `index.html:825-832`. |
| Phone/mobile errors and QR LAN handoff | PASS (physical phone not exercised) | `index.html:99-105,234-243,294-303` exposes mobile uploader, QR handoff, and mirrors errors to `#mobmsg`; mobile Playwright pass confirms uploader/QR visibility. Generator has no OpenCV import (`tools/generate_qr.py`) and QR decoding/regeneration passed. Actual phone upload/on-LAN scan remains untested. |
| Focus/reduced motion/contrast | PASS | Visible cyan `:focus-visible` outline is `index.html:26`; reduced-motion suppression is line 98. `--ink-dim #AEBBC9` over `--panel #161C25` is a visibly high-contrast pairing (approximately 8:1); no focus outline is suppressed. |
| Frontend/backend compatibility | PASS | Console endpoints/messages match server routes: `/upload`, `/status`, `/confirm`, `/test`, `/ws`, episode APIs and `/sft`; Human `teleop`/`skill` payloads are handled at `server.py:1027-1034`. Local desktop/WebSocket/Human smoke completed without page errors. |

## Required remediation

1. **Replace the performance proxy with a real network WebSocket client.**
   - **File/lines:** `roombotsim/tests/run_all.py:233-244`.
   - Start an actual ASGI server bound to a temporary/local port for the performance fixture, register the fixture runtime/job with that server, and connect a real client to `ws://127.0.0.1:<port>/ws/<job>` for the full 30.1-second measurement. Continuously receive state frames (20 Hz server cadence) and JSON-decode them. Assert connection, receipt of state frames, six people, three active robots, and `elapsed_sim >= 29.0`.
   - Do not replace this with `Runtime.state_since`, FastAPI direct-call testing, or an in-process serialization loop; those bypass `server.py:994-1035` and do not satisfy the requirement.

## Audit limitations

- The local browser smoke ran against `127.0.0.1`; no physical handset, four real photos, or reachable 10.104.4.240 LAN server was available to independently exercise.
- The test suite performance assertion is real-time (30.10 seconds) but the only client workload in that test is the inadequate proxy identified above.
