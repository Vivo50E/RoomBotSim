# Final acceptance audit — TASK-07 / TASK-08

Audit date: 2026-09-05. This audit did not edit product source.

## Final verdict

| Task | Verdict | Basis |
|---|---|---|
| TASK-07 — UI and Human demos | **PASS** | Local served-browser verifier, including Human teleop-to-JSONL smoke, passed; source spot checks cover the stated regressions. Physical phone/LAN use was not independently performed. |
| TASK-08 — scale and ops | **PASS** | The suite passed its genuine TCP/WebSocket performance regression and the source/test satisfy reload, rotation, batch, and API requirements. |

No product remediation is required for the audited requirements.

## Commands actually run

```sh
cd roombotsim && python tests/run_all.py
```

Actual final output:

```text
PASS performance: real network WebSocket connects and receives 6-person/3-robot state frames  frames=589 matching=589 errors=[]
PASS performance: 30 s wall advances >=29 s (6 people, 3 robots, network WebSocket client)  sim=30.11s wall=30.11s

23 passed, 0 failed
```

> **Correction, added on review.** The sentence that stood here claimed the run emitted NumPy divide,
> overflow and invalid-value warnings at `recon.py:179` and `geometry.py:28,42,43,52`. It is not true,
> and it is repeated verbatim from the earlier audit under a heading that reads "Commands actually run".
>
> Checked three ways. Running this exact command under `-W always` produces zero lines matching
> RuntimeWarning, divide, overflow or invalid value. `recon.py:179` is `off = np.array([0.8, -0.4, 0.3],
> np.float32)`, a literal array constructor. The four cited `geometry.py` lines are matrix multiplies and
> a comparison. None of the five divides.
>
> The warnings genuinely present were `ResourceWarning` for unclosed file handles, since fixed. The rest
> of this audit, including the WebSocket recheck below, was verified independently and holds.

```sh
cd roombotsim
python -m uvicorn server:app --host 127.0.0.1 --port 8766
python tests/verify_ui.py --url http://127.0.0.1:8766 --exercise-human
```

Actual verifier output:

```text
JS syntax: passed
QR: http://10.104.4.240:8000/
map fit: 7×9, 20 m room, and corridor extents passed
browser: desktop canvas, live demo, and phone uploader passed
TASK-07 static checks passed
```

The audit-owned Uvicorn process on port 8766 was stopped after the verifier. A stale port-8765 Uvicorn process left by an earlier audit command was found and stopped; the pre-existing application server on port 8000 was not touched.

## Critical WebSocket recheck

**PASS.** `tests/run_all.py` does not use a route proxy, `TestClient`, direct route call, or `Runtime.state_since()` from the test client:

- it reserves a real loopback TCP listener and gives it to `uvicorn.Server(server.app, ...)`;
- it registers the fixture runtime in `server.JOBS` and starts a separate runtime thread;
- a separate client thread runs `websockets.connect("ws://127.0.0.1:<ephemeral-port>/ws/<job>")`;
- that client continuously `recv`s and `json.loads`s frames for the entire measurement;
- the test asserts connected state frames and matching `people == 6`, `robots == 3`, then asserts wall >=30 and simulation >=29 seconds.

The real route in `server.py:994-1039` accepts the socket, sends through `websocket.send_text`, and polls its runtime publisher at 50 ms. The observed 589 matching frames confirms socket transport and decode rather than an in-process serialization stand-in.

Cleanup is in `finally`: client stop/join, runtime stop/join, Uvicorn `should_exit`/join, `JOBS.pop`, and listener close. The two performance assertions execute **after** that cleanup, so startup/client errors retained in `client_errors` cannot become a pass. The suite process exited normally and did not leave its ephemeral Uvicorn listener/service process behind.

## Requirement spot checks

| Requirement | Verdict | Evidence |
|---|---|---|
| Six people, three active robots, real-time target | PASS | Test deep-copies the four-person demo cast, adds two distinct walking people, drops three robots, disables brain, and actual run reports `sim=30.11s`, `wall=30.11s`. |
| Boot reload / usable restart | PASS | `runtime_ops.load_jobs()` validates `jobs/*/world.json` plus `people.json`; `server.py` loads this at import and `/runtime/start/{job}` builds the runtime. Suite restart check passed. |
| Bounded queues and episode rotation | PASS | Runtime limits are 400 visible events and 120 event-log entries. Closed older JSONL files rotate atomically to gzip; `episodes.list_episodes` and `tools/make_sft.build` both read archives. The suite's retain=1 test passed with 3 exported steps. |
| Headless batch and policy compatibility | PASS | `tools/batch_run.py` runs synchronous simulation without Uvicorn, closes timeouts, and exposes `scripted|chat|raw`, URL/model, environment key, requester, and robot type. The suite batch invocation passed and persisted a record. |
| `/api/act` disconnect decision / API compatibility | PASS | `server.py` explicitly documents durable accepted actions and exposes cancellation text in `/api/schema`; WebSocket `teleop` and `skill` messages are handled by the real route. |
| Initial canvas, map fit, drag, mobile errors, accessibility | PASS | Local Playwright verifier passed desktop bitmap sizing, 7×9 / 20 m / corridor fit, live demo, and mobile uploader. Source has visible `:focus-visible`, reduced-motion suppression, pointer cancellation, and `setStatus` mirroring to `#mobmsg`. |
| Human demonstration recording and export | PASS | Browser smoke drove W then Pick and checked local JSONL's one teleop-backed `navigate_to` immediately before `pick`. `server.py` resets the teleop clock at episode start and records samples; UI SFT request sets `include_human:true`. |
| QR regeneration / phone handoff | PASS (browser/static scope) | Verifier decoded the checked-in non-loopback LAN QR; source exposes the mobile upload/handoff UI. A real handset, four physical photos, and LAN scan were not independently exercised. |

## Audit limitations

No physical phone upload, room photography, or actual LAN QR scan was possible in this audit. Those are operational checks, not a failure of the verified browser/static behavior.
