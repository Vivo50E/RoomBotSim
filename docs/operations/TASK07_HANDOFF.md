# TASK-07 handoff

## Console/API contract

The static console uses the existing API unchanged: `POST /upload`, `GET /status/{job}`,
`POST /confirm/{job}`, `GET /test`, `WS /ws/{job}`, episode start/stop/list endpoints,
and `POST /sft/{job}`. WebSocket state supplies `world`, people, robots, objects, and
`episode.last_action`/`last_result`. Human controls send `teleop` and `skill` messages;
the runtime writes its existing episode JSONL/SFT schema.

## QR handoff

`qr.png` is a valid, decoded QR for `http://10.104.4.240:8000/`, the LAN address detected
on this machine. Run `python tools/generate_qr.py http://LAN.IP:PORT/` after a network or
port change. `QR_README.md` documents the LAN server requirement and generator.

## Verification run

From `roombotsim/`, with the server running on `127.0.0.1:8000`:

```sh
python3 -m py_compile tests/verify_ui.py tools/generate_qr.py
python3 tests/verify_ui.py --url http://127.0.0.1:8000 --exercise-human
python3 tools/generate_qr.py http://192.168.50.42:8000/  # decoded with cv2, then restored qr.png
```

Results on this machine: Python and JS-module syntax passed; OpenCV decoded the checked-in
QR as `http://10.104.4.240:8000/`; a generated test QR decoded to its exact supplied LAN URL;
7x9, 20 m, and thin-corridor fit invariants passed; and Playwright passed desktop first-paint
canvas sizing, live demo/websocket, the useful phone uploader/QR handoff, and a Human run with
no duplicate visible navigation, no page errors, and a local JSONL sequence of exactly one
teleop-backed `navigate_to` immediately before `pick`. The verifier now skips—not
claims to run—QR decoding, JS syntax, or browser checks when their optional runtime dependency
is unavailable (the project venv lacks OpenCV and Python Playwright).

## Completion status

TASK-07 static acceptance pass is complete. The only completion-pass code repair was stable,
key-order-independent episode-step de-duplication in `index.html`; no API or backend file changed.

## Boundary note

No backend change was made. One backend-only polish item observed during the live run:
`Runtime.start_episode()` does not reset `ep_t_step` before a Human teleop segment, so the
recorded `duration_s` for that consolidated navigation can include time before the episode.
The samples, ordering, JSONL schema, and SFT export are correct; resetting that timer in
`roombotsim/server.py` would make the duration precise, but is outside the static-only boundary.
