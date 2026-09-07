# Task 07 — Finish the console and the human demonstration mode

**Owner:** one agent. **Depends on:** nothing.

## What exists

`static/index.html` is the whole operator console in one file: upload and status polling, a confirm stage
where people and landmarks are dragged into place on the top-down map, the live map, a three.js view with
Spot and humanoid rigs built from primitives, the event feed, and the episode panel with policy selection,
step log, result banner, replay grouping and the training-set button.

Its JavaScript parses clean and the server serves it, but **it has never been opened in a browser** — the
Chrome extension disconnected before the visual pass. That is the first job.

## Do this

1. **Open it and fix what is broken.** Load the demo room, drop Spot, click the map to send it somewhere,
   run a scripted episode. Watch the console. Expect small things: canvas sizing on first paint, the
   confirm-stage drag hit test, the step log deduplication key.
2. **Check the map at a real room's proportions.** The demo room is 7 m × 9 m. A long thin corridor or a
   20 m room will expose the `fit()` scaling.
3. **Human demo mode.** Pick "Human demo" as the policy, then drive with W/S and A/D over the map and use
   the skill buttons. The runtime records the teleop segment as one `navigate_to` step with its `[t, v, ω]`
   samples, then the skill itself, so a demonstration produces the same JSONL schema as a policy episode
   and goes straight into SFT. Verify that round trip: drive, pick, place, then export and read the file.
4. **The phone path.** Under 700 px the page shows only the top bar and the uploader. Someone has to
   actually stand in a room, shoot four photos on a phone, upload, and walk to the laptop. Generate
   `static/qr.png` pointing at the laptop's LAN address so this takes one scan.

   Two things to save you time. This laptop's address as of 5 September is **http://10.104.4.240:8000**, and a
   server is already running there, so the browser pass can start immediately without booting one.

   And a boundary note: `requirements.txt` lives in `roombotsim/`, which the other worker owns, so you cannot
   add a QR dependency to it. Do one of these instead. Best: have the page render the QR itself in
   JavaScript from `location.origin` and drop the PNG entirely, which also survives the laptop changing
   networks. Otherwise vendor a small pure-Python encoder under `static/` with a one-line generator
   script, and say in a comment that the address is environment-specific.
5. **Accessibility floor.** Focus outlines are deliberately left visible and `prefers-reduced-motion`
   disables the one entrance animation. Keep both. Check colour contrast on `--ink-dim` over `--panel`.

## Defects found by review, 5 September

These were found reading the code, not running it. Fix them as part of the task.

1. **On a phone, every error message is invisible.** The mobile rule sets `#app { display: none }`, and
   `<header>` lives inside `#app`, so `#statusline` is hidden. `uploadPhotos()` reports every failure
   through `setStatus()` — wrong file count, upload failure, pipeline error. On a phone all of those are
   silent: the person taps upload with one photo and nothing appears to happen. Route status to `#mobmsg`
   as well as `#statusline`, or give `setStatus` a mobile-aware target.

2. **`tools/generate_qr.py` does not run.** Its docstring says "without adding a Python dependency" and
   then line 6 is `import cv2`. OpenCV is not installed in `.venv` and is not in `requirements.txt`, so
   the script raises `ModuleNotFoundError` before it reaches any of its fallback logic. `qr.png` itself is
   fine — a valid 296x296 PNG, evidently produced some other way — but nobody can regenerate it, which was
   the whole point of shipping a generator.

   Fix by removing the hard import. Either encode the QR in pure Python inside the script (a QR encoder is
   ~150 lines and has no dependencies), or shell out to `qrencode` and fail with a clear message when it is
   missing, or drop the file and draw the code client-side from `location.origin`, which is still the best
   option because it survives the laptop changing networks.

## Done when

A person who has not seen the project can upload four photos from a phone, confirm the room on the laptop,
run an episode, and download the training set, without being told what to click.
