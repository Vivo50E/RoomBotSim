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
5. **Accessibility floor.** Focus outlines are deliberately left visible and `prefers-reduced-motion`
   disables the one entrance animation. Keep both. Check colour contrast on `--ink-dim` over `--panel`.

## Done when

A person who has not seen the project can upload four photos from a phone, confirm the room on the laptop,
run an episode, and download the training set, without being told what to click.
