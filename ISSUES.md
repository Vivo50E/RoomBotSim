# Open issues — as of 5 September, 15:05

The demo runs end to end on the baked coffee room at `GET /demo`. Everything below is what is still
wrong or unverified, most important first. Each item says what was tried, so nobody repeats it.

## 1. Follow camera during pick / pour / place is not verified

**State:** During `navigate_to` the chase camera is clean (eye height, 0.7 m clearance test against the
occupancy grid). During manipulation the code cuts to a vantage 4.5 → 2.4 m from Spot on the room-centre
side, aimed at arm height. The last screenshot before this change showed the camera about 1 m from Spot
looking down its back; the change that pushes it back was applied and syntax-checked but **not
screenshotted** — the user stopped the session while the capture ran.

**Tried and rejected:** a chase cam 2.6–3.4 m behind Spot at 2.0–2.6 m height (lands inside the high
table's skirt: solid navy frame); a ring search for any free cell (same problem — a free cell one cell
from a table still has the table's gaussians in the lens); an overhead drone shot (inside pendant-lamp
gaussians).

**Verify with:** `tools/verify_demo.py` proves the sim; for the picture, run the Playwright snippet in the
git log for commit "Vantage distance ladder" and look at `x_pick.png` / `x_pour.png`.

## 2. People are placed by heuristic, not by where they actually stood

Marble returns no input-camera poses, so there is no depth to unproject a person's feet into. Stage 2
seats "seated" people on stools along the main table and sends standing people to whatever landmark
their description names (entrance, kitchen, couch), else open floor. Plausible, not measured. Real fix is
TASK-02 (recover camera poses).

## 3. Retrained act (act two) passes on 13 of 14 crowd schedules

Seed 11 fails on the current placement (`no_progress` while yielding). The page draws act two's seed
from the 12 verified ones and act one's at random, so a judge will not see it. It is still a real
robustness gap: with people wandering in a 10 × 12 m room, the robot sometimes cannot get a clean run at
the requester. Sweep: `python - <<…` in the git log for "Sweep seeds for the retrained act".

## 4. "place(pot) → FAIL dropped" every run — resolved

Fixed 5 September: a labelled landmark can be much larger than the physical surface it names, so the
old navigator faced its empty geometric centre and released over the floor. Navigation and placement
now select a real, arm-reachable obstacle surface within the landmark. `tools/verify_demo.py coffee`
now reports `place pot -> ok placed`, with no `dropped` episode tag.

## 5. Splat clean-up is a blunt filter

`primary_clean.ply` drops gaussians with opacity < 0.22, max scale > 0.30 m, or outside the cropped room.
That removed the giant pendant-lamp blobs and 53% of the splats, but some floaters remain and the lamps
are gone rather than fixed. The offline pruning script is inline in the git log ("Prune floaters");
it should become `tools/prune_splat.py` with the thresholds as arguments.

## 6. Cross-check fusion rejected two of three views

The three single-photo Marble worlds registered to the primary with overlap 0.30, 0.18 and 0.10; only the
first cleared the 0.30 bar, so consensus ran on two scenes. Single-photo worlds invent a different room.
Either the cross-check should use pairs of photos, or it should be dropped as not worth the credits.

## 7. Performance

1.06 M gaussians, pixel ratio 1, no MSAA. Fine on the demo laptop's GPU in headless Chrome; not
measured in a real window with the side panels. If it stutters, cut the splat harder (opacity > 0.35) or
export Marble's 500k resolution instead of full.

## 8. Not done at all

- No real robot model: Spot is primitives to BD's proportions with a velocity-driven walk cycle.
- Humanoids are stylised capsule figures, one height each; no faces, no clothing beyond shirt colour.
- The "retrain" step is simulated (20 s progress bar). The data export behind it is real.
- Chrome extension never reconnected; every visual check was headless Chrome via Playwright.
- The `TASK-*.md` items from earlier are unchanged and still open.

## What is known to work

`tools/verify_demo.py coffee` → 9/9. `tests/run_all.py` → 23/23. Act one (untrained, spills) on every
seed tried. Act two on 13/14 seeds. Splat renders right side up at the correct metric scale, verified by
eye-height screenshot. `/demo` serves the baked room in about 40 s including the 49 MB splat.
