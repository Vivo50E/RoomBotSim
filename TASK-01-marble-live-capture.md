# Task 01 — Get four real photos through Marble end to end

**Owner:** one agent. **Depends on:** nothing. **Blocks:** 02, 03.

Everything downstream is built and tested against the synthetic room. What has not been exercised is a
real Marble round trip. The client is written and the credit endpoint answers, but no room has actually
been generated from photos yet.

## What exists

`walkin/marble.py` implements the whole World API flow against `https://api.worldlabs.ai`:
`media-assets:prepare_upload` → PUT bytes → `worlds:generate` with a `multi-image` prompt carrying one
azimuth per photo and `reconstruct_images: true` → poll `operations/{id}` → `worlds/{id}:export` for PLY
splats and a GLB collider mesh. `reconstruct_room()` also fires one single-image world per photo when
`MARBLE_CROSS_CHECK=1`, so `fusion.py` has independent reconstructions to compare.

## Do this

1. Take four photos of one room at chest height, one facing each wall, with overlap at the corners. Save
   them somewhere you can re-run from; you will use this same set for weeks.
2. `WORLD_SOURCE=marble` and upload them through the UI, or call `recon.reconstruct(paths, "jobs/x")`
   directly from a script so you can iterate without the browser.
3. Check the numbers that the code currently trusts on faith:
   - **Azimuth.** The code assigns `[0, 90, 180, 270]` by list order. Confirm that is what Marble expects
     for a camera rotating in place, and whether azimuth is clockwise or counter-clockwise.
   - **Axes.** `fusion.MARBLE_TO_ZUP` maps `(x, y, z) → (x, z, −y)` on the assumption that Marble exports
     OpenCV axes with `+y` down. Load the PLY, plot it, and confirm the floor comes out horizontal before
     `floor_align` runs. If it does not, fix that one matrix, not anything downstream.
   - **Scale.** `assets.splats.semantics_metadata.metric_scale_factor` and `ground_plane_offset` are read
     and applied in `fusion.normalize`. Measure something you know the size of (a door is 2.03 m) and
     confirm the grid comes out metric. If the factor is already applied in the export, remove it.
4. Record what a run costs in credits and how long it takes, in this file, so the next person can plan.

## Done when

`jobs/<id>/world.json` from four real photos gives a grid whose room dimensions match a tape measure to
within 10%, and `plan_xy` finds a path from the door landmark to the table landmark.
