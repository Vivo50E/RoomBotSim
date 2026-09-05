# Task 04 — Make the cross-comparison sharper

**Owner:** one agent. **Depends on:** 01.

## What exists and what it is worth

`walkin/fusion.py` takes the multi-image world plus one single-image world per photo and repairs the
geometry:

- `normalize` puts every cloud in the same kind of frame: Marble axes to Z-up, metric scale and ground
  offset applied, RANSAC floor to `z = 0`, origin at the floor centroid.
- `register` aligns each secondary to the primary. Floor alignment has already fixed roll, pitch and z,
  so only yaw and xy remain. It proposes candidates by FFT cross-correlation of top-down density images
  and then verifies each with a 3D voxel overlap, which is what rejects the plausible-looking 90° errors
  that a top-down correlation alone accepts.
- `fuse` votes at 5 cm. A primary point is only eligible for culling where the other scenes actually
  looked, and is then kept only if `min_support` scenes contain it. Voxels every scene agrees on that the
  primary missed are added back, filling holes behind furniture.
- `mesh_filter` takes a second opinion from the collider mesh, which is built by a different process than
  the splats, so agreement there is real evidence.

On synthetic scenes with 6000 injected floaters this culls all 6000 and about 7% of genuine points, with
registration overlap 0.95 and higher. The 7% is sampling noise between independently sampled clouds and
should be much lower on dense real splats — measure it.

## Do this

1. Replace the yaw grid with a proper refinement: take the best voxel-overlap candidate and run point-to-
   plane ICP for a few iterations. 5° yaw granularity leaves up to 17 cm of smear at 4 m.
2. Report quality to the operator. Write `jobs/<id>/marble/fusion.json` into the UI: how many points each
   scene contributed, the overlap score per scene, culled and added counts. Right now it only reaches the
   log.
3. Decide `min_support` from evidence rather than the current constant 2. With five reconstructions,
   requiring 3 may be better; measure against a hand-labelled occupancy grid of one real room.
4. Handle the case where a single-image world reconstructs a *different* room. `register` already refuses
   below 0.30 overlap; confirm that threshold on real data and surface the rejection in the UI.
5. Consider dropping the per-photo worlds entirely if they never help — they cost credits and minutes.
   Measure before deciding.

## Done when

There is a number: occupancy IoU against a hand-labelled grid of one real room, with cross-checking on
and off, in this file.
