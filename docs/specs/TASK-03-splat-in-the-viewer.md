# Task 03 — Put the real splat in the 3D view

**Owner:** one agent. **Depends on:** 01.

The room currently renders as translucent grey obstacle boxes with photo A on a billboard behind them. It
reads fine but it is not the room.

## What exists

`static/index.html` → `loadBackdrop()` already tries to import `SplatMesh` from
`https://cdn.jsdelivr.net/npm/@sparkjsdev/spark/dist/spark.module.js`, sets `matrixAutoUpdate = false`,
and applies `world.T_atlas_to_sim`. On any failure it falls back to the billboard. `world.splat_url`
points at the SPZ or PLY that `marble.fetch_scene` downloaded into `jobs/<id>/marble/`. Obstacle boxes
hide themselves when a splat is present, and `window.toggleBoxes()` brings them back.

## Do this

1. Confirm the Spark import URL and the `SplatMesh` constructor signature against the current Spark
   release; the version pinned here was not verified. Spark 2.0 added streaming, which is worth using for
   a 2M-splat scene.
2. Get the transform right. Press `f` to cycle the four axis-sign flips; the splat floor should sit on the
   grey floor plane. When you find the right one, bake it into `fusion.MARBLE_TO_ZUP` or into
   `T_atlas_to_sim` rather than leaving it as a keystroke.
3. Load `collider.glb` as a shadow-receiving invisible mesh, or as a wireframe on a toggle, so obstacle
   boxes can be checked against the real geometry.
4. Watch the frame rate with people and robots moving. If 2M splats will not hold 60 fps on the demo
   laptop, request the ~500k export instead — `marble.export` already takes a `resolution` argument.

## Done when

The 3D view shows the actual room, the floor lines up with the simulated floor without pressing anything,
and the scene holds 60 fps with three robots active.
