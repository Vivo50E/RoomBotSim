# Task 02 — Recover camera poses so people land where they actually stood

**Owner:** one agent. **Depends on:** 01.

Right now people are placed by the weakest path in the system, and it shows.

## The problem

The Marble world object exposes splats, meshes and a panorama, but the pipeline never gets the poses of
the four input cameras back. Without them there is no depth to unproject a person's feet into, so
`server.pipeline` falls back to `perception.fallback_layout`: the vision model estimates a room rectangle
and everyone's position in it, and `geometry.fit_fallback_to_grid` drops that rectangle onto the
reconstructed grid by matching aspect ratio and picking one of four 90° rotations.

That is a guess. It is good enough that the operator can drag people to the right spot in the confirm
stage, and no worse than nothing, but it is the main source of "why is that person inside the couch".

## Three routes, in order of preference

1. **Ask Marble for the poses.** Re-read the API reference and the OpenAPI spec at
   `https://docs.worldlabs.ai/api/reference/openapi.md`. If input camera extrinsics and intrinsics are
   returned anywhere, fill `Recon.cameras` in `recon._marble` using `recon.to_cam_to_world`, `gl_to_cv`
   and `K_for`. Everything downstream already handles the case where cameras exist.
2. **Register the photos against the splat.** Render the splat or the collider mesh from candidate poses
   and match against each photo (a feature matcher over rendered views, or a small pose regressor). Solve
   for four poses, then produce a depth map per view by rasterising the mesh. Fill `Recon.depths` in
   metres with 0 for invalid. `geometry.unproject` and `scale_from_door` then start working with no other
   changes, and `place_fixture` will put doors and whiteboards in the right place too.
3. **Run a separate two-view reconstructor** on the same photos purely for poses and depth, and keep
   Marble for the splat.

## Done when

`Recon.cameras` has four entries and `Recon.depths` has four maps, the people-from-depth branch in
`server.pipeline` fires (you will see people placed without `fallback_layout` being called), and a
standing person's position is within 0.5 m of where they actually stood.
