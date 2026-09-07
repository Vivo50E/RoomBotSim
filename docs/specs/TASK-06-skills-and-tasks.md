# Task 06 — More skills, more objects, more task types

**Owner:** one agent. **Depends on:** nothing.

## What exists

Six skills in `roombotsim/skills.py`, each with real preconditions checked in a fixed order so the first
failure is the reported reason: `navigate_to`, `pick`, `place`, `pour`, `say`, `done`. Manipulation is
skill level — `pick` welds the object to the hand via a MuJoCo equality constraint, `place` unwelds it
onto whatever surface is under the hand, `pour` is a tilt animation with an analytic outcome. Two objects,
a cup and a pot, spawned just inside the front edge of the counter by `sim.spawn_objects`.

The geometry that decides whether a skill can run:

```
d       = distance base centre to object
dtheta  = heading error toward it
lateral = d · sin(dtheta)
in_reach = 0.35 ≤ d ≤ 0.95  and  |dtheta| ≤ 30°  and  |lateral| ≤ 0.12
```

`navigate_to` on an object samples a ring of stances and picks one that lands 0.72 m away, then creeps
forward if the arm still cannot reach. That ring exists because a single standoff point snapped onto the
inflated grid lands past 1.0 m next to a counter, which was a constant `too_far` failure.

## Do this

1. **More objects.** `sim.OBJECTS` is a dict of geometry, mass, colour, half-height and hold offset. Add a
   plate, a bottle, a book. Each needs a spawn surface and an entry in the weld and contact-exclude lists,
   both generated from the same dict, so adding one is a few lines.
2. **More task types.** `episodes.evaluate` has one branch per task type. `bring_object` and `go_to` exist
   but have no oracle, which blocks relabeling (see Task 05). Write oracles for them in `policy.py`
   alongside `oracle()`, keeping them stateless so they can relabel any recorded state.
3. **`open` and `close`.** A door or cabinet as a hinged body would make navigation genuinely harder and
   is the most interesting missing skill.
4. **Motor-level policy kind.** The recorder already stores a `teleop` array of `[t, v, ω]` samples per
   navigate segment in human mode. A second policy kind acting at 10 Hz on `(v, ω, grasp)` with the
   robot-eye frame as observation would reuse the whole recorder unchanged. `sim.render_eye` renders that
   frame already; it needs `MUJOCO_GL` set and is silently skipped when there is no GL context.
5. **Bipedal locomotion is out of scope.** Both rigs ride the same planar base on purpose. Do not open
   that door.

## Done when

At least two task types have working oracles, and `tests/run_all.py` has a case per new skill that proves
both its success path and one precondition failure.
