"""Skill runner. One skill at a time per robot; tick() returns None while running and a result dict
when it finishes. The base is held still during every manipulation skill."""
import math, logging
import numpy as np
from planning import wrap
from sim import OBJECTS, surface_top, on_what

log = logging.getLogger("skills")

TIMEOUT_S = 45.0


def clamp(v, lo, hi): return lo if v < lo else (hi if v > hi else v)


def reach_geometry(rx, ry, ryaw, ox, oy):
    d = math.hypot(ox - rx, oy - ry)
    dtheta = wrap(math.atan2(oy - ry, ox - rx) - ryaw)
    lateral = d * math.sin(dtheta)
    reach = clamp(d * math.cos(dtheta) - 0.40, 0.0, 0.55)
    in_reach = (0.35 <= d <= 0.95) and abs(dtheta) <= math.radians(30) and abs(lateral) <= 0.12
    return d, dtheta, lateral, reach, in_reach


def precondition_reach(d, dtheta, lateral):
    if d < 0.35: return "too_close"
    if d > 0.95: return "too_far"
    if abs(dtheta) > math.radians(30): return "not_facing"
    if abs(lateral) > 0.12: return "not_facing"
    return None


def ray_box_exit(center, size, direction):
    dx = max(abs(direction[0]), 1e-6); dy = max(abs(direction[1]), 1e-6)
    return min(size[0] / 2 / dx, size[1] / 2 / dy)


class SkillRunner:
    """Owned by Runtime, one per robot."""

    def __init__(self, rt, k):
        self.rt = rt
        self.k = k
        self.action = None
        self.phase = None
        self.t0 = 0.0
        self.tp = 0.0
        self.params = {}
        self.result = None

    @property
    def busy(self):
        return self.phase is not None

    def start(self, action, t):
        self.action = dict(action)
        self.t0 = t
        self.tp = t
        self.params = {}
        self.result = None
        a = action.get("action")
        if a == "navigate_to":
            self.phase = "drive"
            self._setup_navigate(action.get("target"), t)
        elif a == "pick":
            self.phase = "check"
        elif a == "place":
            self.phase = "check"
        elif a == "pour":
            self.phase = "check"
        elif a == "say":
            self.phase = None
            txt = str(action.get("text") or "")[:80]
            self.rt.push_event(f"robot {self.k} says: {txt}")
            self.result = {"ok": True, "reason": "said", "detail": {"text": txt}}
        elif a == "done":
            self.phase = None
            self.result = {"ok": True, "reason": "done", "detail": {}}
        else:
            self.phase = None
            self.result = {"ok": False, "reason": "bad_action", "detail": {"action": a}}
        return self.result

    # ------------------------------------------------------------ navigate
    def _standoff(self, ox, oy):
        """Best free stance around an object: close enough for the arm (0.35-0.95 m) and reachable.
        A single point snapped onto the inflated grid lands 1.0 m+ away next to a counter, which is
        exactly the too_far failure, so sample a ring and score by how close the snapped cell lands."""
        from planning import snap_free_xy, blocked_at
        rt = self.rt; r = rt.robots[self.k]
        base = math.atan2(r.y - oy, r.x - ox)
        best, bd = None, 1e9
        for dist in (0.60, 0.70, 0.80, 0.52):
            for da in (0, 0.35, -0.35, 0.7, -0.7, 1.2, -1.2, 1.9, -1.9, 2.6, -2.6, math.pi):
                a = base + da
                px, py = ox + dist * math.cos(a), oy + dist * math.sin(a)
                if blocked_at(rt.goal_grid_robot, rt.origin, px, py):
                    continue
                sx, sy = snap_free_xy(rt.goal_grid_robot, rt.origin, px, py)
                d = math.hypot(sx - ox, sy - oy)
                if not (0.36 <= d <= 0.92):
                    continue
                cost = abs(d - 0.72) + 0.25 * abs(da)
                if cost < bd:
                    best, bd = (sx, sy), cost
        if best is not None:
            return best
        v = np.array([r.x - ox, r.y - oy]); n = np.linalg.norm(v)
        v = v / n if n > 1e-6 else np.array([1.0, 0.0])
        return (ox + v[0] * 0.7, oy + v[1] * 0.7)

    def _target_xy(self, target):
        rt = self.rt
        r = rt.robots[self.k]
        if isinstance(target, str) and target in rt.objects_by_id():
            o = rt.sim.object_pose(target)
            return self._standoff(o["x"], o["y"]), "object"
        if isinstance(target, str) and target in rt.landmarks:
            lm = rt.landmarks[target]
            c = np.array(lm["center_xy"], float)
            v = np.array([r.x, r.y]) - c; n = np.linalg.norm(v)
            v = v / n if n > 1e-6 else np.array([1.0, 0.0])
            t_exit = ray_box_exit(c, lm["size_xy"], v)
            return tuple(c + v * (t_exit + 0.55)), "landmark"
        if isinstance(target, dict) and "x" in target:
            return (float(target["x"]), float(target["y"])), "point"
        if isinstance(target, (int, float)) and int(target) in rt.cast.by_id:
            a = rt.cast.by_id[int(target)]
            return (a.x, a.y), "person"
        return None, None

    def _setup_navigate(self, target, t):
        rt = self.rt
        xy, kind = self._target_xy(target)
        if xy is None:
            self.phase = None
            self.result = {"ok": False, "reason": "unknown_target", "detail": {"target": target}}
            return
        from planning import snap_free_xy
        g = snap_free_xy(rt.goal_grid_robot, rt.origin, xy[0], xy[1])
        r = rt.robots[self.k]
        r.goal = g
        r.follow = int(target) if kind == "person" else None
        r.path = []
        r.last_plan = -1
        self.params = dict(target=target, kind=kind, best_d=1e9, best_t=t,
                           arrive=0.85 if kind == "person" else 0.20)

    def _face_target(self, t):
        rt = self.rt; r = rt.robots[self.k]
        xy, kind = self._target_xy(self.params.get("target"))
        if self.params.get("kind") == "object" or self.params.get("kind") == "landmark":
            # face the thing itself, not the standoff point
            tgt = self.params.get("target")
            if tgt in rt.objects_by_id():
                o = rt.sim.object_pose(tgt); fx, fy = o["x"], o["y"]
            else:
                fx, fy = rt.landmarks[tgt]["center_xy"]
        elif self.params.get("kind") == "person":
            a = rt.cast.by_id[int(self.params["target"])]; fx, fy = a.x, a.y
        else:
            fx, fy = xy if xy else (r.x + 1, r.y)
        err = wrap(math.atan2(fy - r.y, fx - r.x) - r.yaw)
        if abs(err) < math.radians(3):
            rt.sim.cmd_robot(self.k, 0.0, 0.0)
            if t - self.tp > 0.3:
                return True
        else:
            self.tp = t
            rt.sim.cmd_robot(self.k, 0.0, clamp(2.5 * err, -1.2, 1.2))
        return False

    # ------------------------------------------------------------ tick
    def tick(self, t):
        if self.phase is None:
            return None
        if t - self.t0 > TIMEOUT_S:
            return self._finish(False, "timeout", {"phase": self.phase})
        fn = getattr(self, f"_tick_{self.action['action']}")
        return fn(t)

    def _finish(self, ok, reason, detail=None):
        self.phase = None
        self.rt.sim.cmd_robot(self.k, 0.0, 0.0)
        self.result = {"ok": bool(ok), "reason": reason, "detail": detail or {}}
        return self.result

    def _tick_navigate_to(self, t):
        rt = self.rt; r = rt.robots[self.k]
        if self.phase == "drive":
            if self.params.get("kind") == "person":
                a = rt.cast.by_id[int(self.params["target"])]
                r.goal = (a.x, a.y)
            elif self.params.get("kind") == "object":
                xy, _ = self._target_xy(self.params["target"])
                if xy:
                    from planning import snap_free_xy
                    r.goal = snap_free_xy(rt.goal_grid_robot, rt.origin, xy[0], xy[1])
            if r.goal is None:
                return self._finish(False, "unreachable", {})
            d = math.dist((r.x, r.y), r.goal)
            if d < self.params["best_d"] - 0.10:
                self.params["best_d"] = d; self.params["best_t"] = t
            if t - self.params["best_t"] > 8.0:
                return self._finish(False, "blocked", {"d": round(d, 2)})
            ok = rt.drive_robot(self.k, t, arrive_radius=self.params["arrive"],
                                ignore_pid=int(self.params["target"]) if self.params.get("kind") == "person" else None)
            if ok == "unreachable":
                return self._finish(False, "unreachable", {})
            if ok == "arrived":
                self.phase = "face"; self.tp = t
            return None
        if self.phase == "face":
            if self._face_target(t):
                if self.params.get("kind") == "object":
                    self.phase = "creep"; self.tp = t
                    return None
                return self._finish(True, "arrived", {"x": round(r.x, 2), "y": round(r.y, 2)})
            return None
        if self.phase == "creep":
            o = rt.sim.object_pose(self.params["target"])
            d, dth, lat, reach, ok = reach_geometry(r.x, r.y, r.yaw, o["x"], o["y"])
            if ok or t - self.tp > 3.0 or d < 0.40:
                rt.sim.cmd_robot(self.k, 0.0, 0.0)
                return self._finish(True, "arrived", {"x": round(r.x, 2), "y": round(r.y, 2), "d": round(d, 2)})
            rt.sim.cmd_robot(self.k, 0.25 if d > 0.85 else 0.0, clamp(2.0 * dth, -0.8, 0.8))
            return None
        return None

    def _tick_pick(self, t):
        rt = self.rt; r = rt.robots[self.k]
        oid = self.action.get("object")
        if self.phase == "check":
            if oid not in rt.objects_by_id():
                return self._finish(False, "unknown_object", {"object": oid})
            if r.holding and r.holding != oid:
                return self._finish(False, "holding_other", {"holding": r.holding})
            if rt.held_by(oid) not in (None, r.id):
                return self._finish(False, "object_held", {"by": rt.held_by(oid)})
            vx, vy = rt.sim.robot_vel(self.k)
            if math.hypot(vx, vy) > 0.05:
                return self._finish(False, "moving", {"speed": round(math.hypot(vx, vy), 2)})
            o = rt.sim.object_pose(oid)
            d, dth, lat, reach, ok = reach_geometry(r.x, r.y, r.yaw, o["x"], o["y"])
            bad = precondition_reach(d, dth, lat)
            if bad:
                return self._finish(False, bad, {"d": round(d, 2), "yaw_err_deg": round(math.degrees(dth), 1)})
            self.params["reach"] = reach
            self.phase = "extend"; self.tp = t
            rt.sim.set_arm(self.k, reach, 0.0)
            return None
        rt.sim.cmd_robot(self.k, 0.0, 0.0)
        if self.phase == "extend" and t - self.tp >= 0.8:
            rt.sim.hold(self.k, oid); r.holding = oid
            rt.set_held(oid, r.id)
            self.phase = "retract"; self.tp = t
            rt.sim.set_arm(self.k, 0.15, 0.0)
            return None
        if self.phase == "retract" and t - self.tp >= 0.8:
            return self._finish(True, "picked", {"object": oid})
        return None

    def _tick_place(self, t):
        rt = self.rt; r = rt.robots[self.k]
        oid = self.action.get("object") or r.holding
        target = self.action.get("target")
        if self.phase == "check":
            if not oid or r.holding != oid:
                return self._finish(False, "not_holding", {"holding": r.holding, "object": oid})
            if isinstance(target, (int, float)) and int(target) in rt.cast.by_id:
                a = rt.cast.by_id[int(target)]
                d = math.dist((r.x, r.y), (a.x, a.y))
                if d > 1.0:
                    return self._finish(False, "too_far", {"d": round(d, 2)})
                self.params = dict(kind="person", pid=int(target), reach=0.35)
            elif isinstance(target, str) and target in rt.landmarks:
                lm = rt.landmarks[target]
                c = np.array(lm["center_xy"], float); sx, sy = lm["size_xy"]
                dx = max(abs(r.x - c[0]) - sx / 2, 0.0); dy = max(abs(r.y - c[1]) - sy / 2, 0.0)
                de = math.hypot(dx, dy)
                if de > 0.95:
                    return self._finish(False, "too_far", {"d": round(de, 2)})
                self.params = dict(kind="landmark", lm=target, reach=clamp(de + 0.10 - 0.40, 0.0, 0.55))
            else:
                return self._finish(False, "unknown_target", {"target": target})
            self.phase = "extend"; self.tp = t
            rt.sim.set_arm(self.k, self.params["reach"], 0.0)
            return None
        rt.sim.cmd_robot(self.k, 0.0, 0.0)
        if self.phase == "extend" and t - self.tp >= 0.8:
            if self.params["kind"] == "person":
                rt.sim.data.eq_active[rt.sim.weld[(self.k, oid)]] = 0
                rt.sim.set_object_collisions(oid, False)
                rt.set_held(oid, f"person_{self.params['pid']}")
                r.holding = None
                self.phase = "retract"; self.tp = t
                rt.sim.set_arm(self.k, 0.15, 0.0)
                self.params["delivered"] = True
                return None
            hp = rt.sim.hand_pos(self.k)
            z = surface_top(rt.world, (float(hp[0]), float(hp[1])))
            rt.sim.release(self.k, oid, z)
            rt.set_held(oid, None)
            r.holding = None
            self.phase = "settle"; self.tp = t
            rt.sim.set_arm(self.k, 0.15, 0.0)
            return None
        if self.phase == "retract" and t - self.tp >= 0.8:
            return self._finish(True, "handed_over", {"to": self.params.get("pid"), "delivered": True})
        if self.phase == "settle" and t - self.tp >= 1.0:
            o = rt.sim.object_pose(oid)
            half = OBJECTS[oid]["half_h"]
            where = on_what(rt.world, o, half)
            if o["upright"] <= 0.9:
                return self._finish(False, "tipped", {"upright": round(o["upright"], 2)})
            if where == "floor" or o["z"] < 0.3:
                return self._finish(False, "dropped", {"z": round(o["z"], 2)})
            if o["speed"] >= 0.05:
                return self._finish(False, "dropped", {"speed": round(o["speed"], 2)})
            if where != self.params["lm"]:
                return self._finish(False, "missed", {"on": where})
            return self._finish(True, "placed", {"on": where})
        return None

    def _tick_pour(self, t):
        rt = self.rt; r = rt.robots[self.k]
        if self.phase == "check":
            if r.holding != "pot":
                return self._finish(False, "not_holding_pot", {"holding": r.holding})
            if rt.held_by("cup") is not None:
                return self._finish(False, "cup_held", {"by": rt.held_by("cup")})
            c = rt.sim.object_pose("cup")
            if c["z"] < 0.3:
                return self._finish(False, "cup_on_floor", {"z": round(c["z"], 2)})
            if c["upright"] < 0.9:
                return self._finish(False, "cup_tipped", {"upright": round(c["upright"], 2)})
            if rt.cup_filled:
                return self._finish(False, "already_full", {})
            vx, vy = rt.sim.robot_vel(self.k)
            if math.hypot(vx, vy) > 0.05:
                return self._finish(False, "moving", {"speed": round(math.hypot(vx, vy), 2)})
            d, dth, lat, reach, ok = reach_geometry(r.x, r.y, r.yaw, c["x"], c["y"])
            bad = precondition_reach(d, dth, lat)
            if bad:
                return self._finish(False, bad, {"d": round(d, 2), "yaw_err_deg": round(math.degrees(dth), 1)})
            self.params = dict(reach=reach, lateral=lat)
            self.phase = "extend"; self.tp = t
            rt.sim.set_arm(self.k, reach, 0.0)
            return None
        rt.sim.cmd_robot(self.k, 0.0, 0.0)
        if self.phase == "extend" and t - self.tp >= 0.8:
            self.phase = "tilt"; self.tp = t
            rt.sim.set_arm(self.k, self.params["reach"], 1.3)
            return None
        if self.phase == "tilt" and t - self.tp >= 1.0:
            self.phase = "hold"; self.tp = t
            rng = np.random.default_rng(int(rt.seed) * 1000 + int(rt.ep_step))
            miss = abs(self.params["lateral"]) + abs(float(rng.normal(0, 0.02)))
            self.params["miss"] = miss
            if miss < 0.10:
                rt.cup_filled = True
                rt.sim.set_cup_filled(True)
                rt.push_event(f"robot {self.k} poured coffee")
            else:
                hp = rt.sim.hand_pos(self.k)
                rt.sim.show_stain(float(hp[0]), float(hp[1]))
                rt.spilled = True
                rt.push_event(f"robot {self.k} spilled the coffee")
            return None
        if self.phase == "hold" and t - self.tp >= 1.5:
            self.phase = "untilt"; self.tp = t
            rt.sim.set_arm(self.k, self.params["reach"], 0.0)
            return None
        if self.phase == "untilt" and t - self.tp >= 1.0:
            self.phase = "retract"; self.tp = t
            rt.sim.set_arm(self.k, 0.15, 0.0)
            return None
        if self.phase == "retract" and t - self.tp >= 0.6:
            miss = self.params.get("miss", 1.0)
            if miss < 0.10:
                return self._finish(True, "poured", {"miss_m": round(miss, 3)})
            return self._finish(False, "spill", {"miss_m": round(miss, 3)})
        return None
