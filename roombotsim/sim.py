"""MJCF builder + MuJoCo wrapper.

People are mocap capsules driven by the crowd controller. Robots ride a planar base (slide x, slide y,
hinge yaw) with a two-DoF arm (reach slide, wrist tilt); the legs of the Spot rig and the limbs of the
humanoid rig are visual only, so locomotion is never simulated at the joint level. Manipulation is
skill level: pick welds an object to the hand, place unwelds it onto the surface under the hand.
"""
import mujoco, numpy as np, math

PERSON_R, PERSON_H = 0.22, 1.70
CAP_HALF = (PERSON_H - 2 * PERSON_R) / 2
N_ROBOTS = 3
ARM_Z = 1.25
ROBOT_TYPES = {1: "spot", 2: "humanoid", 3: "spot"}

OBJECTS = {
    "cup": dict(size="0.04 0.05", mass=0.2, rgba="0.92 0.92 0.92 1", half_h=0.05, hold_off=-0.08),
    "pot": dict(size="0.07 0.10", mass=0.8, rgba="0.15 0.15 0.15 1", half_h=0.10, hold_off=-0.12),
}
COFFEE_RGBA = [0.45, 0.28, 0.12, 1.0]


def rgba(hexs, a=1.0):
    h = hexs.lstrip("#")
    return f"{int(h[0:2],16)/255:.3f} {int(h[2:4],16)/255:.3f} {int(h[4:6],16)/255:.3f} {a}"


def robot_xml(k, rtype):
    """Base + camera + collision box are shared; the rest is a visual costume (contype 0)."""
    base_visible = "0.15 0.15 0.15 1" if rtype == "box" else "0.15 0.15 0.15 0"
    common = f'''<joint name="robot_{k}_x"   type="slide" axis="1 0 0" damping="2"/>
  <joint name="robot_{k}_y"   type="slide" axis="0 1 0" damping="2"/>
  <joint name="robot_{k}_yaw" type="hinge" axis="0 0 1" damping="0.5"/>
  <camera name="robot_{k}_eye" pos="0.05 0 1.35" xyaxes="0 -1 0 0 0 1"/>
  <geom name="robot_{k}_base" type="box" pos="0 0 0.11" size="0.17 0.165 0.09" mass="20" rgba="{base_visible}"/>'''
    if rtype == "humanoid":
        visual = f'''<geom name="robot_{k}_legL" type="capsule" fromto="0 0.10 0.06 0 0.10 0.55" size="0.07" mass="4" rgba="0.85 0.85 0.9 1" contype="0" conaffinity="0"/>
  <geom name="robot_{k}_legR" type="capsule" fromto="0 -0.10 0.06 0 -0.10 0.55" size="0.07" mass="4" rgba="0.85 0.85 0.9 1" contype="0" conaffinity="0"/>
  <geom name="robot_{k}_torso" type="capsule" fromto="0 0 0.60 0 0 1.40" size="0.14" mass="20" rgba="0.85 0.85 0.9 1" contype="0" conaffinity="0"/>
  <geom name="robot_{k}_head" type="sphere" pos="0 0 1.60" size="0.11" mass="3" rgba="0.85 0.85 0.9 1" contype="0" conaffinity="0"/>
  <geom name="robot_{k}_visor" type="box" pos="0.09 0 1.62" size="0.02 0.07 0.02" mass="0.1" rgba="0.1 0.3 0.9 1" contype="0" conaffinity="0"/>
  <geom name="robot_{k}_armL" type="capsule" fromto="0 0.20 1.35 0 0.22 0.75" size="0.05" mass="2" rgba="0.85 0.85 0.9 1" contype="0" conaffinity="0"/>'''
    else:
        # Spot: 1.1 x 0.5 m body at hip height, four legs, front sensor head, arm mast for the manipulator
        body = "0.99 0.75 0.12 1"; dark = "0.16 0.16 0.18 1"
        legs = []
        for nm, sx, sy in (("fl", 0.36, 0.17), ("fr", 0.36, -0.17), ("hl", -0.36, 0.17), ("hr", -0.36, -0.17)):
            legs.append(f'<geom name="robot_{k}_{nm}u" type="capsule" fromto="{sx} {sy} 0.53 {sx-0.06} {sy} 0.30" size="0.045" mass="1" rgba="{dark}" contype="0" conaffinity="0"/>')
            legs.append(f'<geom name="robot_{k}_{nm}l" type="capsule" fromto="{sx-0.06} {sy} 0.30 {sx+0.02} {sy} 0.05" size="0.033" mass="1" rgba="{dark}" contype="0" conaffinity="0"/>')
        legs = "\n  ".join(legs)
        visual = f'''<geom name="robot_{k}_body" type="box" pos="0 0 0.60" size="0.30 0.16 0.09" mass="18" rgba="{body}" contype="0" conaffinity="0"/>
  <geom name="robot_{k}_hip_f" type="box" pos="0.30 0 0.58" size="0.07 0.19 0.07" mass="2" rgba="{dark}" contype="0" conaffinity="0"/>
  <geom name="robot_{k}_hip_h" type="box" pos="-0.30 0 0.58" size="0.07 0.19 0.07" mass="2" rgba="{dark}" contype="0" conaffinity="0"/>
  {legs}
  <geom name="robot_{k}_head" type="box" pos="0.36 0 0.66" size="0.08 0.11 0.06" mass="1" rgba="{body}" contype="0" conaffinity="0"/>
  <geom name="robot_{k}_face" type="box" pos="0.44 0 0.66" size="0.01 0.08 0.035" mass="0.1" rgba="0.1 0.6 0.9 1" contype="0" conaffinity="0"/>
  <geom name="robot_{k}_mast" type="cylinder" pos="-0.12 0 0.92" size="0.045 0.30" mass="3" rgba="{body}" contype="0" conaffinity="0"/>
  <geom name="robot_{k}_shoulder" type="sphere" pos="-0.02 0 1.25" size="0.075" mass="1" rgba="{dark}" contype="0" conaffinity="0"/>'''
    arm = f'''<body name="robot_{k}_arm" pos="0.10 0 {ARM_Z}">
    <joint name="robot_{k}_reach" type="slide" axis="1 0 0" range="0 0.55" damping="20" armature="0.5"/>
    <geom name="robot_{k}_armgeom" type="capsule" fromto="-0.05 0 0 0.25 0 0" size="0.03" mass="1" rgba="0.6 0.6 0.65 1"/>
    <body name="robot_{k}_hand" pos="0.30 0 0">
      <joint name="robot_{k}_tilt" type="hinge" axis="1 0 0" range="-1.6 1.6" damping="1" armature="0.1"/>
      <geom name="robot_{k}_handgeom" type="sphere" size="0.04" mass="0.3" rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>
    </body>
  </body>'''
    return f'<body name="robot_{k}" pos="0 0 0">\n  {common}\n  {visual}\n  {arm}\n</body>'


def objects_xml(spawns):
    out = []
    for oid, (x, y, z) in spawns.items():
        o = OBJECTS[oid]
        out.append(f'<body name="{oid}" pos="{x:.3f} {y:.3f} {z:.3f}"><freejoint name="{oid}_free"/>'
                   f'<geom name="{oid}_geom" type="cylinder" size="{o["size"]}" mass="{o["mass"]}" rgba="{o["rgba"]}"/></body>')
    out.append('<geom name="stain" type="cylinder" pos="0 0 -1" size="0.18 0.002" rgba="0.35 0.2 0.08 1" contype="0" conaffinity="0"/>')
    return "\n".join(out)


def equality_xml(n_robots, spawns):
    return "\n".join(
        f'<weld name="robot_{k}_hold_{oid}" body1="robot_{k}_hand" body2="{oid}" '
        f'relpose="0 0 {OBJECTS[oid]["hold_off"]} 1 0 0 0" active="false"/>'
        for k in range(1, n_robots + 1) for oid in spawns)


def contact_xml(n_robots, spawns, people=()):
    ex = [f'<exclude body1="{oid}" body2="robot_{k}{suf}"/>'
          for k in range(1, n_robots + 1) for oid in spawns for suf in ("", "_arm", "_hand")]
    # people are mocap capsules driven kinematically; a person brushing a table must not sweep the
    # coffee onto the floor, so they never collide with the objects at all
    ex += [f'<exclude body1="person_{p["id"]}" body2="{oid}"/>' for p in people for oid in spawns]
    if "cup" in spawns and "pot" in spawns:
        ex.append('<exclude body1="pot" body2="cup"/>')
    return "\n".join(ex)


def actuators_xml(n_robots):
    return "\n".join(f'''<velocity name="robot_{k}_vx" joint="robot_{k}_x"   kv="300"/>
<velocity name="robot_{k}_vy" joint="robot_{k}_y"   kv="300"/>
<velocity name="robot_{k}_wz" joint="robot_{k}_yaw" kv="30"/>
<position name="robot_{k}_reach_a" joint="robot_{k}_reach" kp="200" kv="30"/>
<position name="robot_{k}_tilt_a"  joint="robot_{k}_tilt"  kp="10"  kv="1"/>''' for k in range(1, n_robots + 1))


def build_mjcf(world, people, robot_types=None, spawns=None, n_robots=N_ROBOTS):
    robot_types = robot_types or ROBOT_TYPES
    spawns = spawns if spawns is not None else {}
    geoms = []
    for o in world["obstacles"]:
        cx, cy = o["center_xy"]; sx, sy = o["size_xy"]; h = o["height_m"]
        geoms.append(f'<geom name="{o["id"]}" type="box" pos="{cx:.3f} {cy:.3f} {h/2:.3f}" '
                     f'size="{sx/2:.3f} {sy/2:.3f} {h/2:.3f}" rgba="0.55 0.58 0.70 1"/>')
    for w in world["walls"]:
        cx, cy = w["center_xy"]; sx, sy = w["size_xy"]; h = w["height_m"]
        geoms.append(f'<geom name="{w["id"]}" type="box" pos="{cx:.3f} {cy:.3f} {h/2:.3f}" '
                     f'size="{sx/2:.3f} {sy/2:.3f} {h/2:.3f}" rgba="0.8 0.8 0.8 0.3"/>')
    bodies = []
    for p in people:
        x, y = p["pos_xy"]; c = rgba(p["color"])
        bodies.append(f'''<body name="person_{p["id"]}" mocap="true" pos="{x:.3f} {y:.3f} {PERSON_H/2:.3f}">
  <geom name="person_{p["id"]}_geom" type="capsule" size="{PERSON_R} {CAP_HALF:.3f}" rgba="{c}"/>
  <geom name="person_{p["id"]}_nose" type="sphere" size="0.06" pos="{PERSON_R} 0 0.70" rgba="0 0 0 1" contype="0" conaffinity="0"/>
</body>''')
    for k in range(1, n_robots + 1):
        bodies.append(robot_xml(k, robot_types.get(k, "spot")))
    nl = "\n"
    return f'''<mujoco model="roombotsim">
<compiler angle="radian"/>
<option timestep="0.01" gravity="0 0 -9.81" integrator="implicitfast"/>
<default><geom condim="3" friction="0.6 0.005 0.0001"/></default>
<worldbody>
<light pos="0 0 4" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
<geom name="floor" type="plane" size="0 0 0.05" rgba="0.9 0.9 0.9 1"/>
{nl.join(geoms)}
{nl.join(bodies)}
{objects_xml(spawns)}
</worldbody>
<actuator>
{actuators_xml(n_robots)}
</actuator>
<equality>
{equality_xml(n_robots, spawns)}
</equality>
<contact>
{contact_xml(n_robots, spawns, people)}
</contact>
</mujoco>'''


# ---------------------------------------------------------------- surfaces

def surface_top(world, xy):
    x, y = xy[0], xy[1]; best = 0.0
    for o in world["obstacles"]:
        cx, cy = o["center_xy"]; sx, sy = o["size_xy"]
        if abs(x - cx) <= sx / 2 and abs(y - cy) <= sy / 2:
            best = max(best, float(o["height_m"]))
    return best


def on_what(world, obj, half_h):
    x, y, z = obj["x"], obj["y"], obj["z"]
    for lm in world["landmarks"]:
        cx, cy = lm["center_xy"]; sx, sy = lm["size_xy"]
        if abs(x - cx) <= sx / 2 + 0.15 and abs(y - cy) <= sy / 2 + 0.15:
            if surface_top(world, (x, y)) >= z - half_h - 0.1:
                return lm["id"]
    if z < 0.3:
        return "floor"
    return None


def surface_candidates(world, lm, res=0.05):
    """Points inside a landmark footprint that actually have a box under them at graspable height."""
    cx, cy = lm["center_xy"]; sx, sy = lm["size_xy"]
    out = []
    nx = max(int(sx / res), 1); ny = max(int(sy / res), 1)
    for i in range(nx + 1):
        for j in range(ny + 1):
            x = cx - sx / 2 + i * sx / max(nx, 1)
            y = cy - sy / 2 + j * sy / max(ny, 1)
            h = surface_top(world, (x, y))
            if 0.35 <= h <= 1.35:                 # table and counter height, not the floor or a wall
                out.append((x, y, h))
    return out


# Kept as a private alias for scripts written before surface placement became part
# of the skill runner's public contract.
_surface_candidates = surface_candidates


def spawn_objects(world):
    """Put the pot and cup on a real surface the robot can reach.

    The earlier version walked inward from a landmark's front edge and, failing to find anything, fell
    back to assuming 0.75 m. On a reconstructed room that puts the coffee in mid-air: it drops to the
    floor and every pour then fails `cup_on_floor` forever. So only ever spawn where a box really is.
    """
    lms = world["landmarks"]
    ox, oy = world["origin_xy"]; nx, ny = world["size_cells"]; res = world["resolution_m"]
    center = np.array([ox + nx * res / 2, oy + ny * res / 2])

    ranked = sorted(lms, key=lambda l: (0 if l["label"].startswith("counter") else
                                        1 if l["label"].startswith(("table", "desk")) else 2,
                                        -l["size_xy"][0] * l["size_xy"][1]))
    best = None
    for lm in ranked:
        cand = surface_candidates(world, lm)
        if not cand:
            continue
        c = np.array(lm["center_xy"], float)
        toward = center - c; n = np.linalg.norm(toward)
        toward = toward / n if n > 1e-6 else np.array([1.0, 0.0])
        # nearest the room-facing edge, so a robot can stand in front of it
        cand.sort(key=lambda p: -float(np.dot(np.array(p[:2]) - c, toward)))
        best = (lm, cand)
        break

    if best is None:
        # no labelled landmark has a usable top: take the largest graspable-height obstacle in the room
        obs = [o for o in world["obstacles"] if 0.35 <= o["height_m"] <= 1.35]
        if obs:
            o = max(obs, key=lambda o: o["size_xy"][0] * o["size_xy"][1])
            fake = dict(center_xy=o["center_xy"], size_xy=o["size_xy"], label="surface")
            cand = surface_candidates(world, fake)
            if cand:
                best = (fake, cand)
    if best is None:
        p = center + np.array([0.5, 0.0])
        return {"pot": (float(p[0] - 0.18), float(p[1]), 0.86), "cup": (float(p[0] + 0.18), float(p[1]), 0.81)}

    lm, cand = best
    c = np.array(lm["center_xy"], float)
    toward = center - c; n = np.linalg.norm(toward)
    toward = toward / n if n > 1e-6 else np.array([1.0, 0.0])
    along = np.array([-toward[1], toward[0]])
    edge = np.array(cand[0][:2]); top = cand[0][2]

    def settle(p):
        """Nudge along the surface until the spot still has a box under it."""
        for k in range(8):
            q = p - along * 0.04 * k if k % 2 else p + along * 0.04 * k
            if surface_top(world, (q[0], q[1])) >= top - 0.05:
                return q, surface_top(world, (q[0], q[1]))
        return p, top

    pot_xy, pot_h = settle(edge - 0.16 * along)
    cup_xy, cup_h = settle(edge + 0.16 * along)
    return {"pot": (float(pot_xy[0]), float(pot_xy[1]), pot_h + 0.10 + 0.01),
            "cup": (float(cup_xy[0]), float(cup_xy[1]), cup_h + 0.05 + 0.01)}


# ---------------------------------------------------------------- wrapper

class Sim:
    def __init__(self, world, people, robot_types=None, spawns=None, n_robots=N_ROBOTS):
        self.robot_types = dict(robot_types or ROBOT_TYPES)
        self.spawns = dict(spawns if spawns is not None else spawn_objects(world))
        self.world = world
        self.model = mujoco.MjModel.from_xml_string(
            build_mjcf(world, people, self.robot_types, self.spawns, n_robots))
        self.data = mujoco.MjData(self.model)
        self.n_robots = n_robots
        self.mocap_id = {p["id"]: int(self.model.body_mocapid[self.model.body(f'person_{p["id"]}').id]) for p in people}
        R = range(1, n_robots + 1)
        self.jadr = {k: [int(self.model.joint(f"robot_{k}_{ax}").qposadr[0]) for ax in ("x", "y", "yaw")] for k in R}
        self.vadr = {k: [int(self.model.joint(f"robot_{k}_{ax}").dofadr[0]) for ax in ("x", "y", "yaw")] for k in R}
        self.aid = {k: [int(self.model.actuator(f"robot_{k}_{n}").id) for n in ("vx", "vy", "wz")] for k in R}
        self.hand = {k: int(self.model.body(f"robot_{k}_hand").id) for k in R}
        self.arm_a = {k: (int(self.model.actuator(f"robot_{k}_reach_a").id),
                          int(self.model.actuator(f"robot_{k}_tilt_a").id)) for k in R}
        self.arm_q = {k: (int(self.model.joint(f"robot_{k}_reach").qposadr[0]),
                          int(self.model.joint(f"robot_{k}_tilt").qposadr[0])) for k in R}
        self.obj_q = {oid: int(self.model.joint(f"{oid}_free").qposadr[0]) for oid in self.spawns}
        self.obj_v = {oid: int(self.model.joint(f"{oid}_free").dofadr[0]) for oid in self.spawns}
        self.obj_geom = {oid: int(self.model.geom(f"{oid}_geom").id) for oid in self.spawns}
        self.weld = {(k, oid): int(self.model.equality(f"robot_{k}_hold_{oid}").id) for k in R for oid in self.spawns}
        self.stain = int(self.model.geom("stain").id)
        self.geom_name = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(self.model.ngeom)]
        ox, oy = world["origin_xy"]
        self.park = {k: (ox - 5.0, oy + 1.0 * k, 0.0) for k in R}
        for k, (x, y, yaw) in self.park.items():
            self.teleport_robot(k, x, y, yaw)
        mujoco.mj_forward(self.model, self.data)

    # people ----------------------------------------------------
    def set_person(self, pid, x, y, yaw):
        m = self.mocap_id[pid]
        self.data.mocap_pos[m] = [x, y, PERSON_H / 2]
        self.data.mocap_quat[m] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]

    # robots ----------------------------------------------------
    def robot_pose(self, k):
        q = self.data.qpos; a = self.jadr[k]; return float(q[a[0]]), float(q[a[1]]), float(q[a[2]])

    def robot_vel(self, k):
        v = self.data.qvel; a = self.vadr[k]; return float(v[a[0]]), float(v[a[1]])

    def cmd_robot(self, k, v, w):
        _, _, yaw = self.robot_pose(k); a = self.aid[k]
        self.data.ctrl[a[0]] = v * math.cos(yaw); self.data.ctrl[a[1]] = v * math.sin(yaw); self.data.ctrl[a[2]] = w

    def teleport_robot(self, k, x, y, yaw):
        a = self.jadr[k]; self.data.qpos[a[0]] = x; self.data.qpos[a[1]] = y; self.data.qpos[a[2]] = yaw
        for d in self.vadr[k]: self.data.qvel[d] = 0.0
        self.cmd_robot(k, 0.0, 0.0); mujoco.mj_forward(self.model, self.data)

    def set_arm(self, k, reach, tilt):
        a = self.arm_a[k]; self.data.ctrl[a[0]] = reach; self.data.ctrl[a[1]] = tilt

    def arm_state(self, k):
        q = self.arm_q[k]; return float(self.data.qpos[q[0]]), float(self.data.qpos[q[1]])

    def hand_pos(self, k):
        return self.data.xpos[self.hand[k]].copy()

    # objects ---------------------------------------------------
    def object_pose(self, oid):
        q = self.obj_q[oid]; p = self.data.qpos[q:q + 3]; quat = self.data.qpos[q + 3:q + 7]
        upright = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)
        speed = float(np.linalg.norm(self.data.qvel[self.obj_v[oid]:self.obj_v[oid] + 3]))
        return dict(x=float(p[0]), y=float(p[1]), z=float(p[2]), upright=float(upright), speed=speed)

    def set_object_pose(self, oid, x, y, z):
        q, v = self.obj_q[oid], self.obj_v[oid]
        self.data.qpos[q:q + 3] = [x, y, z]; self.data.qpos[q + 3:q + 7] = [1, 0, 0, 0]
        self.data.qvel[v:v + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def hold(self, k, oid):
        hp = self.hand_pos(k); off = OBJECTS[oid]["hold_off"]
        self.set_object_pose(oid, hp[0], hp[1], hp[2] + off)
        self.data.eq_active[self.weld[(k, oid)]] = 1
        mujoco.mj_forward(self.model, self.data)

    def release(self, k, oid, surface_z):
        hp = self.hand_pos(k)
        self.data.eq_active[self.weld[(k, oid)]] = 0
        self.set_object_pose(oid, hp[0], hp[1], surface_z + OBJECTS[oid]["half_h"] + 0.01)

    def set_object_collisions(self, oid, on):
        g = self.obj_geom[oid]
        self.model.geom_contype[g] = 1 if on else 0
        self.model.geom_conaffinity[g] = 1 if on else 0

    def set_cup_filled(self, filled=True):
        self.model.geom_rgba[self.obj_geom["cup"]] = COFFEE_RGBA if filled else [0.92, 0.92, 0.92, 1.0]

    def show_stain(self, x, y):
        self.model.geom_pos[self.stain] = [x, y, 0.002]

    def reset_objects(self):
        for e in self.weld.values():
            self.data.eq_active[e] = 0
        for oid, (x, y, z) in self.spawns.items():
            self.set_object_collisions(oid, True)
            self.set_object_pose(oid, x, y, z)
        if "cup" in self.spawns:
            self.set_cup_filled(False)
        self.model.geom_pos[self.stain] = [0, 0, -1]
        for k in self.arm_a:
            self.set_arm(k, 0.0, 0.0)

    # stepping --------------------------------------------------
    def step(self):
        mujoco.mj_step(self.model, self.data)

    def robot_contacts(self):
        out = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            n1, n2 = self.geom_name[c.geom1], self.geom_name[c.geom2]
            for a, b in ((n1, n2), (n2, n1)):
                if a.startswith("robot_") and b != "floor":
                    try:
                        ka = int(a.split("_")[1])
                    except ValueError:
                        continue
                    if b.startswith("robot_"):
                        try:
                            if int(b.split("_")[1]) == ka: continue
                        except ValueError:
                            continue
                    out.add((ka, b))
        return out

    def render_eye(self, k, path):
        try:
            if not hasattr(self, "_renderer") or self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, 240, 320)
            self._renderer.update_scene(self.data, camera=f"robot_{k}_eye")
            img = self._renderer.render()
            from PIL import Image
            Image.fromarray(img).save(path, quality=80)
            return path
        except Exception:
            self._renderer = None
            return None
