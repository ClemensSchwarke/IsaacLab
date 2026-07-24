# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static semantic comparison of two USD robot assets.

This compares the physically meaningful content of the *old* Isaac Lab robot USDs
(currently loaded by the velocity locomotion tasks) against the *new* MuJoCo
Menagerie USDs, without launching a simulation. It relies only on ``usd-core``
(``pxr``), so it can be run with any Python that has USD available, e.g. the
``isaaclab_newton`` conda env::

    /path/to/envs/isaaclab_newton/bin/python scripts/benchmarks/compare_usd_assets.py

Why not a text diff: the two sides come from different converters (Isaac Lab
pipeline vs. "MuJoCo USD Converter"), one is binary USDC and one is ASCII USDA,
and prim paths / hierarchy differ wildly. We therefore extract a canonical,
order-independent representation keyed by *leaf* prim names and diff that.

Notes on conventions:
    * UsdPhysics revolute joint limits are authored in **degrees**; prismatic in
      stage linear units. We keep the authored values and label them.
    * Inertia is stored as a diagonal in a principal-axes frame that may differ
      between assets. We compare the rotation-invariant **trace** of the inertia
      tensor (sum of the diagonal) rather than raw components.
    * The Menagerie assets expose a ``Physics`` variant set; we select ``physx``
      so we compare the PhysX-authored physics (matching what Isaac Lab loads).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, field

from pxr import Gf, Usd, UsdGeom, UsdPhysics

# Traverse into instance proxies so colliders inside instanceable ``Props`` layers are visible.
_ALL = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
_AXIS_VEC = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}

# ---------------------------------------------------------------------------
# Asset pairs: (robot, old relative path, new relative path).
# Old paths are the exact USDs loaded by the velocity tasks (g1/h1 use *_minimal).
# ---------------------------------------------------------------------------
DEFAULT_OLD_ROOT = "usd_assets_download"
DEFAULT_NEW_ROOT = "usd_assets_menagerie/Isaac/Samples/Mujoco_Menagerie"

PAIRS = [
    ("anymal_b", "Isaac/IsaacLab/Robots/ANYbotics/ANYmal-B/anymal_b.usd", "anybotics_anymal_b/anymal_b/anymal_b.usda"),
    ("anymal_c", "Isaac/IsaacLab/Robots/ANYbotics/ANYmal-C/anymal_c.usd", "anybotics_anymal_c/anymal_c/anymal_c.usda"),
    ("g1", "Isaac/IsaacLab/Robots/Unitree/G1/g1_minimal.usd", "unitree_g1/g1/g1.usda"),
    ("h1", "Isaac/IsaacLab/Robots/Unitree/H1/h1_minimal.usd", "unitree_h1/h1/h1.usda"),
    ("go2", "Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd", "unitree_go2/go2/go2.usda"),
    ("spot", "Isaac/Robots/BostonDynamics/spot/spot.usd", "boston_dynamics_spot/spot/spot.usda"),
]

# Joint types that carry an actuated degree of freedom.
_ACTUATED = {"PhysicsRevoluteJoint", "PhysicsPrismaticJoint"}


@dataclass
class Body:
    name: str
    path: str
    mass: float | None
    mass_source: str  # "body" | "shapes" | "none"
    com: list[float] | None
    inertia_trace: float | None
    is_articulation_root: bool
    colliders: list[str] = field(default_factory=list)  # geom-type[:approximation]


@dataclass
class Joint:
    name: str
    path: str
    type: str
    axis: str | None  # authored physics:axis letter (frame-dependent)
    axis_eff: list[float] | None  # axis rotated by localRot0 into the body0 frame (convention-invariant)
    parent: str | None  # body0 leaf name
    child: str | None  # body1 leaf name
    lower: float | None  # [deg] for revolute, [stage-linear] for prismatic
    upper: float | None
    drive_stiffness: float | None
    drive_damping: float | None
    drive_max_force: float | None
    drive_target: float | None
    armature: float | None
    max_joint_velocity: float | None
    joint_friction: float | None


def _leaf(path) -> str:
    return str(path).rsplit("/", 1)[-1]


def _get(prim, attr):
    a = prim.GetAttribute(attr)
    return a.Get() if a and a.HasAuthoredValue() else None


def _get_any(prim, attrs):
    """First authored value among several attribute names (handles physx vs mjc schema aliases)."""
    for attr in attrs:
        v = _get(prim, attr)
        if v is not None:
            return v
    return None


def _effective_axis(prim) -> list[float] | None:
    """Return the joint axis expressed in the body0 frame (axis letter rotated by ``localRot0``).

    OLD assets often keep ``physics:axis = X`` and bake the true direction into ``localRot0``,
    while NEW assets author the axis letter directly. Composing the two yields a
    convention-invariant direction, canonicalized to a positive dominant component.
    """
    axis = _get(prim, "physics:axis")
    if axis is None or axis not in _AXIS_VEC:
        return None
    v = Gf.Vec3d(_AXIS_VEC[axis])
    rot = _get(prim, "physics:localRot0")
    if rot is not None:
        q = Gf.Quatd(rot.GetReal(), Gf.Vec3d(*rot.GetImaginary()))
        v = q.Transform(v)
    n = v.GetLength()
    if n > 0:
        v = v / n
    comps = [v[0], v[1], v[2]]
    idx = max(range(3), key=lambda i: abs(comps[i]))  # canonical sign: dominant component positive
    if comps[idx] < 0:
        comps = [-c for c in comps]
    return [round(c, 3) for c in comps]


def _inertia_trace(prim) -> float | None:
    diag = _get(prim, "physics:diagonalInertia")
    if diag is None:
        return None
    return float(diag[0] + diag[1] + diag[2])


def _collider_desc(prim) -> str:
    t = prim.GetTypeName()
    if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        approx = _get(prim, "physics:approximation") or "convexHull"
        return f"Mesh:{approx}"
    return str(t)


def extract(path: str, variant: str = "physx") -> dict:
    """Open a stage (selecting the given ``Physics`` variant if present) and extract canonical physics data.

    Args:
        path: USD asset to open.
        variant: ``Physics`` variant to select when the asset exposes the variant set
            (menagerie assets: ``physx`` / ``mujoco`` / ``physics`` / ``none``). Ignored for
            assets without a ``Physics`` variant set (e.g. the old Isaac Lab USDs).
    """
    stage = Usd.Stage.Open(path)
    dp = stage.GetDefaultPrim()
    selected_variant = None
    if dp:
        vs = dp.GetVariantSets()
        if "Physics" in vs.GetNames():
            vs.GetVariantSet("Physics").SetVariantSelection(variant)
            selected_variant = variant

    bodies: dict[str, Body] = {}
    joints: list[Joint] = []
    body_prims = []

    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body_prims.append(prim)

    body_paths = {str(p.GetPath()) for p in body_prims}

    for prim in body_prims:
        name = prim.GetName()
        # mass: prefer body-level MassAPI mass; else aggregate mass authored on descendant shapes
        mass, source = None, "none"
        if prim.HasAPI(UsdPhysics.MassAPI):
            m = _get(prim, "physics:mass")
            if m:
                mass, source = float(m), "body"
        if mass is None:
            shape_mass = 0.0
            found = False
            for d in Usd.PrimRange(prim, _ALL):
                if str(d.GetPath()) == str(prim.GetPath()):
                    continue
                if str(d.GetPath()) in body_paths:  # stop at nested rigid bodies
                    continue
                if d.HasAPI(UsdPhysics.MassAPI):
                    dm = _get(d, "physics:mass")
                    if dm:
                        shape_mass += float(dm)
                        found = True
            if found:
                mass, source = shape_mass, "shapes"
        com = _get(prim, "physics:centerOfMass")
        colliders = []
        for d in Usd.PrimRange(prim, _ALL):
            if str(d.GetPath()) in body_paths and str(d.GetPath()) != str(prim.GetPath()):
                continue
            if d.HasAPI(UsdPhysics.CollisionAPI):
                colliders.append(_collider_desc(d))
        bodies[name] = Body(
            name=name,
            path=str(prim.GetPath()),
            mass=mass,
            mass_source=source,
            com=[round(c, 6) for c in com] if com is not None else None,
            inertia_trace=round(_inertia_trace(prim), 8) if _inertia_trace(prim) is not None else None,
            is_articulation_root=prim.HasAPI(UsdPhysics.ArticulationRootAPI),
            colliders=colliders,
        )

    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        b0 = prim.GetRelationship("physics:body0").GetTargets()
        b1 = prim.GetRelationship("physics:body1").GetTargets()
        joints.append(
            Joint(
                name=prim.GetName(),
                path=str(prim.GetPath()),
                type=str(prim.GetTypeName()),
                axis=_get(prim, "physics:axis"),
                axis_eff=_effective_axis(prim),
                parent=_leaf(b0[0]) if b0 else None,
                child=_leaf(b1[0]) if b1 else None,
                lower=_get(prim, "physics:lowerLimit"),
                upper=_get(prim, "physics:upperLimit"),
                drive_stiffness=_get(prim, "drive:angular:physics:stiffness")
                or _get(prim, "drive:linear:physics:stiffness"),
                drive_damping=_get(prim, "drive:angular:physics:damping") or _get(prim, "drive:linear:physics:damping"),
                drive_max_force=_get(prim, "drive:angular:physics:maxForce")
                or _get(prim, "drive:linear:physics:maxForce"),
                drive_target=_get(prim, "drive:angular:physics:targetPosition")
                or _get(prim, "drive:linear:physics:targetPosition"),
                # physx and mujoco variants author these under different schema names
                armature=_get_any(prim, ["physxJoint:armature", "mjc:armature"]),
                max_joint_velocity=_get(prim, "physxJoint:maxJointVelocity"),
                joint_friction=_get_any(prim, ["physxJoint:jointFriction", "mjc:frictionloss"]),
            )
        )

    total_mass = sum(b.mass for b in bodies.values() if b.mass)
    actuated = [j for j in joints if j.type in _ACTUATED]
    fixed = [j for j in joints if j.type == "PhysicsFixedJoint"]

    return {
        "path": path,
        "variant": selected_variant,
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "kg_per_unit": UsdPhysics.GetStageKilogramsPerUnit(stage),
        "up_axis": UsdGeom.GetStageUpAxis(stage),
        "default_prim": dp.GetName() if dp else None,
        "num_bodies": len(bodies),
        "num_joints": len(joints),
        "num_actuated": len(actuated),
        "num_fixed": len(fixed),
        "total_mass": round(total_mass, 4),
        "bodies": {n: asdict(b) for n, b in bodies.items()},
        "joints": {j.name: asdict(j) for j in joints},
    }


def _fmt(v, ndigits=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{ndigits}f}"
    return str(v)


def compare(robot: str, old: dict, new: dict) -> None:
    print("\n" + "=" * 88)
    print(f"ROBOT: {robot}")
    print("=" * 88)

    # -- stage / conventions ------------------------------------------------
    print("  conventions           OLD                         NEW")
    for k in ("meters_per_unit", "kg_per_unit", "up_axis", "default_prim"):
        print(f"    {k:<18} {str(old[k]):<26}  {new[k]}")

    # -- gross counts -------------------------------------------------------
    print("\n  counts                OLD        NEW        Δ(new-old)")
    for k, label in [
        ("num_bodies", "rigid bodies"),
        ("num_joints", "joints (all)"),
        ("num_actuated", "actuated DOF"),
        ("num_fixed", "fixed joints"),
    ]:
        d = new[k] - old[k]
        print(f"    {label:<18} {old[k]:<10} {new[k]:<10} {d:+d}")
    dm = new["total_mass"] - old["total_mass"]
    print(f"    {'total mass [kg]':<18} {old['total_mass']:<10} {new['total_mass']:<10} {dm:+.4f}")

    # -- joint set diff -----------------------------------------------------
    oj, nj = set(old["joints"]), set(new["joints"])
    only_old, only_new, common = sorted(oj - nj), sorted(nj - oj), sorted(oj & nj)
    print(f"\n  joints only in OLD ({len(only_old)}): {', '.join(only_old) or '—'}")
    print(f"  joints only in NEW ({len(only_new)}): {', '.join(only_new) or '—'}")

    # -- per-joint physics diff (common actuated joints) --------------------
    rows = []
    for name in common:
        jo, jn = old["joints"][name], new["joints"][name]
        if jo["type"] not in _ACTUATED and jn["type"] not in _ACTUATED:
            continue
        flags = []
        ao, an = jo["axis_eff"], jn["axis_eff"]
        if ao is not None and an is not None:
            # compare direction up to sign; flag only a genuine reorientation (> ~5.7°)
            dot = abs(sum(x * y for x, y in zip(ao, an)))
            if dot < 0.995:
                flags.append(f"axis {ao}→{an}")
        for key, lab in [("lower", "lo"), ("upper", "up")]:
            a, b = jo[key], jn[key]
            if a is not None and b is not None and abs(a - b) > 1e-3:
                flags.append(f"{lab} {a:.1f}→{b:.1f}")
            elif (a is None) != (b is None):
                flags.append(f"{lab} {_fmt(a, 1)}→{_fmt(b, 1)}")
        for key, lab in [("armature", "arm"), ("drive_stiffness", "kp"), ("drive_damping", "kd")]:
            a, b = jo[key], jn[key]
            if (a is None) != (b is None) or (a is not None and b is not None and abs(a - b) > 1e-4):
                flags.append(f"{lab} {_fmt(a)}→{_fmt(b)}")
        if flags:
            rows.append((name, "; ".join(flags)))
    print(
        f"\n  actuated-joint differences ("
        f"{len(rows)}/{len([n for n in common if old['joints'][n]['type'] in _ACTUATED])} common):"
    )
    for name, f in rows:
        print(f"    {name:<28} {f}")
    if not rows:
        print("    (none — limits/axes/armature/drives match within tolerance)")

    # -- body mass diff (top movers) ---------------------------------------
    print("\n  per-body mass Δ (matched links, |Δ| ≥ 0.02 kg, top 12):")
    mass_rows = []
    for name in sorted(set(old["bodies"]) & set(new["bodies"])):
        mo, mn = old["bodies"][name]["mass"], new["bodies"][name]["mass"]
        if mo is not None and mn is not None and abs(mn - mo) >= 0.02:
            mass_rows.append((abs(mn - mo), name, mo, mn))
    for _, name, mo, mn in sorted(mass_rows, reverse=True)[:12]:
        print(f"    {name:<28} {mo:7.3f} → {mn:7.3f}   ({mn - mo:+.3f})")
    if not mass_rows:
        print("    (none above threshold)")
    bo, bn = set(old["bodies"]), set(new["bodies"])
    print(f"  bodies only in OLD ({len(bo - bn)}): {', '.join(sorted(bo - bn)) or '—'}")
    print(f"  bodies only in NEW ({len(bn - bo)}): {', '.join(sorted(bn - bo)) or '—'}")

    # -- collider geometry summary -----------------------------------------
    def collider_counter(d):
        c = Counter()
        for b in d["bodies"].values():
            for cd in b["colliders"]:
                c[cd] += 1
        return c

    co, cn = collider_counter(old), collider_counter(new)
    print("\n  collider geometry (type → count):")
    print(f"    OLD: {dict(co)}")
    print(f"    NEW: {dict(cn)}")

    # -- drive presence -----------------------------------------------------
    od = sum(1 for j in old["joints"].values() if j["drive_stiffness"] is not None)
    n_new_drives = sum(1 for j in new["joints"].values() if j["drive_stiffness"] is not None)
    print(f"\n  joints with authored USD drive: OLD={od}  NEW={n_new_drives}")


# ---------------------------------------------------------------------------
# Difference-presence matrix (robots × difference categories)
# ---------------------------------------------------------------------------
# Each category maps to a predicate over the (old, new) canonical dicts. Marks are
# derived, never hand-authored, so the matrix regenerates from the extracted data.
_MATRIX_CATEGORIES = [
    "Actuated-DOF count changed",
    "Kinematic config changed (joints added in NEW)",
    "Fixed-frame links dropped (feet/head/imu)",
    "Foot contact bodies removed/merged",
    "Total mass changed (>0.1 kg)",
    "Main-body mass shift (>0.1 kg)",
    "Real joint limits changed",
    "USD drives dropped (present→none)",
    "USD drives re-tuned (present→different)",
    "Armature changed",
    "Genuine joint-axis reorientation",
    "Colliders fully re-authored",
]


def _unbounded(v) -> bool:
    """A revolute limit is non-binding if unauthored or at the ``±540°``-style sentinel."""
    return v is None or abs(v) >= 359.0


def _root_mass(d: dict) -> float | None:
    for b in d["bodies"].values():
        if b["is_articulation_root"]:
            return b["mass"]
    return None


def _collider_counts(d: dict) -> Counter:
    c: Counter = Counter()
    for b in d["bodies"].values():
        for x in b["colliders"]:
            c[x] += 1
    return c


def difference_flags(old: dict, new: dict) -> dict[str, bool]:
    """Return {category: applies?} for one robot's old/new pair."""
    oj, nj = old["joints"], new["joints"]
    common = set(oj) & set(nj)
    act_common = [j for j in common if oj[j]["type"] in _ACTUATED]
    new_only_act = [j for j in set(nj) - set(oj) if nj[j]["type"] in _ACTUATED]

    limits_changed = False
    for j in act_common:
        for k in ("lower", "upper"):
            a, b = oj[j][k], nj[j][k]
            ua, ub = _unbounded(a), _unbounded(b)
            if ua != ub or (not ua and not ub and abs(a - b) > 2.0):
                limits_changed = True

    od = sum(1 for j in oj.values() if j["drive_stiffness"] is not None)
    n_new_drives = sum(1 for j in nj.values() if j["drive_stiffness"] is not None)
    drives_retuned = (
        od > 0
        and n_new_drives > 0
        and any(
            oj[j]["drive_stiffness"] is not None
            and nj[j]["drive_stiffness"] is not None
            and abs(oj[j]["drive_stiffness"] - nj[j]["drive_stiffness"]) > 1e-4
            for j in act_common
        )
    )
    armature_changed = any(
        (oj[j]["armature"] is None) != (nj[j]["armature"] is None)
        or (
            oj[j]["armature"] is not None
            and nj[j]["armature"] is not None
            and abs(oj[j]["armature"] - nj[j]["armature"]) > 1e-4
        )
        for j in act_common
    )
    axis_reoriented = any(
        oj[j]["axis_eff"]
        and nj[j]["axis_eff"]
        and abs(sum(x * y for x, y in zip(oj[j]["axis_eff"], nj[j]["axis_eff"]))) < 0.995
        for j in act_common
    )
    root_o, root_n = _root_mass(old), _root_mass(new)

    return {
        _MATRIX_CATEGORIES[0]: old["num_actuated"] != new["num_actuated"],
        _MATRIX_CATEGORIES[1]: len(new_only_act) > 0,
        _MATRIX_CATEGORIES[2]: old["num_fixed"] > new["num_fixed"],
        _MATRIX_CATEGORIES[3]: any("foot" in b.lower() for b in set(old["bodies"]) - set(new["bodies"])),
        _MATRIX_CATEGORIES[4]: abs(old["total_mass"] - new["total_mass"]) > 0.1,
        _MATRIX_CATEGORIES[5]: root_o is not None and root_n is not None and abs(root_o - root_n) > 0.1,
        _MATRIX_CATEGORIES[6]: limits_changed,
        _MATRIX_CATEGORIES[7]: od > 0 and n_new_drives == 0,
        _MATRIX_CATEGORIES[8]: drives_retuned,
        _MATRIX_CATEGORIES[9]: armature_changed,
        _MATRIX_CATEGORIES[10]: axis_reoriented,
        _MATRIX_CATEGORIES[11]: _collider_counts(old) != _collider_counts(new),
    }


def write_matrix(summary: list[tuple[str, dict, dict]], out_dir: str) -> None:
    """Compute the difference matrix and write Markdown + CSV, and print it."""
    robots = [r for r, _, _ in summary]
    flags = {r: difference_flags(o, n) for r, o, n in summary}

    # markdown
    md = ["| Difference | " + " | ".join(robots) + " |"]
    md.append("|" + "---|" * (len(robots) + 1))
    for cat in _MATRIX_CATEGORIES:
        cells = ["✔" if flags[r][cat] else "" for r in robots]
        md.append(f"| {cat} | " + " | ".join(cells) + " |")
    totals = [str(sum(flags[r][c] for c in _MATRIX_CATEGORIES)) for r in robots]
    md.append("| **# differences** | " + " | ".join(f"**{t}**" for t in totals) + " |")
    md_text = "\n".join(md) + "\n"
    with open(os.path.join(out_dir, "difference_matrix.md"), "w") as f:
        f.write("# USD asset difference matrix (old → new)\n\n" + md_text)

    # csv
    csv = ["difference," + ",".join(robots)]
    for cat in _MATRIX_CATEGORIES:
        csv.append(f'"{cat}",' + ",".join("1" if flags[r][cat] else "0" for r in robots))
    csv.append("# differences," + ",".join(totals))
    with open(os.path.join(out_dir, "difference_matrix.csv"), "w") as f:
        f.write("\n".join(csv) + "\n")

    # console (aligned)
    print("\n" + "=" * 88)
    print("DIFFERENCE MATRIX (✔ = applies)")
    print("=" * 88)
    w = max(len(c) for c in _MATRIX_CATEGORIES)
    print(f"{'difference':<{w}} | " + " | ".join(f"{r:^9}" for r in robots))
    print("-" * (w + 3 + 12 * len(robots)))
    for cat in _MATRIX_CATEGORIES:
        cells = [" ✔ " if flags[r][cat] else "" for r in robots]
        print(f"{cat:<{w}} | " + " | ".join(f"{c:^9}" for c in cells))
    print(f"{'# differences':<{w}} | " + " | ".join(f"{t:^9}" for t in totals))
    print(f"\n  matrix written to: {out_dir}/difference_matrix.{{md,csv}}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old-root", default=DEFAULT_OLD_ROOT)
    ap.add_argument("--new-root", default=DEFAULT_NEW_ROOT)
    ap.add_argument("--out-dir", default="usd_compare_out", help="where to dump per-asset canonical JSON")
    ap.add_argument("--robots", nargs="*", default=None, help="subset of robots to compare")
    ap.add_argument(
        "--matrix", action="store_true", help="print/write only the difference matrix (skip per-robot report)"
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pairs = [p for p in PAIRS if args.robots is None or p[0] in args.robots]

    summary = []
    for robot, old_rel, new_rel in pairs:
        old_path = os.path.join(args.old_root, old_rel)
        new_path = os.path.join(args.new_root, new_rel)
        old = extract(old_path)
        new = extract(new_path)
        with open(os.path.join(args.out_dir, f"{robot}_old.json"), "w") as f:
            json.dump(old, f, indent=2)
        with open(os.path.join(args.out_dir, f"{robot}_new.json"), "w") as f:
            json.dump(new, f, indent=2)
        if not args.matrix:
            compare(robot, old, new)
        summary.append((robot, old, new))

    if args.matrix:
        write_matrix(summary, args.out_dir)
        return

    # -- final one-line-per-robot summary ----------------------------------
    print("\n" + "=" * 88)
    print("SUMMARY (old → new)")
    print("=" * 88)
    print(f"  {'robot':<10} {'bodies':>12} {'act.DOF':>10} {'fixed':>10} {'mass[kg]':>16}")
    for robot, old, new in summary:
        print(
            f"  {robot:<10} {old['num_bodies']:>5}→{new['num_bodies']:<5} "
            f"{old['num_actuated']:>4}→{new['num_actuated']:<4} "
            f"{old['num_fixed']:>4}→{new['num_fixed']:<4} "
            f"{old['total_mass']:>7.3f}→{new['total_mass']:<7.3f}"
        )
    print(f"\n  per-asset canonical JSON written to: {args.out_dir}/")

    write_matrix(summary, args.out_dir)


if __name__ == "__main__":
    main()
