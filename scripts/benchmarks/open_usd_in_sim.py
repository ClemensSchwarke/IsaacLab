# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open a USD robot asset in the Isaac Sim GUI for interactive inspection.

Intended for eyeballing physics authoring (joint limits, drives, colliders) of a
raw asset. Selects the ``physx`` variant if the asset exposes a ``Physics``
variant set, and adds a dome light so the viewport is visible.

Example::

    /path/to/envs/isaaclab_newton/bin/python scripts/benchmarks/open_usd_in_sim.py \
        --usd usd_assets_menagerie/Isaac/Samples/Mujoco_Menagerie/anybotics_anymal_c/anymal_c/anymal_c.usda
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd", required=True, help="path to the USD asset to open")
parser.add_argument("--variant", default="physx", help="Physics variant selection to apply (if present)")
parser.add_argument(
    "--gravity",
    type=float,
    default=0.0,
    help="gravity magnitude [m/s^2] applied to the physics scene (0 = float in place; use 9.81 for normal)",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# This repo gates the GUI behind a visualizer: only an explicit Kit visualizer opens an editor
# window (headless=False alone hits the "no Kit visualizer requested -> forcing headless" path).
# Default to the Kit visualizer so the asset opens in the Omniverse editor for inspection.
if getattr(args, "visualizer", None) is None:
    args.visualizer = ["kit"]
    args.visualizer_explicit = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os

import omni.usd
from pxr import Gf, Sdf, UsdLux, UsdPhysics

# open the asset as the root stage. Use an ABSOLUTE path: Kit anchors the asset's
# relative references (``@./payloads/...@``) to the root layer's resolved path, and a
# relative root path leaves them unresolved (red prim, "missing references found").
usd_abspath = os.path.abspath(args.usd)
print(f"[open_usd_in_sim] opening {usd_abspath}")
ctx = omni.usd.get_context()
ctx.open_stage(usd_abspath)
stage = ctx.get_stage()

# select the requested Physics variant if the asset exposes one
dp = stage.GetDefaultPrim()
if dp:
    vset = dp.GetVariantSets()
    if "Physics" in vset.GetNames():
        vset.GetVariantSet("Physics").SetVariantSelection(args.variant)
        print(f"[open_usd_in_sim] selected Physics variant = {args.variant}")

# set gravity on the physics scene so Play doesn't drop the robot (create one if absent)
scenes = [UsdPhysics.Scene(p) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
if not scenes:
    scenes = [UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))]
    print("[open_usd_in_sim] created /physicsScene")
for scene in scenes:
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(float(args.gravity))
print(f"[open_usd_in_sim] gravity magnitude set to {args.gravity} m/s^2")

# add a dome light for visibility if the stage has none
if not any(p.IsA(UsdLux.DomeLight) or p.IsA(UsdLux.DistantLight) for p in stage.Traverse()):
    light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/InspectLight"))
    light.CreateIntensityAttr(1000.0)
    print("[open_usd_in_sim] added dome light /World/InspectLight")

print(f"[open_usd_in_sim] opened {args.usd} — inspect joint limits in the Property panel; Ctrl+C to close.")

# keep the GUI alive
while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
