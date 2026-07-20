# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import KaminoSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_ovphysx.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from .rough_env_cfg import UnitreeGo1RoughEnvCfg


@configclass
class PhysicsCfg(PresetCfg):
    physx = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    ovphysx = OvPhysxCfg()
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=60,
            nconmax=25,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
        ),
        num_substeps=1,
        debug_mode=False,
    )
    newton_kamino = NewtonCfg(solver_cfg=KaminoSolverCfg(max_contacts_per_world=64), num_substeps=2)
    default = physx


@configclass
class UnitreeGo1FlatEnvCfg(UnitreeGo1RoughEnvCfg):
    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())

    def __post_init__(self):
        super().__post_init__()

        # scene
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        # observations
        self.observations.policy.height_scan = None
        # rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25
        # curriculum
        self.curriculum.terrain_levels = None


@configclass
class UnitreeGo1FlatEnvCfg_PLAY(UnitreeGo1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # scene
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # observations
        self.observations.policy.enable_corruption = False
        # events
        self.events.base_external_force_torque = None
        self.events.push_robot = None
