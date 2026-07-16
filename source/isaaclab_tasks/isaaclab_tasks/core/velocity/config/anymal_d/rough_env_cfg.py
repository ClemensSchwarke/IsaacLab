# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # isort: skip


@configclass
class AnymalDRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # sim
        # lower margin to avoid self-collision
        self.sim.physics.newton_mjwarp.default_shape_cfg.margin = 0.001
        # scene
        self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class AnymalDRoughEnvCfg_PLAY(AnymalDRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # scene
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        # observations
        self.observations.policy.enable_corruption = False
        # events
        self.events.base_external_force_torque = None
        self.events.push_robot = None
