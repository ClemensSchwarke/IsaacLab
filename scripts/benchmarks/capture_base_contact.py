# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scratch: roll out a trained checkpoint and capture state+action windows AROUND a base-slam (a large
base contact-force spike, e.g. the AnymalD-rough Newton belly-slam), saved in the frame format
``replay_nan_frame.py`` consumes so you can scrub/replay the actual crash.

Key differences from a plain termination capture:
  * The ``base_contact`` termination is DISABLED during capture, so the episode does NOT end at the slam
    and we can record the impact and its aftermath (does it faceplant and fall, or bounce and recover?).
  * Capture triggers on the base contact-force crossing ``--force`` [N] (the slam itself), and records a
    window of ``--pre`` frames before + ``--post`` frames after, so the slam sits in the middle of the trace.
  * Each frame also stores the base contact force and projected-gravity tilt, so the crash is readable
    without any quaternion-convention guessing.

The terrain seed is pinned to 0 (matching ``replay_nan_frame.py``) so the captured world position lands
on the identical mesh at replay time.

    PYTHONPATH=$PWD python scripts/benchmarks/capture_base_contact.py \
        --task Isaac-Velocity-Rough-AnymalD --checkpoint latest --checkpoint_run_name newton_mjwarp \
        --num_envs 512 --num_captures 5 --output logs/rsl_rl/anymal_d_rough/bc_frames presets=newton_mjwarp

    python scripts/benchmarks/replay_nan_frame.py --frame <...>/bc_frame_0.pt --scrub --viz newton presets=newton_mjwarp

Dispatched directly (not via play.py). Run from the repo root with PYTHONPATH=$PWD in the
isaaclab_newton conda env.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent
_RL_SCRIPTS = _BENCH_DIR.parent / "reinforcement_learning"
if str(_RL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_RL_SCRIPTS))

import common as _common  # noqa: E402


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse CLI args and forward the remaining Hydra preset tokens via ``sys.argv``."""
    from isaaclab.app import add_launcher_args

    from isaaclab_tasks.utils import setup_preset_cli

    parser = argparse.ArgumentParser(description="Capture base-slam windows (with aftermath) for replay.")
    parser.add_argument("--task", type=str, required=True, help="Gym task id to roll out (use the training task).")
    parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent cfg entry point.")
    parser.add_argument("--checkpoint", type=str, default="latest", help="'latest'/'best' selector or a path.")
    parser.add_argument(
        "--checkpoint_run_name",
        type=str,
        default=None,
        help="Only consider run dirs whose name contains this (e.g. 'newton_mjwarp').",
    )
    parser.add_argument("--num_envs", type=int, default=512, help="Parallel envs (more = capture faster).")
    parser.add_argument("--num_frames", type=int, default=3000, help="Max rollout steps before giving up.")
    parser.add_argument("--num_captures", type=int, default=5, help="Stop after saving this many slam windows.")
    parser.add_argument("--pre", type=int, default=32, help="Frames recorded before the slam.")
    parser.add_argument("--post", type=int, default=24, help="Frames recorded after the slam.")
    parser.add_argument("--force", type=float, default=200.0, help="Base contact force [N] that counts as a slam.")
    parser.add_argument("--seed", type=int, default=0, help="Env/agent seed.")
    parser.add_argument(
        "--no_self_collisions",
        action="store_true",
        help="Disable articulation self-collisions on the robot, to test whether the base slams are"
        " self-contacts (a leg striking the base) rather than terrain contacts.",
    )
    parser.add_argument("--output", type=str, required=True, help="Directory to write bc_frame_*.pt files.")
    add_launcher_args(parser)

    args, remaining = setup_preset_cli(parser, argv)
    sys.argv = [sys.argv[0]] + remaining
    return args, remaining


def run(argv: list[str]) -> None:
    import collections
    import importlib.metadata as metadata
    import os

    import gymnasium as gym
    import torch
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    from isaaclab.app import launch_simulation

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    args, _ = _parse_args(argv)
    env_cfg, agent_cfg = resolve_task_config(args.task, args.agent)
    window = args.pre + args.post

    with launch_simulation(env_cfg, args):
        env_cfg.scene.num_envs = args.num_envs
        agent_cfg.seed = args.seed
        env_cfg.seed = args.seed
        _tg = getattr(getattr(env_cfg.scene, "terrain", None), "terrain_generator", None)
        if _tg is not None:
            _tg.seed = 0  # same constant replay_nan_frame.py uses -> identical mesh at replay

        if args.no_self_collisions:
            env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
            print("[capture] self-collisions DISABLED on the robot")

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        if args.checkpoint in _common.CHECKPOINT_SELECTORS:
            resume_path = _common.resolve_checkpoint_selector(
                log_root,
                args.checkpoint,
                library="rsl_rl",
                task=args.task,
                checkpoint_pattern=r"model_.*\.pt",
                metadata={"agent": args.agent},
                run_name_contains=args.checkpoint_run_name,
            )
        else:
            resume_path = _common.resolve_play_checkpoint(args.checkpoint, "rsl_rl", args.task)
        print(f"[capture] checkpoint: {resume_path}")

        env = gym.make(args.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        u = env.unwrapped
        device = u.device
        robot = u.scene["robot"]
        term_mgr = u.termination_manager
        if "base_contact" not in term_mgr.active_terms:
            raise RuntimeError(f"task {args.task} has no 'base_contact' termination; active: {term_mgr.active_terms}")

        # Grab the base sensor cfg (resolved body ids) from the base_contact term, then DISABLE that term so
        # the episode continues past the slam and we can record the aftermath.
        _bc_cfg = term_mgr.get_term_cfg("base_contact")
        base_sensor_cfg = _bc_cfg.params["sensor_cfg"]
        contact_sensor = u.scene.sensors[base_sensor_cfg.name]
        _bc_cfg.func = lambda env, **kw: torch.zeros(u.num_envs, dtype=torch.bool, device=device)
        term_mgr.set_term_cfg("base_contact", _bc_cfg)

        def base_force() -> torch.Tensor:
            f = contact_sensor.data.net_forces_w.torch  # [num_envs, num_bodies, 3]
            return f[:, base_sensor_cfg.body_ids, :].norm(dim=-1).amax(dim=1)  # [num_envs]

        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=device)

        outdir = Path(args.output)
        outdir.mkdir(parents=True, exist_ok=True)
        clip = getattr(agent_cfg, "clip_actions", None)
        joint_names = list(robot.joint_names)

        buf: collections.deque = collections.deque(maxlen=window)
        alive = torch.zeros(u.num_envs, device=device)
        pending: dict[int, int] = {}  # env_idx -> step at which to extract its window
        captured = 0
        slam_events = 0  # rising-edge count of base force spikes across all envs (rate proxy)
        prev_spike = torch.zeros(u.num_envs, dtype=torch.bool, device=device)

        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        for step in range(args.num_frames):
            with torch.inference_mode():
                raw = policy(obs)
            act_applied = raw.clamp(-clip, clip) if clip is not None else raw
            buf.append(
                {
                    "root": robot.data.root_state_w.torch.clone(),
                    "jp": robot.data.joint_pos.torch.clone(),
                    "jv": robot.data.joint_vel.torch.clone(),
                    "act": act_applied.clone(),
                    "grav": robot.data.projected_gravity_b.torch.clone(),  # convention-robust tilt
                    "force": base_force().clone(),
                }
            )

            result = env.step(raw)
            obs = result[0]
            if len(result) == 5:
                dones = (torch.as_tensor(result[2], device=device) | torch.as_tensor(result[3], device=device)).bool()
            else:
                dones = torch.as_tensor(result[2], device=device).bool()

            # trigger: base force spike this step, env has a full pre-window buffered, not already pending
            spike = buf[-1]["force"] > args.force
            slam_events += int((spike & ~prev_spike).sum())  # count rising edges as distinct slams
            prev_spike = spike
            fresh = spike & (alive >= args.pre)
            for e in torch.nonzero(fresh, as_tuple=False).flatten().tolist():
                pending.setdefault(int(e), step + args.post)

            # extract any windows whose post-roll has completed and that stayed in one episode
            for e in [e for e, s in pending.items() if step >= s]:
                if float(alive[e]) >= window and len(buf) == window:  # whole window is one episode
                    root_trace = torch.stack([b["root"][e] for b in buf]).detach().cpu()
                    joint_trace = torch.stack([b["jp"][e] for b in buf]).detach().cpu()
                    joint_vel_trace = torch.stack([b["jv"][e] for b in buf]).detach().cpu()
                    action_trace = torch.stack([b["act"][e] for b in buf]).detach().cpu()
                    force_trace = torch.stack([b["force"][e] for b in buf]).detach().cpu()
                    grav_trace = torch.stack([b["grav"][e] for b in buf]).detach().cpu()
                    frame = {
                        "task": args.task,
                        "trigger": "base_slam",
                        "num_bad_envs": 1,
                        "offending_env": int(e),
                        "sane_steps_before_trigger": window - 1,  # replay seeds oldest & plays whole window
                        "root_state_w": root_trace[0],
                        "joint_pos": joint_trace[0],
                        "joint_vel": joint_vel_trace[0],
                        "action": action_trace[0],
                        "root_trace": root_trace,
                        "joint_trace": joint_trace,
                        "joint_vel_trace": joint_vel_trace,
                        "action_trace": action_trace,
                        "force_trace": force_trace,
                        "grav_trace": grav_trace,
                        "slam_frame": args.pre,  # index in the window where the force spiked
                        "obs_bad": None,
                        "env_origin": u.scene.env_origins[e].detach().cpu(),
                        "joint_names": joint_names,
                    }
                    out = outdir / f"bc_frame_{captured}.pt"
                    torch.save(frame, out)
                    print(
                        f"[capture] {out.name}: env {e} slam at step {step - args.post} "
                        f"(peak force {force_trace.max():.0f} N; tilt "
                        f"{torch.rad2deg(torch.acos((-grav_trace[:, 2]).clamp(-1, 1))).max():.0f} deg)"
                    )
                    captured += 1
                del pending[e]
                if captured >= args.num_captures:
                    break

            if captured >= args.num_captures:
                print(f"[capture] reached {captured} captures; stopping.")
                break

            alive = alive + 1.0
            alive[dones] = 0.0

        print(f"[capture] total base-slam events (force > {args.force:.0f} N rising edges): {slam_events}")
        if captured == 0:
            print("[capture] no base slams captured — lower --force or raise --num_envs/--num_frames.")
        env.close()


if __name__ == "__main__":
    run(sys.argv[1:])
