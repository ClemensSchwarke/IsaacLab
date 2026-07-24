# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scratch: load a NaN frame dumped by the bench_rsl_rl nan-capture and replay it to inspect the state.

Writes the captured (last-finite, pre-NaN) robot state into env 0. By default it *freezes* that pose
in the viewer (no stepping, so the already-terminal state never resets) until you close the window,
so you can look at exactly what diverged. Pass --play to instead step forward with the captured action
and watch it blow up. The saved state is backend-agnostic, so you can view it on PhysX even when the
NaN itself is Newton-specific:

    # look at the pose (default = freeze). Headless is the default now, so you MUST pass --viz to open
    # a window: --viz kit for the Omniverse viewport on PhysX (renders reliably), or --viz newton for
    # the Newton viewer. The saved state is backend-agnostic, so PhysX is fine for viewing:
    python scripts/benchmarks/replay_nan_frame.py --frame logs/rsl_rl/<exp>/<run>/nan_frame.pt --viz kit presets=physx

    # step forward and try to reproduce the NaN on Newton (headless is fine — no --viz needed):
    python scripts/benchmarks/replay_nan_frame.py --frame <...>/nan_frame.pt --play presets=newton_mjwarp

Dispatched directly (not via play.py). Run from the repo root with PYTHONPATH=$PWD in the
isaaclab_newton conda env.
"""

from __future__ import annotations

import argparse
import sys


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse CLI args and forward the remaining Hydra preset tokens via ``sys.argv``."""
    from isaaclab.app import add_launcher_args

    from isaaclab_tasks.utils import setup_preset_cli

    parser = argparse.ArgumentParser(description="Replay a captured NaN frame for a velocity task.")
    parser.add_argument("--frame", type=str, required=True, help="Path to the nan_frame.pt from nan-capture.")
    parser.add_argument("--task", type=str, default=None, help="Task id (default: read from the frame file).")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of envs to spawn for the replay.")
    parser.add_argument(
        "--play",
        action="store_true",
        help="Step forward from the captured state (watch it diverge) instead of freezing the pose.",
    )
    parser.add_argument(
        "--scrub",
        action="store_true",
        help="Animate through every buffered frame (the runaway ramp), looping, instead of one pose.",
    )
    parser.add_argument("--frame_hold", type=int, default=20, help="Render frames to hold each pose in --scrub mode.")
    parser.add_argument(
        "--lift",
        type=float,
        default=0.0,
        help="Raise the base by this many metres (debug only; leave 0 to see the true ground relationship).",
    )
    parser.add_argument("--steps", type=int, default=300, help="Steps to run in --play mode.")
    parser.add_argument("--action", choices=["saved", "zero"], default="saved", help="Action applied in --play mode.")
    parser.add_argument(
        "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent entry point (only for cfg resolution)."
    )
    add_launcher_args(parser)

    args, remaining = setup_preset_cli(parser, argv)
    sys.argv = [sys.argv[0]] + remaining
    return args, remaining


def run(argv: list[str]) -> None:
    import torch

    from isaaclab.app import launch_simulation

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    args, _ = _parse_args(argv)

    frame = torch.load(args.frame, map_location="cpu", weights_only=False)
    task = args.task or frame["task"]
    print(
        f"[replay] frame task={frame['task']} | trigger={frame.get('trigger', 'nan')} "
        f"| offending_env={frame['offending_env']} | bad_envs={frame.get('num_bad_envs', frame.get('num_nan_envs'))} "
        f"| last-sane {frame.get('sane_steps_before_trigger', 0)} steps before trigger"
    )
    obs_bad = frame.get("obs_bad", frame.get("obs_nan"))
    if obs_bad is not None:
        nan_idx = torch.nonzero(torch.isnan(obs_bad)).flatten().tolist()
        print(f"[replay] NaN obs indices at the trigger step: {nan_idx}")
    # show the base-height / speed ramp into divergence, if captured
    trace = frame.get("root_trace")
    if trace is not None:
        z = trace[:, 2].tolist()
        speed = trace[:, 7:10].norm(dim=1).tolist()
        print("[replay] runaway ramp (oldest→trigger):")
        print("         base z : " + ", ".join(f"{v:.1f}" for v in z))
        print("         |v|    : " + ", ".join(f"{v:.1f}" for v in speed))

    env_cfg, _ = resolve_task_config(task, args.agent)

    with launch_simulation(env_cfg, args):
        import gymnasium as gym

        env_cfg.scene.num_envs = args.num_envs
        # Pin the terrain seed to the SAME constant bench_rsl_rl uses, so the generator rebuilds the
        # identical mesh (its RNG is independent of num_envs when seeded). Then the robot's real captured
        # world position always lands on exactly the terrain it diverged on.
        _tg = getattr(getattr(env_cfg.scene, "terrain", None), "terrain_generator", None)
        if _tg is not None:
            _tg.seed = 0
        env = gym.make(task, cfg=env_cfg).unwrapped
        env.reset()
        robot = env.scene["robot"]
        device = env.device

        # The terrain seed is pinned, so the mesh matches training exactly — always keep the REAL captured
        # world position. That places the robot on the terrain it actually diverged on, which is what
        # reveals terrain penetration.
        root = frame["root_state_w"].clone().to(device)
        root[2] = root[2] + args.lift
        ids = torch.tensor([0], device=device)

        def _write_state() -> None:
            robot.write_root_state_to_sim(root.unsqueeze(0), env_ids=ids)
            robot.write_joint_state_to_sim(
                frame["joint_pos"].to(device).unsqueeze(0), frame["joint_vel"].to(device).unsqueeze(0), env_ids=ids
            )

        _write_state()

        # aim the viewer at the robot — its env-0 origin on rough terrain is nowhere near the world
        # origin the camera defaults to, which is why it looks "missing".
        pos = root[:3].tolist()
        env.sim.set_camera_view(eye=(pos[0] + 3.0, pos[1] + 3.0, pos[2] + 2.0), target=(pos[0], pos[1], pos[2]))
        print(f"[replay] robot at world ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}); camera aimed there.")

        if args.scrub:
            # Scrub mode: animate through every buffered frame (oldest → divergence trigger), looping.
            # The base is pinned to env-0's origin so the pose stays framed while its orientation and
            # joint angles evolve — you watch the body deform into the runaway. The real position/speed
            # ramp is printed per frame (base z, |v|) since pinning hides the launch itself.
            if not env.sim.has_active_visualizers():
                print(
                    "[replay] no visualizer active — re-run with a viewer:\n"
                    "         '--viz newton presets=newton_mjwarp' or '--viz kit presets=physx'."
                )
            elif "root_trace" not in frame:
                print("[replay] this frame has no root_trace to scrub — re-capture with the updated nan-capture.")
            else:
                root_trace = frame["root_trace"].to(device)  # [buf, 13]
                jt = frame.get("joint_trace")
                joint_trace = jt.to(device) if jt is not None else None
                if joint_trace is None:
                    print("[replay] no per-frame joint_trace (old capture) — animating base orientation only;")
                    print("         re-run training to capture full-pose scrub.")
                nfr = root_trace.shape[0]
                # aim the camera at the last sane frame's real world location
                cx = float(root_trace[nfr - 1, 0])
                cy = float(root_trace[nfr - 1, 1])
                cz = float(root_trace[:, 2].median()) + args.lift  # robot base world height (matches the write)
                env.sim.set_camera_view(eye=(cx + 3.0, cy + 3.0, cz + 1.0), target=(cx, cy, cz))
                zt = root_trace[:, 2].tolist()
                vt = root_trace[:, 7:10].norm(dim=1).tolist()
                print(f"[replay] scrubbing {nfr} frames (oldest→trigger), looping — Ctrl-C or close window to exit.")
                try:
                    while env.sim.is_headless_or_exist_active_visualizer():
                        for i in range(nfr):
                            r = root_trace[i].clone()  # real world pose on the matching terrain
                            r[2] = r[2] + args.lift  # keep the real base height (+ optional lift)
                            r[7:13] = 0.0  # zero velocities — show a static pose per frame
                            jp = joint_trace[i] if joint_trace is not None else frame["joint_pos"].to(device)
                            robot.write_root_state_to_sim(r.unsqueeze(0), env_ids=ids)
                            robot.write_joint_state_to_sim(
                                jp.unsqueeze(0), torch.zeros_like(jp).unsqueeze(0), env_ids=ids
                            )
                            print(
                                f"\r[replay] frame {i + 1:2d}/{nfr}  base z={zt[i]:9.1f}  |v|={vt[i]:9.1f}   ",
                                end="",
                                flush=True,
                            )
                            for _ in range(args.frame_hold):
                                env.sim.forward()
                                env.sim.render()
                    print()
                except KeyboardInterrupt:
                    print()
        elif not args.play:
            # Freeze mode: hold the captured pose (no env.step, so the terminal state never resets)
            # and keep rendering until the viewer window is closed. This is what you want to *look* at.
            # Gate on the visualizer framework (works for both the Kit viewport and the kitless Newton
            # viewer) rather than has_gui, which is Kit-only.
            if not env.sim.has_active_visualizers():
                print(
                    "[replay] no visualizer active — headless is the default. Re-run with a viewer to see the pose:\n"
                    "         '--viz kit presets=physx' (Omniverse viewport) or '--viz newton presets=newton_mjwarp'."
                )
            else:
                print("[replay] holding captured pose — close the viewer window (or Ctrl-C) to exit.")
                try:
                    while env.sim.is_headless_or_exist_active_visualizer():
                        _write_state()
                        env.sim.forward()
                        env.sim.render()
                except KeyboardInterrupt:
                    pass
        else:
            # Play mode: replay the RECORDED action sequence from the last-sane frame so the physics
            # reproduces the exact runaway (re-applying a single action wouldn't). We seed the full sane
            # state (root pose+vel and joint pos+vel are already written by _write_state) and then feed
            # action_trace[sane_idx:] one step at a time.
            atrace = frame.get("action_trace")
            if atrace is not None:
                buf = atrace.shape[0]
                sane_idx = buf - 1 - int(frame.get("sane_steps_before_trigger", 0))
                seq = atrace[sane_idx:].to(device)  # [k, action_dim] real actions, sane → trigger
                print(f"[replay] replaying {seq.shape[0]} recorded actions from the last-sane frame...")
            else:
                # old capture without action_trace — fall back to repeating the single saved action
                print("[replay] no action_trace (old capture) — repeating the single saved action;")
                print("         re-run training for a faithful physics replay.")
                seq = frame["action"].to(device).unsqueeze(0).repeat(args.steps, 1)
            reproduced = False
            for i in range(seq.shape[0]):
                a = seq[i].unsqueeze(0).repeat(args.num_envs, 1)
                if args.action == "zero":
                    a = torch.zeros_like(a)
                obs, _, terminated, truncated, _ = env.step(a)
                obs_t = obs["policy"] if hasattr(obs, "keys") else obs  # dict / TensorDict / tensor
                lin = obs_t[0, 0:3].norm() if obs_t.shape[-1] >= 3 else torch.tensor(0.0)
                print(f"[replay] step {i:2d}  |base_lin_vel_obs|={float(lin):8.2f}")
                if not torch.isfinite(obs_t).all() or (obs_t.abs() > 1.0e3).any():
                    print(f"[replay] divergence REPRODUCED at replay step {i}.")
                    reproduced = True
                    break
                if bool((terminated | truncated)[0]):
                    print(f"[replay] env 0 terminated/reset at step {i}.")
                    break
            if not reproduced:
                print(
                    "[replay] sequence finished without reproducing divergence "
                    "(warm-start/contact state isn't fully captured, so exact replay isn't guaranteed)."
                )

        env.close()


if __name__ == "__main__":
    run(sys.argv[1:])
