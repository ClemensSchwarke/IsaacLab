#!/usr/bin/env bash
# Scratch: render a short rollout video of the latest checkpoint for each velocity task.
# Drives the standard RSL-RL play script (scripts/reinforcement_learning/rsl_rl/play.py --video),
# which picks the latest matching run via --load_run and records via gymnasium RecordVideo.
# Pairs with run_velocity_train.sh (which produces the checkpoints).
#
# Requires a display: the command/velocity arrow markers only render when a marker-capable
# visualizer is active, and the Kit visualizer (--viz kit) needs the live viewport, so this runs
# windowed rather than headless. Run on a machine with a display (DISPLAY set) or a virtual X.
#
# Run with the isaaclab_newton (Python 3.12) conda env active:
#   bash scripts/benchmarks/run_velocity_video.sh
set -u

cd "$(dirname "$0")/../.." || exit 1  # repo root, so relative paths resolve regardless of CWD

# Abort the whole sweep on Ctrl-C (SIGINT) / SIGTERM instead of just interrupting the current task
# and marching on to the next one. Isaac Sim's Kit runtime installs its own SIGINT handler and
# ignores Ctrl-C, so a plain interrupt (or even SIGTERM) does not stop it. Each task is launched
# via `setsid` in its own process group; on abort we signal the whole group, escalating TERM ->
# SIGKILL so a Kit process that ignores signals is force-killed.
child_pid=""
abort() {
  echo
  echo "[abort] interrupted -- killing current task and stopping sweep"
  if [ -n "$child_pid" ]; then
    kill -TERM "-$child_pid" 2>/dev/null       # negative pid == whole process group
    for _ in $(seq 1 10); do                   # up to ~5s for a graceful shutdown
      kill -0 "-$child_pid" 2>/dev/null || break
      sleep 0.5
    done
    kill -KILL "-$child_pid" 2>/dev/null        # hammer anything still alive
  fi
  exit 130
}
trap abort INT TERM

# --- backend selection (single switch) ---
# Set the physics backend here; PRESETS, RUN_TAG and OUTPUT_PATH are all derived from it.
# Must match the backend the checkpoints were trained with (see run_velocity_train.sh).
BACKEND=mjwarp   # one of: physx | mjwarp | kamino
case "$BACKEND" in
  physx)  PRESET_NAME=physx         ;;
  mjwarp) PRESET_NAME=newton_mjwarp ;;
  kamino) PRESET_NAME=newton_kamino ;;
  *) echo "[error] unknown BACKEND '$BACKEND' (use physx|mjwarp|kamino)" >&2; exit 1 ;;
esac
PRESETS="presets=$PRESET_NAME"                    # backend preset token passed to the launcher
RUN_TAG="$PRESET_NAME"                             # matches the training --run_name; picks the run dir
OUTPUT_PATH="logs/benchmarks/velocity_$BACKEND"    # base dir for this backend's artifacts
VIDEO_DIR="$OUTPUT_PATH/videos"                    # collected mp4s land here, named per task

# --- other parameters (edit here) ---
SEED=0            # single eval seed is enough for a video
VIDEO_LENGTH=300  # recorded steps (~6 s at 0.02 s/step)
TIMEOUT=300       # kill a task after this many seconds (guards against startup hangs)
SLEEP_BETWEEN=5   # seconds to wait between launches (let Isaac Sim fully shut down)

export PYTHONPATH=$PWD                     # so `scripts` package is importable

# Play the -Play variants (reduced envs, corruption off); checkpoint discovery strips '-Play'.
TASKS=(
  Isaac-Velocity-Flat-AnymalD-Play
  Isaac-Velocity-Flat-Cassie-Play
  # Isaac-Velocity-Flat-Digit-Play
  Isaac-Velocity-Flat-G1-Play
  Isaac-Velocity-Flat-H1-Play
  Isaac-Velocity-Flat-UnitreeGo2-Play
  Isaac-Velocity-Rough-AnymalD-Play
  # Isaac-Velocity-Rough-Cassie-Play
  # Isaac-Velocity-Rough-Digit-Play
  # Isaac-Velocity-Rough-G1-Play
  Isaac-Velocity-Rough-H1-Play
  Isaac-Velocity-Rough-UnitreeGo2-Play
)

mkdir -p "$VIDEO_DIR"
preset_args=()
[ -n "$PRESETS" ] && preset_args=("$PRESETS")

failed=()
total=0
for task in "${TASKS[@]}"; do
  total=$((total + 1))
  echo "=================================================================="
  echo "[video] $task  (seed=$SEED, length=$VIDEO_LENGTH, backend=$BACKEND)"
  echo "=================================================================="
  # Marker to find the mp4 this run produces (play.py writes it under the checkpoint's run dir).
  marker="$(mktemp)"
  # timeout guards against Kit startup deadlocks; --kill-after SIGKILLs a wedged app that ignores SIGTERM.
  # Run via setsid (own process group) in the background so the SIGINT trap fires immediately
  # and can kill the whole Kit process tree. `wait` yields the child's real exit status.
  # NOTE: --viz kit is required so the command/velocity arrow markers render in the recording. The
  # marker callbacks only fire when a marker-capable visualizer is active; the Kit visualizer needs
  # the live viewport, so this runs windowed (no --headless) and requires a display (e.g. DISPLAY set).
  # --load_run picks the latest run whose name carries the backend tag.
  setsid timeout --kill-after=30 "$TIMEOUT" python scripts/reinforcement_learning/rsl_rl/play.py \
      --task "$task" \
      --video \
      --video_length "$VIDEO_LENGTH" \
      --seed "$SEED" \
      --load_run ".*$RUN_TAG" \
      --viz kit \
      "${preset_args[@]}" &
  child_pid=$!
  wait "$child_pid"
  rc=$?
  child_pid=""
  if [ "$rc" -ne 0 ]; then
    [ "$rc" = 124 ] && echo "[TIMEOUT] $task (>${TIMEOUT}s)" || echo "[FAIL] $task (rc=$rc)"
    failed+=("$task")
    rm -f "$marker"
    sleep "$SLEEP_BETWEEN"
    continue
  fi
  # Collect the newest mp4 produced after the marker into VIDEO_DIR under a task-named file.
  # Poll briefly: RecordVideo finalizes the mp4 via ffmpeg, which can lag python's exit by a second.
  vid=""
  for _ in $(seq 1 10); do
    vid="$(find logs/rsl_rl -name '*.mp4' -newer "$marker" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
    [ -n "$vid" ] && break
    sleep 1
  done
  rm -f "$marker"
  if [ -n "$vid" ]; then
    dest="$VIDEO_DIR/${task}_seed${SEED}.mp4"
    cp "$vid" "$dest"
    echo "[video] saved $task -> $dest"
  else
    echo "[warn] no video file found for $task (left in place by play.py, if any)"
  fi
  sleep "$SLEEP_BETWEEN"
done

echo "=================================================================="
if [ ${#failed[@]} -eq 0 ]; then
  echo "All $total videos rendered -> $VIDEO_DIR"
else
  echo "${#failed[@]}/$total FAILED:"
  printf '  %s\n' "${failed[@]}"
fi
