#!/usr/bin/env bash
# Scratch: evaluate (play) each velocity task's trained checkpoint on PhysX and collect PlayBundles.
# Pairs with run_velocity_train.sh (which produces the checkpoints).
# Run with the isaaclab_newton (Python 3.12) conda env active:
#   bash scripts/benchmarks/run_velocity_play.sh
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
# Set the physics backend here; PRESETS, CKPT_RUN_NAME and OUTPUT_PATH are all derived from it.
# Must match the backend the checkpoints were trained with (see run_velocity_train.sh).
BACKEND=kamino   # one of: physx | mjwarp | kamino | ovphysx
case "$BACKEND" in
  physx)   PRESET_NAME=physx         ;;
  mjwarp)  PRESET_NAME=newton_mjwarp ;;
  kamino)  PRESET_NAME=newton_kamino ;;
  ovphysx) PRESET_NAME=ovphysx       ;;  # requires the optional 'ovphysx' runtime wheel
  *) echo "[error] unknown BACKEND '$BACKEND' (use physx|mjwarp|kamino|ovphysx)" >&2; exit 1 ;;
esac
PRESETS="presets=$PRESET_NAME"                   # backend preset token passed to the launcher
CKPT_RUN_NAME="$PRESET_NAME"                      # only pick run dirs whose --run_name tag matches this
                                                  # backend, so we do not grab another backend's policy
OUTPUT_PATH="logs/benchmarks/velocity_$BACKEND"   # where PlayBundle JSONs are written

# --- other parameters (edit here) ---
SEEDS=(0 1 2)    # evaluate each checkpoint under these eval seeds; the summary script averages them
SELECTOR=latest  # 'latest' or 'best' (checkpoint-manifest selector)
NUM_ENVS=4096    # override the Play cfg's small (~50) count for a stable success rate
NUM_FRAMES=1000  # inference steps per task (~1 episode/env at 1024 envs)
TERRAIN_ROWS=10  # override Play cfg's num_rows (10x20 matches training density at 4096 envs)
TERRAIN_COLS=20  # override Play cfg's num_cols; set empty to use the Play cfg's default (5x5)
TIMEOUT=3000      # kill a task after this many seconds (guards against startup hangs)
SLEEP_BETWEEN=5  # seconds to wait between launches (let Isaac Sim fully shut down)

export PYTHONPATH=$PWD                     # so `scripts` package is importable

# Evaluate the -Play variants (reduced envs, corruption off). Checkpoint discovery normalizes
# '-Play' away, so `--checkpoint latest` finds the matching training run's checkpoint.
TASKS=(
  # Isaac-Velocity-Flat-AnymalD-Play
  # Isaac-Velocity-Flat-Cassie-Play
  # Isaac-Velocity-Flat-Digit-Play
  # Isaac-Velocity-Flat-G1-Play
  # Isaac-Velocity-Flat-H1-Play
  # Isaac-Velocity-Flat-UnitreeGo2-Play
  # Isaac-Velocity-Rough-AnymalD-Play
  # Isaac-Velocity-Rough-Cassie-Play
  # Isaac-Velocity-Rough-Digit-Play
  # Isaac-Velocity-Rough-G1-Play
  # Isaac-Velocity-Rough-H1-Play
  # Isaac-Velocity-Rough-UnitreeGo2-Play
)

mkdir -p "$OUTPUT_PATH"
preset_args=()
[ -n "$PRESETS" ] && preset_args=("$PRESETS")
ckpt_filter=()
[ -n "$CKPT_RUN_NAME" ] && ckpt_filter=(--checkpoint_run_name "$CKPT_RUN_NAME")
terrain_override=()
[ -n "$TERRAIN_ROWS" ] && terrain_override+=(--terrain_rows "$TERRAIN_ROWS")
[ -n "$TERRAIN_COLS" ] && terrain_override+=(--terrain_cols "$TERRAIN_COLS")

failed=()
total=0
for task in "${TASKS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    total=$((total + 1))
    echo "=================================================================="
    echo "[eval] $task  (seed=$seed, checkpoint=$SELECTOR, envs=$NUM_ENVS, frames=$NUM_FRAMES, backend=$BACKEND)"
    echo "=================================================================="
    # timeout guards against Kit startup deadlocks; --kill-after SIGKILLs a wedged app that ignores SIGTERM.
    # Run via setsid (own process group) in the background so the SIGINT trap fires immediately
    # and can kill the whole Kit process tree. `wait` yields the child's real exit status, so
    # timeout (124) and failures are reported correctly.
    setsid timeout --kill-after=30 "$TIMEOUT" python scripts/benchmarks/play.py \
        --rl_library rsl_rl \
        --task "$task" \
        --headless \
        --seed "$seed" \
        --checkpoint "$SELECTOR" \
        "${ckpt_filter[@]}" \
        --num_envs "$NUM_ENVS" \
        --num_frames "$NUM_FRAMES" \
        "${terrain_override[@]}" \
        --output_path "$OUTPUT_PATH" \
        "${preset_args[@]}" &
    child_pid=$!
    wait "$child_pid"
    rc=$?
    child_pid=""
    if [ "$rc" -ne 0 ]; then
      [ "$rc" = 124 ] && echo "[TIMEOUT] $task seed=$seed (>${TIMEOUT}s)" || echo "[FAIL] $task seed=$seed (rc=$rc)"
      failed+=("$task (seed=$seed)")
    fi
    sleep "$SLEEP_BETWEEN"
  done
done

echo "=================================================================="
if [ ${#failed[@]} -eq 0 ]; then
  echo "All $total evaluations completed."
else
  echo "${#failed[@]}/$total FAILED:"
  printf '  %s\n' "${failed[@]}"
fi
