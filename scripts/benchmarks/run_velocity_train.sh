#!/usr/bin/env bash
# Scratch: run all velocity flat/rough tasks through the training benchmark, seed 0.
# Continues past any failing/timed-out task (logs it) so it can run unattended overnight.
# Run with the isaaclab_newton (Python 3.12) conda env active:
#   bash scripts/benchmarks/run_velocity_train.sh
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
# Set the physics backend here; PRESETS, RUN_NAME and OUTPUT_PATH are all derived from it.
BACKEND=kamino   # one of: physx | mjwarp | kamino | ovphysx
case "$BACKEND" in
  physx)   PRESET_NAME=physx         ;;
  mjwarp)  PRESET_NAME=newton_mjwarp ;;
  kamino)  PRESET_NAME=newton_kamino ;;
  ovphysx) PRESET_NAME=ovphysx       ;;  # requires the optional 'ovphysx' runtime wheel
  *) echo "[error] unknown BACKEND '$BACKEND' (use physx|mjwarp|kamino|ovphysx)" >&2; exit 1 ;;
esac
PRESETS="presets=$PRESET_NAME"                  # backend preset token passed to the launcher
RUN_NAME="$PRESET_NAME"                          # tags the log dir (<timestamp>_<RUN_NAME>)
OUTPUT_PATH="logs/benchmarks/velocity_$BACKEND"  # where bundles/JSON are written

# --- other parameters (edit here) ---
SEED=0
TIMEOUT=0        # per-task timeout [s]; 0 = off. Overnight: set ~1.5-2x the slowest run
SLEEP_BETWEEN=5  # seconds to wait between launches (let Isaac Sim fully shut down)

export PYTHONPATH=$PWD                        # so `scripts` package is importable

# NOTE: the real Kamino solver (KaminoSolverCfg) is only defined in the FLAT configs. On rough tasks
# the base RoughPhysicsCfg aliases newton_kamino -> newton_mjwarp, so rough+kamino would just re-run
# MJWarp. So the Kamino sweep is the flat tasks only.
TASKS=(
  # Isaac-Velocity-Flat-AnymalD
  # Isaac-Velocity-Flat-Cassie
  # Isaac-Velocity-Flat-Digit
  # Isaac-Velocity-Flat-G1
  # Isaac-Velocity-Flat-H1
  # Isaac-Velocity-Flat-UnitreeGo2
  # Isaac-Velocity-Rough-AnymalD
  # Isaac-Velocity-Rough-Cassie
  # Isaac-Velocity-Rough-Digit
  # Isaac-Velocity-Rough-G1
  # Isaac-Velocity-Rough-H1
  # Isaac-Velocity-Rough-UnitreeGo2
)

mkdir -p "$OUTPUT_PATH"
failed=()
for task in "${TASKS[@]}"; do
  echo "=================================================================="
  echo "[run] $task  (seed=$SEED, backend=$BACKEND)"
  echo "=================================================================="
  preset_args=()
  [ -n "$PRESETS" ] && preset_args=("$PRESETS")
  runner=(python)
  # timeout guards against Kit startup hangs; --kill-after SIGKILLs a wedged app that ignores SIGTERM.
  [ "$TIMEOUT" -gt 0 ] && runner=(timeout --kill-after=30 "$TIMEOUT" python)
  # Run via setsid (own process group) in the background so the SIGINT trap fires immediately
  # and can kill the whole Kit process tree. `wait` yields the child's real exit status, so
  # timeout (124) and failures are reported correctly.
  setsid "${runner[@]}" scripts/benchmarks/training.py \
      --rl_library rsl_rl \
      --task "$task" \
      --headless \
      --seed "$SEED" \
      --run_name "$RUN_NAME" \
      --output_path "$OUTPUT_PATH" \
      "${preset_args[@]}" &
  child_pid=$!
  wait "$child_pid"
  rc=$?
  child_pid=""
  if [ "$rc" -ne 0 ]; then
    [ "$rc" = 124 ] && echo "[TIMEOUT] $task (>${TIMEOUT}s)" || echo "[FAIL] $task (rc=$rc)"
    failed+=("$task")
  fi
  sleep "$SLEEP_BETWEEN"
done

echo "=================================================================="
if [ ${#failed[@]} -eq 0 ]; then
  echo "All ${#TASKS[@]} runs completed."
else
  echo "${#failed[@]}/${#TASKS[@]} FAILED/TIMED OUT:"
  printf '  %s\n' "${failed[@]}"
fi
