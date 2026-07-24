#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Scratch: summarize velocity training + eval (play) benchmark bundles into Markdown comparison tables.
# Reads <repo>/logs/benchmarks/velocity_{physx,mjwarp,ovphysx,kamino} and writes velocity_summary.md there.
# No console output, no CLI args. Tables:
#   1. PhysX   - train vs eval
#   2. MJWarp  - train vs eval
#   3. OvPhysX - train vs eval
#   4. Kamino  - train vs eval
#   5. Train   - cross-solver (PhysX vs MJWarp vs OvPhysX vs Kamino)

from __future__ import annotations

import glob
import json
import os

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BENCH = os.path.join(_REPO, "logs", "benchmarks")
BACKEND_DIRS = {
    "physx": os.path.join(_BENCH, "velocity_physx"),
    "mjwarp": os.path.join(_BENCH, "velocity_mjwarp"),
    "ovphysx": os.path.join(_BENCH, "velocity_ovphysx"),
    "kamino": os.path.join(_BENCH, "velocity_kamino"),
}
OUT_PATH = os.path.join(_BENCH, "velocity_summary.md")

# Flag a metric cell that falls short of its threshold, so sub-par values stand out (most pass).
# A trailing plain-text marker is used instead of markdown, which many viewers do not render in cells.
SUCC_THRESHOLD = 97.0  # success rate [%]
EPLEN_THRESHOLD = 970.0  # episode length
FLAG_MARKER = "★"  # appended to a cell that does NOT clear its threshold


def _norm(task: str) -> str:
    """Merge train/eval variants: strip the '-Play' suffix so both map to one row."""
    return task.removesuffix("-Play")


def _disp(task: str) -> str:
    return _norm(task).replace("Isaac-Velocity-", "")


def _load(bundle_dir: str, prefix: str) -> dict[str, list[dict]]:
    """Return {normalized_task: [newest bundle per seed]}, keeping the newest file per (task, seed)."""
    by_key: dict[tuple[str, object], tuple[float, dict]] = {}
    for path in glob.glob(os.path.join(bundle_dir, f"{prefix}_*.json")):
        try:
            d = json.load(open(path))  # noqa
        except Exception:
            continue
        key = (_norm(d["run"]["task"]), _g(d, "run", "seed"))
        mtime = os.path.getmtime(path)
        if key not in by_key or mtime > by_key[key][0]:
            by_key[key] = (mtime, d)
    by_task: dict[str, list[dict]] = {}
    for (task, _seed), (_mtime, d) in by_key.items():
        by_task.setdefault(task, []).append(d)
    return by_task


def _g(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def pct(v):
    return "-" if v is None else f"{v * 100:.1f}"


def f1(v):
    return "-" if v is None else f"{v:.1f}"


def f2(v):
    return "-" if v is None else f"{v:.2f}"


def i0(v):
    return "-" if v is None else f"{v:.0f}"


def kf(v):
    return "-" if v is None else f"{v / 1000:.1f}"


def _metrics(bundle: dict | None, kind: str) -> dict:
    """Pull the shared metric set from a train or eval bundle (differ in reward/ep_length shape)."""
    if bundle is None:
        return {}
    if kind == "train":
        return dict(
            succ=_g(bundle, "success_rate"),
            rew=_g(bundle, "learning", "reward", "final_ema"),
            eplen=_g(bundle, "learning", "ep_length", "final_ema"),
            fps=_g(bundle, "runtime", "total_fps", "mean"),
            gpu=_g(bundle, "resources", "gpu_mem_gb", "peak"),
            iters=_g(bundle, "runtime", "iterations_completed"),
        )
    return dict(
        succ=_g(bundle, "success_rate"),
        rew=_g(bundle, "reward", "mean"),
        eplen=_g(bundle, "ep_length", "mean"),
        fps=_g(bundle, "runtime", "total_fps", "mean"),
        gpu=_g(bundle, "resources", "gpu_mem_gb", "peak"),
        iters=None,
    )


def _avg_metrics(bundles: list[dict] | None, kind: str) -> dict:
    """Average each metric across the available seeds, ignoring missing values (None if all missing)."""
    per = [_metrics(b, kind) for b in (bundles or [])]
    if not per:
        return {}
    out: dict = {}
    for key in set().union(*(m.keys() for m in per)):
        vals = [m[key] for m in per if m.get(key) is not None]
        out[key] = sum(vals) / len(vals) if vals else None
    return out


# metric columns (iters handled separately so it can go last, train-only): (header, key, formatter)
MCOLS = [
    ("succ%", "succ", pct),
    ("rew", "rew", f1),
    ("eplen", "eplen", i0),
    ("fps(k)", "fps", kf),
    ("gpuGB", "gpu", f2),
]

# Maps metric key -> predicate that is True when the raw value clears its threshold.
PASSES = {
    "succ": lambda v: v * 100 > SUCC_THRESHOLD,  # succ is stored as a 0-1 fraction
    "eplen": lambda v: v > EPLEN_THRESHOLD,
}


def _emph(key: str, value, text: str) -> str:
    """Append the flag marker when the metric falls short of its threshold (missing data is left as-is)."""
    pred = PASSES.get(key)
    if value is not None and pred is not None and not pred(value):
        return f"{text} {FLAG_MARKER}"
    return text


def _table(title: str, sides: list[tuple]) -> list[str] | None:
    """Build one comparison table over N sides.

    Each side is ``(tag, {task: [bundles]}, kind)`` with ``kind`` in ``{'train', 'eval'}``.
    Metric columns are grouped by metric (one sub-column per side), then a trailing ``iters``
    column per train side (eval bundles have no iteration count).
    """
    tasks = sorted(set().union(*(set(data) for _, data, _ in sides))) if sides else []
    if not tasks:
        return None

    headers = ["task"]
    for h, _, _ in MCOLS:
        headers += [f"{h}_{tag}" for tag, _, _ in sides]
    # iters last, one column per train side (eval has no iterations)
    headers += [f"iters_{tag}" for tag, _, kind in sides if kind == "train"]

    rows = []
    for task in tasks:
        metrics = [(_avg_metrics(data.get(task), kind), kind) for _, data, kind in sides]
        row = [_disp(task)]
        for _, key, fmt in MCOLS:
            for m, _ in metrics:
                v = m.get(key)
                row.append(_emph(key, v, fmt(v)))
        for m, kind in metrics:
            if kind == "train":
                row.append(i0(m.get("iters")))
        rows.append(row)

    lines = [f"## {title}", "", "| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    lines.append("")
    return lines


def main() -> int:
    os.makedirs(_BENCH, exist_ok=True)
    px, mj, ov, ka = (BACKEND_DIRS["physx"], BACKEND_DIRS["mjwarp"], BACKEND_DIRS["ovphysx"], BACKEND_DIRS["kamino"])
    physx_tr, physx_ev = _load(px, "benchmark_training"), _load(px, "benchmark_play")
    mjwarp_tr, mjwarp_ev = _load(mj, "benchmark_training"), _load(mj, "benchmark_play")
    ovphysx_tr, ovphysx_ev = _load(ov, "benchmark_training"), _load(ov, "benchmark_play")
    kamino_tr, kamino_ev = _load(ka, "benchmark_training"), _load(ka, "benchmark_play")

    out = [
        "# Velocity benchmark summary",
        "",
        "Legend: **succ%** = success rate over completed episodes · **rew** = reward (train: final EMA, "
        "eval: mean over episodes) · **eplen** = episode length · **fps(k)** = total throughput [1000 steps/s] · "
        "**gpuGB** = peak GPU memory [GB] · **iters** = training iterations completed.",
        "",
        f"A {FLAG_MARKER} flags a cell below threshold: succ% ≤ {SUCC_THRESHOLD:g} or eplen ≤ {EPLEN_THRESHOLD:g}.",
        "",
        "Each metric is averaged over all available seeds per task.",
        "",
    ]
    tables = [
        _table("PhysX — train vs eval", [("tr", physx_tr, "train"), ("ev", physx_ev, "eval")]),
        _table("MJWarp — train vs eval", [("tr", mjwarp_tr, "train"), ("ev", mjwarp_ev, "eval")]),
        _table("OvPhysX — train vs eval", [("tr", ovphysx_tr, "train"), ("ev", ovphysx_ev, "eval")]),
        _table("Kamino — train vs eval", [("tr", kamino_tr, "train"), ("ev", kamino_ev, "eval")]),
        _table(
            "Train — cross-solver (PhysX vs MJWarp vs OvPhysX vs Kamino)",
            [
                ("physx", physx_tr, "train"),
                ("mjwarp", mjwarp_tr, "train"),
                ("ovphysx", ovphysx_tr, "train"),
                ("kamino", kamino_tr, "train"),
            ],
        ),
    ]
    for t in tables:
        if t is None:
            continue
        out += t

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
