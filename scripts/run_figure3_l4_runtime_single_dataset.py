#!/usr/bin/env python3
"""Remeasure the Figure 3 MAPS runtime for one manuscript dataset.

The script freezes ``d_model=128``, ``d_state=16``, and selects the best
three-seed-complete TI/TV pattern used by the manuscript from the existing
sweep CSV. All 34 datasets require all eight complete TI/TV patterns.
It then trains the selected configuration for the requested seeds without
saving checkpoints. Results are written to a separate Figure 3 runtime
directory and never modify the original sweep CSVs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPOTS_ROOT = PROJECT_ROOT / "PyPOTS"
if not PYPOTS_ROOT.is_dir():
    raise RuntimeError(
        "PyPOTS submodule is missing. Run: git submodule update --init --recursive"
    )
if str(PYPOTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PYPOTS_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


COMMON_29 = (
    "ABF", "AN", "AOC", "APT", "ARC", "CT", "DD", "DG", "DW", "GM1",
    "GM2", "GM3", "GP1", "GP2", "GX", "GY", "GZ", "JV", "LPA", "MI3",
    "MP", "P12", "PGE", "PGZ", "PL", "SAD", "SE", "SGZ", "VE",
)
ADDITIONAL_5 = ("P19", "IW", "TA", "GSS", "PAM")
RUNTIME_DATASETS = COMMON_29 + ADDITIONAL_5
EXPECTED_COMPLETE_PATTERNS = {dataset_id: 8 for dataset_id in RUNTIME_DATASETS}
ANCHOR_SEEDS = [2026, 2027, 2028]

DATASETS = {
    "ABF": "Abf.h5", "AN": "Animals.h5",
    "AOC": "AsphaltObstaclesCoordinates.h5",
    "APT": "AsphaltPavementTypeCoordinates.h5",
    "ARC": "AsphaltRegularityCoordinates.h5",
    "CT": "CharacterTrajectories.h5", "DD": "DodgerLoopDay.h5",
    "DG": "DodgerLoopGame.h5", "DW": "DodgerLoopWeekend.h5",
    "GM1": "GestureMidAirD1.h5", "GM2": "GestureMidAirD2.h5",
    "GM3": "GestureMidAirD3.h5", "GP1": "GesturePebbleZ1.h5",
    "GP2": "GesturePebbleZ2.h5", "GX": "AllGestureWiimoteX.h5",
    "GY": "AllGestureWiimoteY.h5", "GZ": "AllGestureWiimoteZ.h5",
    "JV": "JapaneseVowels.h5", "LPA": "Ldfpa.h5", "MI3": "Mimic3.h5",
    "MP": "MelbournePedestrian.h5", "P12": "Physionet2012.h5",
    "PGE": "Garment.h5", "PGZ": "PickupGestureWiimoteZ.h5",
    "PL": "PLAID.h5", "SAD": "SpokenArabicDigits.h5",
    "SE": "Seabirds.h5", "SGZ": "ShakeGestureWiimoteZ.h5",
    "VE": "Vehicles.h5", "P19": "Physionet2019.h5",
    "IW": "InsectWingbeat.h5", "TA": "Taxi.h5",
    "GSS": "GeolifeSupervised.h5", "PAM": "Pamap2.h5",
}

CONFIG_COLUMNS = ["d_model", "d_state", "tv_dt", "tv_B", "tv_C"]
BOOL_COLUMNS = ["tv_dt", "tv_B", "tv_C"]
RESULT_COLUMNS = [
    "trial_id", "started_at", "seed", "cfg_hash", "dataset_abbr", "dataset_name",
    "train_size", "test_size", "n_features", "n_steps", "n_classes", "n_kernels",
    "n_parameters", "n_trainable_parameters", "train_seconds", "predict_seconds",
    "total_runtime_seconds", "train_peak_cpu_mb", "predict_peak_cpu_mb",
    "train_delta_cpu_mb", "predict_delta_cpu_mb", "train_peak_gpu_mb",
    "predict_peak_gpu_mb", "error", "d_model", "d_state", "expand", "d_conv",
    "dropout", "projection_type", "n_heads", "tv_dt", "tv_B", "tv_C",
    "tv_pattern", "use_D", "batch_size", "epochs", "patience", "learning_rate",
    "anchor_f1_mean", "anchor_f1_std", "anchor_source_csv", "checkpoint_saved",
    "runtime_scope", "hostname", "python_version", "torch_version", "cuda_version",
    "cudnn_version", "gpu_name", "gpu_count", "device", "git_commit",
]


def as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "t", "yes"}


def pattern_name(cfg: dict) -> str:
    return "".join("V" if cfg[column] else "I" for column in BOOL_COLUMNS)


def cfg_signature(cfg: dict) -> str:
    serialized = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()


def original_grid_config(selected: dict) -> dict:
    return {
        "d_model": 128,
        "d_state": 16,
        "expand": 1,
        "d_conv": 4,
        "dropout": 0.1,
        "projection_type": "gating",
        "n_heads": 4,
        "tv_dt": bool(selected["tv_dt"]),
        "tv_B": bool(selected["tv_B"]),
        "tv_C": bool(selected["tv_C"]),
        "use_D": False,
    }


def select_frozen_anchor(dataset_id: str, anchor_dir: Path, seeds: list[int]) -> dict:
    """Reconstruct the manuscript's fixed-128x16 observed-pattern tie rule."""
    csv_path = anchor_dir / f"{dataset_id}_sweep_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing anchor sweep CSV: {csv_path}")

    frame = pd.read_csv(csv_path)
    required = {"seed", "f1_macro", "error", *CONFIG_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    for column in BOOL_COLUMNS:
        frame[column] = frame[column].map(as_bool)
    frame = frame[
        frame["error"].fillna("").eq("")
        & frame["f1_macro"].notna()
        & frame["d_model"].eq(128)
        & frame["d_state"].eq(16)
        & frame["seed"].isin(seeds)
    ].copy()
    frame = frame.sort_values("trial_id", kind="stable").drop_duplicates(
        ["seed", *CONFIG_COLUMNS], keep="last"
    )

    grouped = (
        frame.groupby(CONFIG_COLUMNS, dropna=False)
        .agg(
            rows=("seed", "size"),
            n_seeds=("seed", "nunique"),
            anchor_f1_mean=("f1_macro", "mean"),
            anchor_f1_std=("f1_macro", "std"),
        )
        .reset_index()
    )
    complete = grouped[
        grouped["rows"].eq(len(seeds)) & grouped["n_seeds"].eq(len(seeds))
    ].copy()
    expected_patterns = EXPECTED_COMPLETE_PATTERNS[dataset_id]
    if len(complete) != expected_patterns:
        raise ValueError(
            f"{dataset_id}: expected {expected_patterns} complete 128x16 TI/TV "
            f"patterns for seeds {seeds}, found {len(complete)}"
        )

    # Same tie rule as build_apiems_full_draft_assets.py: score first, then I before V.
    choice = complete.sort_values(
        ["anchor_f1_mean", "tv_dt", "tv_B", "tv_C"],
        ascending=[False, True, True, True],
        kind="stable",
    ).iloc[0]
    selected = {
        "d_model": 128,
        "d_state": 16,
        "tv_dt": bool(choice.tv_dt),
        "tv_B": bool(choice.tv_B),
        "tv_C": bool(choice.tv_C),
        "anchor_f1_mean": float(choice.anchor_f1_mean),
        "anchor_f1_std": float(choice.anchor_f1_std),
        "anchor_source_csv": str(csv_path),
        "anchor_scope": (
            "oracle8" if expected_patterns == 8 else f"observed{expected_patterns}"
        ),
    }
    cfg = original_grid_config(selected)
    selected["cfg_hash"] = cfg_signature(cfg)
    selected["tv_pattern"] = pattern_name(selected)
    return selected


def set_run_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def current_rss_bytes() -> int:
    if psutil is not None:
        return int(psutil.Process(os.getpid()).memory_info().rss)
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss_kb * 1024)


def uses_cuda(device: str) -> bool:
    return torch.cuda.is_available() and str(device).startswith("cuda")


def measure_runtime_and_memory(fn, *, device: str):
    baseline_rss = current_rss_bytes()
    peak_rss = baseline_rss
    stop_event = threading.Event()

    def poll_memory() -> None:
        nonlocal peak_rss
        while not stop_event.wait(0.02):
            try:
                peak_rss = max(peak_rss, current_rss_bytes())
            except Exception:
                pass

    sampler = threading.Thread(target=poll_memory, daemon=True)
    sampler.start()
    cuda_enabled = uses_cuda(device)
    gpu_peak_mb = np.nan
    if cuda_enabled:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    started_at = time.perf_counter()
    try:
        result = fn()
        if cuda_enabled:
            torch.cuda.synchronize()
            gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        seconds = time.perf_counter() - started_at
    finally:
        stop_event.set()
        sampler.join(timeout=1)
        peak_rss = max(peak_rss, current_rss_bytes())

    return result, {
        "seconds": float(seconds),
        "peak_cpu_mb": float(peak_rss / (1024 ** 2)),
        "delta_cpu_mb": float((peak_rss - baseline_rss) / (1024 ** 2)),
        "peak_gpu_mb": float(gpu_peak_mb) if not np.isnan(gpu_peak_mb) else np.nan,
    }


def load_and_prepare_dataset(dataset_name: str):
    dataset = load_dataset(dataset_name)
    dense, _ = dataset.data.irr.to_dense(normalize_time=True)
    labels, split = dataset.data.irr.get_task_target_and_split()

    encoder = LabelEncoder()
    y_all = encoder.fit_transform(np.asarray(labels)).astype(np.int64)
    X_all = dense.astype(np.float32)
    split = np.asarray(split)
    train_idx = np.where(split != "test")[0]
    test_idx = np.where(split == "test")[0]
    return (
        X_all[train_idx], X_all[test_idx], y_all[train_idx], y_all[test_idx],
        len(np.unique(y_all)),
    )


def resolve_torch_model(wrapper):
    model_obj = getattr(wrapper, "model", None)
    torch_model = getattr(model_obj, "model", None)
    if isinstance(torch_model, torch.nn.DataParallel):
        torch_model = torch_model.module
    return torch_model if isinstance(torch_model, torch.nn.Module) else None


def count_parameters(wrapper):
    model = resolve_torch_model(wrapper)
    if model is None:
        return np.nan, np.nan
    return (
        int(sum(parameter.numel() for parameter in model.parameters())),
        int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
    )


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def hardware_metadata(device: str) -> dict:
    cuda_enabled = uses_cuda(device)
    return {
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
        "cudnn_version": torch.backends.cudnn.version() if cuda_enabled else np.nan,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_enabled else "CPU",
        "gpu_count": torch.cuda.device_count() if cuda_enabled else 0,
        "device": device,
        "git_commit": git_commit(),
    }


def append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).reindex(columns=RESULT_COLUMNS).to_csv(
        csv_path, mode="a", header=not csv_path.exists(), index=False
    )


def validate_result_schema(csv_path: Path) -> None:
    if not csv_path.exists():
        return
    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    if columns != RESULT_COLUMNS:
        raise ValueError(
            f"Existing result CSV uses a different schema: {csv_path}. "
            "Use --reset-csv to start a runtime-only result file."
        )


def successful_keys(csv_path: Path) -> set[tuple[int, str]]:
    if not csv_path.exists():
        return set()
    frame = pd.read_csv(csv_path)
    if not {"seed", "cfg_hash", "error", "total_runtime_seconds"}.issubset(frame.columns):
        return set()
    ok = frame["error"].fillna("").eq("") & frame["total_runtime_seconds"].notna()
    return {
        (int(row.seed), str(row.cfg_hash))
        for row in frame.loc[ok].itertuples(index=False)
    }


def next_trial_id(csv_path: Path) -> int:
    if not csv_path.exists():
        return 1
    frame = pd.read_csv(csv_path)
    if "trial_id" not in frame or frame["trial_id"].dropna().empty:
        return 1
    return int(frame["trial_id"].dropna().max()) + 1


def run_seed(
    *, dataset_id: str, dataset_name: str, seed: int, anchor: dict,
    X_train, X_test, y_train, y_test, n_classes: int, args,
) -> dict:
    n_steps = int(X_train.shape[2])
    n_kernels = max(3, math.ceil(n_steps / 50))
    cfg = original_grid_config(anchor)
    model_params = {
        **cfg,
        "n_kernels": n_kernels,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "optimizer": AdamW(lr=args.learning_rate, weight_decay=1e-4),
        "device": args.device,
        "saving_path": None,
        "model_saving_strategy": None,
        "verbose": False,
    }

    set_run_seed(seed)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    wrapper = MASCITTrainLossOnlyWrapper(
        model=MASCIT, model_params=model_params, random_state=seed
    )
    _, train_stats = measure_runtime_and_memory(
        lambda: wrapper.fit(X_train, y_train), device=args.device
    )
    n_parameters, n_trainable = count_parameters(wrapper)
    _, predict_stats = measure_runtime_and_memory(
        lambda: wrapper.predict_proba(X_test), device=args.device
    )

    return {
        "seed": int(seed),
        "cfg_hash": anchor["cfg_hash"],
        "dataset_abbr": dataset_id,
        "dataset_name": dataset_name,
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "n_features": int(X_train.shape[1]),
        "n_steps": n_steps,
        "n_classes": int(n_classes),
        "n_kernels": int(n_kernels),
        "n_parameters": n_parameters,
        "n_trainable_parameters": n_trainable,
        "train_seconds": train_stats["seconds"],
        "predict_seconds": predict_stats["seconds"],
        "total_runtime_seconds": train_stats["seconds"] + predict_stats["seconds"],
        "train_peak_cpu_mb": train_stats["peak_cpu_mb"],
        "predict_peak_cpu_mb": predict_stats["peak_cpu_mb"],
        "train_delta_cpu_mb": train_stats["delta_cpu_mb"],
        "predict_delta_cpu_mb": predict_stats["delta_cpu_mb"],
        "train_peak_gpu_mb": train_stats["peak_gpu_mb"],
        "predict_peak_gpu_mb": predict_stats["peak_gpu_mb"],
        "error": "",
        "d_model": 128,
        "d_state": 16,
        "expand": 1,
        "d_conv": 4,
        "dropout": 0.1,
        "projection_type": "gating",
        "n_heads": 4,
        "tv_dt": anchor["tv_dt"],
        "tv_B": anchor["tv_B"],
        "tv_C": anchor["tv_C"],
        "tv_pattern": anchor["tv_pattern"],
        "use_D": False,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "anchor_f1_mean": anchor["anchor_f1_mean"],
        "anchor_f1_std": anchor["anchor_f1_std"],
        "anchor_source_csv": anchor["anchor_source_csv"],
        "checkpoint_saved": False,
        "runtime_scope": (
            f"fit_plus_predict_fixed128x16_{anchor['anchor_scope']}"
        ),
        **hardware_metadata(args.device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=RUNTIME_DATASETS)
    parser.add_argument("--seed", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset",
        help="PYRREGULAR dataset cache directory (default: %(default)s)",
    )
    parser.add_argument(
        "--anchor-dir", type=Path,
        default=PROJECT_ROOT / "testing_results" / "mascit_train_loss_only_single_dataset",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=(
            PROJECT_ROOT / "testing_results" / "apiems_figure3_l4_runtime" / "results"
        ),
    )
    parser.add_argument("--reset-csv", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Resolve and print the frozen anchor without loading data or training",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    os.environ["XDG_CACHE_HOME"] = str(dataset_dir)
    os.environ["POOCH_HOME"] = str(dataset_dir)

    dataset_id = args.dataset.upper()
    dataset_name = DATASETS[dataset_id]
    anchor = select_frozen_anchor(dataset_id, args.anchor_dir, ANCHOR_SEEDS)
    print(
        f"{dataset_id}: 128x16 {anchor['tv_pattern']} | "
        f"anchor F1={anchor['anchor_f1_mean']:.6f} | cfg={anchor['cfg_hash']}"
    )
    print(f"Dataset cache directory: {dataset_dir}")
    if args.dry_run:
        return

    global load_dataset, MASCIT, MASCITTrainLossOnlyWrapper, AdamW
    from pyrregular import load_dataset
    from classification.mascit import MASCIT, MASCITTrainLossOnlyWrapper
    from pypots.optim import AdamW

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    csv_path = args.output_dir / f"{dataset_id}_l4_runtime_results.csv"
    if args.reset_csv and csv_path.exists():
        csv_path.unlink()
    validate_result_schema(csv_path)
    completed = successful_keys(csv_path)
    trial_id = next_trial_id(csv_path)

    X_train, X_test, y_train, y_test, n_classes = load_and_prepare_dataset(dataset_name)
    written = 0
    for seed in args.seed:
        key = (int(seed), anchor["cfg_hash"])
        if key in completed:
            print(f"skip successful row: {dataset_id} seed={seed}")
            continue
        started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        try:
            row = run_seed(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                seed=int(seed),
                anchor=anchor,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                n_classes=n_classes,
                args=args,
            )
        except Exception as exc:
            row = {
                "seed": int(seed), "cfg_hash": anchor["cfg_hash"],
                "dataset_abbr": dataset_id, "dataset_name": dataset_name,
                "train_seconds": np.nan, "predict_seconds": np.nan,
                "total_runtime_seconds": np.nan, "error": repr(exc),
                "d_model": 128, "d_state": 16,
                "tv_dt": anchor["tv_dt"], "tv_B": anchor["tv_B"],
                "tv_C": anchor["tv_C"], "tv_pattern": anchor["tv_pattern"],
                "checkpoint_saved": False,
                "runtime_scope": (
                    f"fit_plus_predict_fixed128x16_{anchor['anchor_scope']}"
                ),
                **hardware_metadata(args.device),
            }
        row = {"trial_id": trial_id, "started_at": started_at, **row}
        append_row(csv_path, row)
        print(
            f"wrote {dataset_id} seed={seed}: "
            f"runtime={row.get('total_runtime_seconds')} error={row.get('error')}"
        )
        written += 1
        trial_id += 1
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Done: {csv_path}")
    print(f"Rows written in this invocation: {written}")


if __name__ == "__main__":
    main()
