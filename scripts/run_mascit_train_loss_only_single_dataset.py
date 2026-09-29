#!/usr/bin/env python3
"""Run the MASCIT train-loss-only sweep for exactly one pyrregular dataset."""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import os
import random
import resource
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

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


C2_ABBR_TO_DATASET = {
    "ABF": "Abf.h5",
    "MI3": "Mimic3.h5",
    "P12": "Physionet2012.h5",
    "P19": "Physionet2019.h5",
    "CT": "CharacterTrajectories.h5",
    "GM1": "GestureMidAirD1.h5",
    "GM2": "GestureMidAirD2.h5",
    "GM3": "GestureMidAirD3.h5",
    "GP1": "GesturePebbleZ1.h5",
    "GP2": "GesturePebbleZ2.h5",
    "GX": "AllGestureWiimoteX.h5",
    "GY": "AllGestureWiimoteY.h5",
    "GZ": "AllGestureWiimoteZ.h5",
    "LPA": "Ldfpa.h5",
    "PAM": "Pamap2.h5",
    "PGZ": "PickupGestureWiimoteZ.h5",
    "SGZ": "ShakeGestureWiimoteZ.h5",
    "AN": "Animals.h5",
    "AOC": "AsphaltObstaclesCoordinates.h5",
    "APT": "AsphaltPavementTypeCoordinates.h5",
    "ARC": "AsphaltRegularityCoordinates.h5",
    "GS": "Geolife.h5",
    "MP": "MelbournePedestrian.h5",
    "SE": "Seabirds.h5",
    "TA": "Taxi.h5",
    "VE": "Vehicles.h5",
    "DD": "DodgerLoopDay.h5",
    "DG": "DodgerLoopGame.h5",
    "DW": "DodgerLoopWeekend.h5",
    "IW": "InsectWingbeat.h5",
    "JV": "JapaneseVowels.h5",
    "PGE": "Garment.h5",
    "PL": "PLAID.h5",
    "SAD": "SpokenArabicDigits.h5",
}

EXTRA_ABBR_TO_DATASET = {
    "AIS": "Ais.h5",
    "CTB": "CombinedTrajectories.h5",
    "GSS": "GeolifeSupervised.h5",
    "TD": "TDrive.h5",
}

ALL_ABBR_TO_DATASET = {**C2_ABBR_TO_DATASET, **EXTRA_ABBR_TO_DATASET}


def resolve_dataset_name(raw_name: str) -> tuple[str, str]:
    """Map a short abbreviation or file name to the dataset file expected by pyrregular."""
    candidate = raw_name.strip()
    if not candidate:
        raise ValueError("Dataset name cannot be empty.")

    normalized = candidate.strip().lower()
    for abbr, dataset_file in ALL_ABBR_TO_DATASET.items():
        if candidate.upper() == abbr or candidate.lower() == dataset_file.lower():
            return abbr, dataset_file

    for dataset_file in ALL_ABBR_TO_DATASET.values():
        if normalized == dataset_file.lower() or normalized == dataset_file[:-3].lower():
            return next(abbr for abbr, value in ALL_ABBR_TO_DATASET.items() if value == dataset_file), dataset_file

    if candidate.endswith(".h5"):
        return candidate, candidate

    available = list_datasets()
    if candidate in available:
        return candidate, candidate
    if candidate.lower() in {x.lower() for x in available}:
        match = next(x for x in available if x.lower() == candidate.lower())
        return match, match

    known = ", ".join(sorted(ALL_ABBR_TO_DATASET.keys()))
    raise ValueError(f"Unknown dataset '{raw_name}'. Supported examples: {known} or a pyrregular filename like 'Physionet2012.h5'.")


def set_run_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_torch_model(wrapper):
    model_obj = getattr(wrapper, "model", None)
    torch_model = getattr(model_obj, "model", None)
    if isinstance(torch_model, torch.nn.DataParallel):
        torch_model = torch_model.module
    return torch_model if isinstance(torch_model, torch.nn.Module) else None


def count_parameters(wrapper):
    torch_model = resolve_torch_model(wrapper)
    if torch_model is None:
        return np.nan, np.nan

    total_params = sum(param.numel() for param in torch_model.parameters())
    trainable_params = sum(param.numel() for param in torch_model.parameters() if param.requires_grad)
    return int(total_params), int(trainable_params)


def uses_cuda(device):
    if isinstance(device, list) and len(device) > 0:
        return "cuda" in str(device[0])
    return "cuda" in str(device)


def current_rss_bytes():
    if psutil is not None:
        return psutil.Process(os.getpid()).memory_info().rss
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss_kb * 1024)


def measure_runtime_and_memory(fn, *, device):
    baseline_rss = current_rss_bytes()
    peak_rss = baseline_rss
    stop_event = threading.Event()

    def poll_memory():
        nonlocal peak_rss
        while not stop_event.wait(0.02):
            try:
                peak_rss = max(peak_rss, current_rss_bytes())
            except Exception:
                pass

    sampler = threading.Thread(target=poll_memory, daemon=True)
    sampler.start()

    gpu_peak_mb = np.nan
    cuda_enabled = torch.cuda.is_available() and uses_cuda(device)
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
        elapsed_sec = time.perf_counter() - started_at
    finally:
        stop_event.set()
        sampler.join(timeout=1)
        peak_rss = max(peak_rss, current_rss_bytes())

    return result, {
        "seconds": float(elapsed_sec),
        "peak_cpu_mb": float(peak_rss / (1024 ** 2)),
        "delta_cpu_mb": float((peak_rss - baseline_rss) / (1024 ** 2)),
        "peak_gpu_mb": float(gpu_peak_mb) if not np.isnan(gpu_peak_mb) else np.nan,
    }


def load_and_prepare_dataset(dataset_name: str):
    ds = load_dataset(dataset_name)
    da = ds.data
    X_all, _ = da.irr.to_dense(normalize_time=True)
    y_raw, split = da.irr.get_task_target_and_split()

    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(np.asarray(y_raw)).astype(np.int64)
    split = np.asarray(split)
    X_all = X_all.astype(np.float32)

    train_idx = np.where(split != "test")[0]
    test_idx = np.where(split == "test")[0]

    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    n_classes = len(np.unique(y_all))
    return X_train, X_test, y_train, y_test, n_classes, label_encoder


def cfg_signature(cfg):
    serialized = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()


def evaluate_trial(
    cfg,
    trial_id,
    output_root,
    dataset_abbr,
    dataset_name,
    X_train,
    y_train,
    X_test,
    y_test,
    n_classes,
    train_size,
    test_size,
    n_features,
    n_steps,
    seed,
    device,
    epochs,
    patience,
    batch_size,
    learning_rate,
):
    cfg_hash = cfg_signature(cfg)
    run_dir = output_root / dataset_abbr / f"cfg_{cfg_hash}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    n_kernels_dynamic = max(3, math.ceil(n_steps / 50))

    model_params = {
        **cfg,
        "n_kernels": n_kernels_dynamic,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
        "optimizer": AdamW(lr=learning_rate, weight_decay=1e-4),
        "device": device,
        "saving_path": str(run_dir),
        "model_saving_strategy": "best",
        "verbose": False,
    }

    set_run_seed(seed)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = MASCITTrainLossOnlyWrapper(
        model=MASCIT,
        model_params=model_params,
        random_state=seed,
    )

    _, train_stats = measure_runtime_and_memory(lambda: model.fit(X_train, y_train), device=device)
    n_parameters, n_trainable_parameters = count_parameters(model)

    proba, predict_stats = measure_runtime_and_memory(lambda: model.predict_proba(X_test), device=device)
    y_pred = np.argmax(proba, axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    try:
        if n_classes == 2:
            auc = roc_auc_score(y_test, proba[:, 1])
        else:
            auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
    except ValueError:
        auc = np.nan

    return {
        "trial_id": trial_id,
        "seed": int(seed),
        "cfg_hash": cfg_hash,
        "dataset_abbr": dataset_abbr,
        "dataset_name": dataset_name,
        "train_size": int(train_size),
        "test_size": int(test_size),
        "n_features": int(n_features),
        "n_steps": int(n_steps),
        "n_classes": int(n_classes),
        "n_kernels": int(n_kernels_dynamic),
        "n_parameters": int(n_parameters) if not np.isnan(n_parameters) else np.nan,
        "n_trainable_parameters": int(n_trainable_parameters) if not np.isnan(n_trainable_parameters) else np.nan,
        "train_seconds": train_stats["seconds"],
        "predict_seconds": predict_stats["seconds"],
        "train_peak_cpu_mb": train_stats["peak_cpu_mb"],
        "predict_peak_cpu_mb": predict_stats["peak_cpu_mb"],
        "train_delta_cpu_mb": train_stats["delta_cpu_mb"],
        "predict_delta_cpu_mb": predict_stats["delta_cpu_mb"],
        "train_peak_gpu_mb": train_stats["peak_gpu_mb"],
        "predict_peak_gpu_mb": predict_stats["peak_gpu_mb"],
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "auc": float(auc) if not np.isnan(auc) else np.nan,
        "error": "",
        **cfg,
    }


def load_sweep_space(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as file:
        sweep_space = json.load(file)

    required_keys = {
        "d_model",
        "d_state",
        "expand",
        "d_conv",
        "dropout",
        "projection_type",
        "n_heads",
        "tv_dt",
        "tv_B",
        "tv_C",
        "use_D",
    }
    if not isinstance(sweep_space, dict):
        raise ValueError(f"Sweep config must contain a JSON object: {config_path}")

    missing_keys = required_keys - sweep_space.keys()
    unknown_keys = sweep_space.keys() - required_keys
    if missing_keys or unknown_keys:
        raise ValueError(
            f"Invalid sweep keys in {config_path}. "
            f"Missing: {sorted(missing_keys)}; unknown: {sorted(unknown_keys)}"
        )
    empty_keys = [key for key, values in sweep_space.items() if not isinstance(values, list) or not values]
    if empty_keys:
        raise ValueError(f"Every sweep value must be a non-empty list. Invalid keys: {empty_keys}")

    return sweep_space


def build_trial_combos(sweep_space: dict, max_trials: int):
    keys = list(sweep_space.keys())
    all_combos = [dict(zip(keys, vals)) for vals in itertools.product(*(sweep_space[k] for k in keys))]
    return all_combos[:max_trials]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MASCIT train-loss-only sweep for one pyrregular dataset.")
    parser.add_argument("--dataset", required=True, help="Dataset abbreviation or file name, e.g. ABF or Physionet2012.h5")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset",
        help="PYRREGULAR dataset cache directory (default: %(default)s)",
    )
    parser.add_argument(
        "--sweep-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "mascit_sweep_128x16.json",
        help="JSON file defining the MASCIT sweep space (default: %(default)s)",
    )
    parser.add_argument("--seed", nargs="+", type=int, default=[2026, 2027, 2028], help="Random seeds to evaluate")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience in epochs")
    parser.add_argument("--max-trials", type=int, default=int(1e4), help="Maximum number of hyperparameter trials to evaluate")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory override")
    parser.add_argument("--reset-csv", action="store_true", help="Delete existing CSV before running")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device string")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    sweep_config = args.sweep_config.expanduser().resolve()
    sweep_space = load_sweep_space(sweep_config)
    os.environ["XDG_CACHE_HOME"] = str(dataset_dir)
    os.environ["POOCH_HOME"] = str(dataset_dir)

    global list_datasets, load_dataset, MASCIT, MASCITTrainLossOnlyWrapper, AdamW
    from pyrregular import list_datasets, load_dataset
    from classification.mascit import MASCIT, MASCITTrainLossOnlyWrapper
    from pypots.optim import AdamW

    dataset_abbr, dataset_name = resolve_dataset_name(args.dataset)
    print(f"Requested dataset: {args.dataset}")
    print(f"Resolved dataset name: {dataset_name}")
    print(f"Dataset cache directory: {dataset_dir}")
    print(f"Sweep config: {sweep_config}")
    print(f"Target device: {args.device}")

    output_root = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "testing_results" / "mascit_train_loss_only_single_dataset"
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / f"{dataset_abbr}_sweep_results.csv"

    if args.reset_csv and csv_path.exists():
        csv_path.unlink()

    results = []
    trial_id = 0
    completed = set()

    if csv_path.exists():
        try:
            existing = pd.read_csv(csv_path)
            if "trial_id" in existing.columns and existing["trial_id"].notna().any():
                trial_id = int(existing["trial_id"].dropna().max())

            if "dataset_abbr" in existing.columns and "seed" in existing.columns:
                for _, row in existing.iterrows():
                    row_dataset_abbr = row.get("dataset_abbr")
                    seed = row.get("seed")
                    cfg_hash = row.get("cfg_hash")
                    if pd.isna(row_dataset_abbr) or pd.isna(seed):
                        continue
                    if row_dataset_abbr != dataset_abbr:
                        continue
                    if pd.notna(cfg_hash) and str(cfg_hash).strip() != "":
                        completed.add((str(row_dataset_abbr), int(seed), str(cfg_hash)))
                        continue

                    cfg = {c: row[c] for c in existing.columns if c not in {"trial_id", "seed", "cfg_hash", "dataset_abbr", "dataset_name", "train_size", "test_size", "n_features", "n_steps", "n_classes", "n_kernels", "n_parameters", "n_trainable_parameters", "train_seconds", "predict_seconds", "train_peak_cpu_mb", "predict_peak_cpu_mb", "train_delta_cpu_mb", "predict_delta_cpu_mb", "train_peak_gpu_mb", "predict_peak_gpu_mb", "accuracy", "f1_macro", "auc", "error"} and not pd.isna(row[c])}
                    if cfg:
                        completed.add((str(row_dataset_abbr), int(seed), cfg_signature(cfg)))
        except Exception as exc:  # pragma: no cover
            print(f"Could not read existing CSV. Continuing fresh. Reason: {exc}")

    try:
        X_train, X_test, y_train, y_test, n_classes, _ = load_and_prepare_dataset(dataset_name)
    except Exception as exc:
        fail_row = {
            "trial_id": 1,
            "seed": int(args.seed[0]) if args.seed else 2026,
            "cfg_hash": "",
            "dataset_abbr": dataset_abbr,
            "dataset_name": dataset_name,
            "train_size": np.nan,
            "test_size": np.nan,
            "n_features": np.nan,
            "n_steps": np.nan,
            "n_classes": np.nan,
            "n_kernels": np.nan,
            "n_parameters": np.nan,
            "n_trainable_parameters": np.nan,
            "train_seconds": np.nan,
            "predict_seconds": np.nan,
            "train_peak_cpu_mb": np.nan,
            "predict_peak_cpu_mb": np.nan,
            "train_delta_cpu_mb": np.nan,
            "predict_delta_cpu_mb": np.nan,
            "train_peak_gpu_mb": np.nan,
            "predict_peak_gpu_mb": np.nan,
            "accuracy": np.nan,
            "f1_macro": np.nan,
            "auc": np.nan,
            "error": f"dataset_load_error: {exc}",
        }
        results.append(fail_row)
        pd.DataFrame([fail_row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        print(f"Dataset load failed for {dataset_name}: {exc}")
        return

    train_size = len(y_train)
    test_size = len(y_test)
    n_features = X_train.shape[1]
    n_steps = X_train.shape[2]

    trial_combos = build_trial_combos(sweep_space, args.max_trials)
    for cfg in tqdm(trial_combos, desc=f"{dataset_abbr}", unit="trial"):
        cfg_hash = cfg_signature(cfg)
        pending_seeds = [seed for seed in args.seed if (dataset_abbr, int(seed), cfg_hash) not in completed]
        if not pending_seeds:
            print(f"Skip duplicate experiment: {dataset_abbr} | {cfg}")
            continue

        for seed in pending_seeds:
            trial_id += 1
            try:
                row = evaluate_trial(
                    cfg=cfg,
                    trial_id=trial_id,
                    output_root=output_root,
                    dataset_abbr=dataset_abbr,
                    dataset_name=dataset_name,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    n_classes=n_classes,
                    train_size=train_size,
                    test_size=test_size,
                    n_features=n_features,
                    n_steps=n_steps,
                    seed=seed,
                    device=args.device,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                )
                completed.add((dataset_abbr, int(seed), cfg_hash))
            except Exception as exc:
                row = {
                    "trial_id": trial_id,
                    "seed": int(seed),
                    "cfg_hash": cfg_hash,
                    "dataset_abbr": dataset_abbr,
                    "dataset_name": dataset_name,
                    "train_size": int(train_size),
                    "test_size": int(test_size),
                    "n_features": int(n_features),
                    "n_steps": int(n_steps),
                    "n_classes": int(n_classes),
                    "n_kernels": int(max(3, math.ceil(n_steps / 50))),
                    "n_parameters": np.nan,
                    "n_trainable_parameters": np.nan,
                    "train_seconds": np.nan,
                    "predict_seconds": np.nan,
                    "train_peak_cpu_mb": np.nan,
                    "predict_peak_cpu_mb": np.nan,
                    "train_delta_cpu_mb": np.nan,
                    "predict_delta_cpu_mb": np.nan,
                    "train_peak_gpu_mb": np.nan,
                    "predict_peak_gpu_mb": np.nan,
                    "accuracy": np.nan,
                    "f1_macro": np.nan,
                    "auc": np.nan,
                    "error": str(exc),
                    **cfg,
                }
                completed.add((dataset_abbr, int(seed), cfg_hash))

            results.append(row)
            pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

    print(f"Done. CSV path: {csv_path}")
    print(f"Rows written: {len(results)}")
    print(f"Seeds used: {args.seed}")


if __name__ == "__main__":
    main()
