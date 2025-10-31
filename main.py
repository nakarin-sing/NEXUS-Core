#!/usr/bin/env python3
"""
main.py — NEXUS v7.0.0 FULL BENCHMARK
ประชัน 5 คู่แข่ง | ครองอันดับ 1 | CI/CD READY
"""

from __future__ import annotations

import numpy as np
import logging
import time
import json
from collections import deque
from tqdm import tqdm
from typing import Dict, Any, Iterable, Optional, Callable, Tuple, List, Literal
import random
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings
import sys
from contextlib import contextmanager

# ------------------ GITHUB ACTIONS FIXES ------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

try:
    import psutil
except ImportError:
    psutil = None

# ------------------ RIVER IMPORTS ------------------
from river import datasets, metrics, ensemble, tree, preprocessing
from river.base import Classifier
from nexus_core import NEXUS_River

warnings.filterwarnings("ignore")

# ------------------ CONFIGURATION ------------------
@dataclass(frozen=True)
class Config:
    seed: int = 42
    n_runs: int = 1
    max_snapshots: int = 3
    stress_history_len: int = 100
    datasets: Tuple[str, ...] = ("Electricity",)
    results_dir: str = "results"
    version: str = "7.0.0"
    max_samples: int = 1000

CONFIG = Config()
Path(CONFIG.results_dir).mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NEXUS")

random.seed(CONFIG.seed)
np.random.seed(CONFIG.seed)

if CONFIG.seed == 42:
    logger.info("หล่อทะลุจักรวาล mode activated!")

# ------------------ BASELINES — 5 คู่แข่ง + NEXUS ------------------
BASELINES = {
    "NEXUS": lambda: preprocessing.StandardScaler() | NEXUS_River(
        enable_ncra=True,
        enable_rfc=False,
        max_snapshots=CONFIG.max_snapshots
    ),
    "HATT": lambda: preprocessing.StandardScaler() | tree.HoeffdingAdaptiveTreeClassifier(seed=CONFIG.seed),
    "OzaBag": lambda: preprocessing.StandardScaler() | ensemble.BaggingClassifier(
        model=tree.HoeffdingTreeClassifier(),
        n_models=10,
        seed=CONFIG.seed
    ),
    "ARF": lambda: preprocessing.StandardScaler() | ensemble.AdaptiveRandomForestClassifier(
        n_models=10,
        seed=CONFIG.seed
    ),
    "SRP": lambda: preprocessing.StandardScaler() | ensemble.StreamingRandomPatchesClassifier(
        n_models=10,
        seed=CONFIG.seed
    ),
    "LB": lambda: preprocessing.StandardScaler() | ensemble.LeveragingBaggingClassifier(
        model=tree.HoeffdingTreeClassifier(),
        n_models=10,
        seed=CONFIG.seed
    ),
}

# ------------------ DATASETS ------------------
DATASET_MAP = {
    "Electricity": datasets.Elec2,
}

# ------------------ EVALUATION ------------------
def evaluate_model(model_cls: Callable[[], Any], dataset_name: str) -> pd.DataFrame:
    results = []
    for run in tqdm(range(CONFIG.n_runs), desc=dataset_name, leave=False):
        np.random.seed(CONFIG.seed + run)
        model = model_cls()
        if hasattr(model, "reset"):
            model.reset()
        metric = metrics.ROCAUC()
        start_time = time.perf_counter()
        start_mem = psutil.Process().memory_info().rss / 1024**2 if psutil else 0

        sample_count = 0
        try:
            for x, y in DATASET_MAP[dataset_name]():
                if sample_count >= CONFIG.max_samples:
                    break
                y_proba = model.predict_proba_one(x)
                p_true = y_proba.get(True, 0.5)
                model.learn_one(x, y)
                metric.update(y, p_true)
                sample_count += 1
        except Exception as e:
            logger.error(f"Error: {e}")
            results.append({"run": run, "AUC": 0.5, "Runtime": 0, "Memory_MB": 0, "samples": 0})
            continue

        runtime = time.perf_counter() - start_time
        memory = max(0, (psutil.Process().memory_info().rss / 1024**2 if psutil else 0) - start_mem)

        results.append({
            "run": run,
            "AUC": float(metric.get()),
            "Runtime": runtime,
            "Memory_MB": memory,
            "samples": sample_count
        })
    return pd.DataFrame(results)

# ------------------ LOGGING REDIRECT ------------------
@contextmanager
def logging_redirect(name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.info(f"{name}: {time.perf_counter() - start:.4f}s")

# ------------------ MAIN ------------------
def main() -> None:
    all_results = []
    for name in CONFIG.datasets:
        logger.info(f"Evaluating {name}")
        for model_name, model_cls in BASELINES.items():
            with logging_redirect(f"{name}-{model_name}"):
                df = evaluate_model(model_cls, name)
            df["Model"] = model_name
            df["Dataset"] = name
            all_results.append(df)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(f"{CONFIG.results_dir}/all_results.csv", index=False)

    summary = final_df.groupby(["Dataset", "Model"])["AUC"].mean().round(4).reindex(["NEXUS", "HATT", "OzaBag", "ARF", "SRP", "LB"], axis=1)
    rank = summary.rank(axis=1, ascending=False).loc[:, "NEXUS"]
    summary["Rank"] = [f"{int(r)}" + ("st" if r==1 else "nd" if r==2 else "rd" if r==3 else "th") for r in rank]

    summary.to_csv(f"{CONFIG.results_dir}/summary.csv")
    with open(f"{CONFIG.results_dir}/summary.md", "w") as f:
        f.write(f"# NEXUS v{CONFIG.version} — หล่อทะลุจักรวาล\n\n")
        f.write(f"**Mean AUC**: {summary.mean(axis=0)['NEXUS']:.4f}\n")
        f.write(f"**Rank**: {summary['Rank'].iloc[0]}\n\n")
        f.write(summary.to_markdown())

    plt.figure(figsize=(10, 6))
    sns.barplot(data=final_df, x="Dataset", y="AUC", hue="Model")
    plt.title("NEXUS v7.0.0 — FULL BENCHMARK")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{CONFIG.results_dir}/plot.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*80)
    print(f"NEXUS v{CONFIG.version} — FULL BENCHMARK")
    print(f"Mean AUC: {summary.mean(axis=0)['NEXUS']:.4f} | Rank: {summary['Rank'].iloc[0]}")
    print("="*80)
    print(summary.to_markdown())
    print("="*80)

    # --- แสดงไฟล์ใน results/ ---
    print("\n=== ไฟล์ใน results/ ===")
    results_path = Path(CONFIG.results_dir)
    if results_path.exists() and results_path.is_dir():
        files = list(results_path.iterdir())
        for file_path in sorted(files):
            if file_path.is_file():
                size = file_path.stat().st_size
                print(f"   • {file_path.name} ({size:,} bytes)")
        print(f"   รวมไฟล์: {len(files)}")
    else:
        print("   ไม่พบโฟลเดอร์ results/")
    print("=======================\n")

if __name__ == "__main__":
    main()
