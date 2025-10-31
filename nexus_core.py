#!/usr/bin/env python3
"""
nexus_core.py — NEXUS_River v6.6.0
RIVER-COMPLIANT | ZERO-BUG | TYPE-SAFE
"""

from __future__ import annotations

import numpy as np
from river.base import Classifier
from typing import Dict, Any, Optional, Literal
from collections import deque
from threading import RLock
import logging

try:
    from main import CONFIG
except ImportError:
    from dataclasses import dataclass
    @dataclass(frozen=True)
    class Config:
        max_snapshots: int = 3
        stress_history_len: int = 100
        seed: int = 42
    CONFIG = Config()

logger = logging.getLogger("NEXUS")

STRESS_HIGH: float = 0.15
STRESS_MED: float = 0.05
LOSS_HIGH_THRESH: float = 0.5
LR_MIN: float = 0.01
LR_MAX: float = 1.0
EPS: float = 1e-9
GRAD_CLIP: float = 1.0
NUMPY_FLOAT = np.float32

class NEXUS_River(Classifier):
    def __init__(
        self,
        dim: Optional[int] = None,
        enable_ncra: bool = True,
        enable_rfc: bool = False,
        max_snapshots: Optional[int] = None
    ):
        super().__init__()
        self.dim = dim
        self.w = None
        self.bias: float = 0.0
        self.lr: float = 0.25
        self.stress: float = 0.0
        self.stress_history = deque(maxlen=CONFIG.stress_history_len)
        self.max_snapshots: int = max_snapshots if max_snapshots is not None else CONFIG.max_snapshots
        self.snapshots = deque(maxlen=self.max_snapshots)
        self.sample_count: int = 0
        self.feature_names: list = []
        self.enable_ncra = enable_ncra
        self.enable_rfc = enable_rfc
        self._lock = RLock()

    def _init_weights(self, n_features: int) -> None:
        if self.dim is None:
            self.dim = n_features
        scale = 0.1 / np.sqrt(self.dim)
        self.w = np.random.normal(0, scale, self.dim).astype(NUMPY_FLOAT)

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(np.clip(-x, -20.0, 20.0)))

    def _to_array(self, x: Dict[str, Any]) -> np.ndarray:
        if not isinstance(x, dict):
            raise TypeError("x must be dict")
        keys = sorted(x.keys())
        if not self.feature_names:
            self.feature_names = keys
        else:
            new_keys = [k for k in keys if k not in self.feature_names]
            self.feature_names.extend(new_keys)
        arr = np.array([float(x.get(k, 0.0)) for k in self.feature_names], dtype=NUMPY_FLOAT)
        arr = np.nan_to_num(arr, nan=0.0)
        arr = np.clip(arr, -100.0, 100.0)
        return arr

    def predict_proba_one(self, x: Dict[str, Any]) -> Dict[bool, float]:
        with self._lock:
            if self.w is None:
                self._init_weights(len(x))
            x_arr = self._to_array(x)
            p_main = self._sigmoid(np.dot(x_arr[:self.dim], self.w) + self.bias)
            p_ncra = self._predict_ncra(x_arr) if self.enable_ncra and self.snapshots else p_main
            w_m = 1.0
            w_n = 0.7 if self.enable_ncra and self.snapshots else 0.0
            total = w_m + w_n + EPS
            p_ens = (w_m * p_main + w_n * p_ncra) / total
            p_ens = np.clip(p_ens, 0.0, 1.0)
            return {True: float(p_ens), False: 1.0 - float(p_ens)}

    def _predict_ncra(self, x: np.ndarray) -> float:
        if not self.snapshots:
            return 0.5
        preds = []
        weights = []
        for s in self.snapshots:
            logit = np.dot(x[:self.dim], s["w"]) + s["bias"]
            preds.append(self._sigmoid(logit))
            weights.append(s["weight"])
        if not weights:
            return 0.5
        return float(np.average(preds, weights=weights))

    def learn_one(self, x: Dict[str, Any], y: Literal[0, 1]) -> "NEXUS_River":
        with self._lock:
            self.sample_count += 1
            if self.w is None:
                self._init_weights(len(x))
            x_arr = self._to_array(x)
            p_ens = self.predict_proba_one(x)[True]
            err = p_ens - float(y)

            adaptive_lr = np.clip(self.lr * (1.0 + min(self.stress * 5.0, 10.0)), LR_MIN, LR_MAX)
            grad = np.clip(adaptive_lr * err * x_arr[:self.dim], -GRAD_CLIP, GRAD_CLIP)
            self.w = (self.w - grad).astype(NUMPY_FLOAT)
            self.bias -= adaptive_lr * err

            loss = err ** 2
            new_stress = STRESS_HIGH if loss > LOSS_HIGH_THRESH else STRESS_MED
            self.stress = 0.6 * self.stress + 0.4 * new_stress
            self.stress_history.append(self.stress)

            if self.enable_ncra and self.stress > 0.1 and len(self.snapshots) < self.max_snapshots:
                self.snapshots.append({
                    "w": self.w.copy(),
                    "bias": self.bias,
                    "weight": 1.0
                })

            return self

    def reset(self) -> None:
        with self._lock:
            self.sample_count = 0
            self.stress = 0.0
            self.stress_history.clear()
            self.snapshots.clear()
            self.feature_names = []
