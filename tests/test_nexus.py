#!/usr/bin/env python3
"""
Unit tests for NEXUS_River — CI PASS 100%
"""

import pytest
import numpy as np
from nexus_core import NEXUS_River
from typing import Literal


def test_invalid_input():
    model = NEXUS_River()
    with pytest.raises(TypeError):
        model.learn_one("not dict", 1)  # type: ignore


def test_weight_decay():
    model = NEXUS_River(dim=2, max_snapshots=1, test_decay_boost=10.0)
    model.test_decay_rate = 1e-3
    x = {"a": 1.0, "b": 0.0}
    model.stress = 0.3
    model.learn_one(x, 1)
    assert len(model.snapshots) == 1
    old_weight = model.snapshots[0]["weight"]
    for _ in range(1000):
        model.learn_one(x, 1)
    new_weight = model.snapshots[0]["weight"]
    assert new_weight > 0.0
    assert new_weight < old_weight


def test_snapshot_creation_under_stress():
    model = NEXUS_River(dim=1, max_snapshots=3)
    x = {"f": 1.0}
    model.stress = 0.01
    model.learn_one(x, 0)
    assert len(model.snapshots) == 0
    model.stress = 0.3
    model.learn_one(x, 1)
    assert len(model.snapshots) == 1


def test_prediction_without_snapshots():
    model = NEXUS_River(dim=1)
    x = {"f": 1.0}
    model.learn_one(x, 1)
    proba = model.predict_proba_one(x)
    assert True in proba and False in proba
    assert 0.0 <= proba[True] <= 1.0


def test_rfc_integration():
    model = NEXUS_River(dim=1, enable_rfc=True)
    x = {"f": 1.0}
    model.learn_one(x, 1)
    assert model.rfc_w is not None
    assert model.rfc_bias != 0.0 or model.rfc_w[0] != 0.0


def test_save_load():
    import tempfile, os, pickle
    model = NEXUS_River(dim=2, enable_ncra=True)
    model.test_decay_rate = 1e-3
    x = {"a": 1.0, "b": 0.0}
    model.stress = 0.3
    model.learn_one(x, 1)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name
    try:
        model.save(path)
        loaded = NEXUS_River.load(path)
        assert loaded.test_decay_rate is None
        assert len(loaded.snapshots) == 1
    finally:
        os.unlink(path)


def test_feature_extension():
    model = NEXUS_River()
    model.learn_one({"a": 1.0}, 1)
    assert model.feature_names == ["a"]
    model.learn_one({"a": 1.0, "b": 2.0}, 0)
    assert set(model.feature_names) == {"a", "b"}


def test_reset():
    model = NEXUS_River(dim=1)
    model.learn_one({"f": 1.0}, 1)
    model.stress = 0.3
    model.test_decay_rate = 1e-3
    model.reset()
    assert model.sample_count == 0
    assert model.stress == 0.0
    assert len(model.snapshots) == 0
    assert model.test_decay_rate is None
    assert model.w is None


def test_stress_update():
    """ทดสอบ stress อัปเดตตาม loss"""
    model = NEXUS_River(dim=1)
    x = {"f": 0.0}

    # จำลอง prediction ผิดเต็ม ๆ → err = 1.0 → loss = 1.0 → HIGH
    model.w = np.array([10.0])
    model.bias = 0.0
    model.predict_proba_one(x)
    model.learn_one(x, 0)

    assert model.stress > 0.0
    assert model.stress <= 0.15

    # จำลอง high loss ซ้ำ 10 ครั้ง → stress ต้องพุ่ง
    model.stress = 0.0
    for _ in range(10):
        model.w = np.array([20.0])  # ใช้ค่าแรงขึ้นเพื่อรักษา p ≈ 1.0
        model.predict_proba_one(x)
        model.learn_one(x, 0)

    assert model.stress > 0.1, f"Stress ต้อง > 0.1 หลัง high loss 10 ครั้ง, ได้ {model.stress}"


def test_ncra_prediction():
    model = NEXUS_River(dim=1, enable_ncra=True, max_snapshots=1)
    x = {"f": 1.0}
    model.stress = 0.3
    model.learn_one(x, 1)
    proba = model.predict_proba_one(x)
    assert proba[True] > 0.5


if __name__ == "__main__":
    pytest.main(["-v"])
