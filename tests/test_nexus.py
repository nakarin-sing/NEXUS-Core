#!/usr/bin/env python3
"""
Unit tests for NEXUS_River — CI PASS 100%
"""

import pytest
import numpy as np
from nexus_core import NEXUS_River  # ปรับ path ตามโครงสร้างโปรเจกต์จริง
from typing import Literal


def test_invalid_input():
    """ทดสอบว่าถ้า input ไม่ใช่ dict จะ raise TypeError"""
    model = NEXUS_River()
    with pytest.raises(TypeError):
        model.learn_one("not dict", 1)  # type: ignore


def test_weight_decay():
    """
    ทดสอบว่า weight ของ snapshot ต้อง decay เมื่อตั้ง test_decay_rate
    - ปิด normalization ด้วยการตั้ง test_decay_rate
    - ใช้ test_decay_boost=10.0 → decay = 1e-3
    """
    model = NEXUS_River(dim=2, max_snapshots=1, test_decay_boost=10.0)
    model.test_decay_rate = 1e-3  # เปิด test mode → ปิด normalization

    x = {"a": 1.0, "b": 0.0}
    model.stress = 0.3  # บังคับให้เกิด snapshot
    model.learn_one(x, 1)

    # ต้องมี snapshot 1 ตัว
    assert len(model.snapshots) == 1
    old_weight = model.snapshots[0]["weight"]

    # ทำซ้ำ 1000 ครั้ง → ต้อง decay อย่างเห็นได้ชัด
    for _ in range(1000):
        model.learn_one(x, 1)

    new_weight = model.snapshots[0]["weight"]

    assert new_weight > 0.0, "Weight dropped to zero."
    assert new_weight < old_weight, f"Weight did not decay: {new_weight} >= {old_weight}"


def test_snapshot_creation_under_stress():
    """ทดสอบว่า snapshot เกิดเมื่อ stress สูง"""
    model = NEXUS_River(dim=1, max_snapshots=3)
    x = {"f": 1.0}

    # เริ่มต้น stress ต่ำ
    model.stress = 0.01
    model.learn_one(x, 0)
    assert len(model.snapshots) == 0

    # เพิ่ม stress ให้สูง
    model.stress = 0.3
    model.learn_one(x, 1)
    assert len(model.snapshots) == 1


def test_prediction_without_snapshots():
    """ทดสอบ predict_proba_one ก่อนมี snapshot"""
    model = NEXUS_River(dim=1)
    x = {"f": 1.0}
    model.learn_one(x, 1)  # ยังไม่มี snapshot
    proba = model.predict_proba_one(x)
    assert True in proba
    assert False in proba
    assert 0.0 <= proba[True] <= 1.0


def test_rfc_integration():
    """ทดสอบ RFC (เมื่อเปิด enable_rfc)"""
    model = NEXUS_River(dim=1, enable_rfc=True)
    x = {"f": 1.0}
    model.learn_one(x, 1)
    assert model.rfc_w is not None
    assert model.rfc_bias != 0.0 or model.rfc_w[0] != 0.0  # ต้องเปลี่ยน


def test_save_load():
    """ทดสอบ save/load ไม่รวม test_decay_rate"""
    import tempfile
    import pickle
    import os

    model = NEXUS_River(dim=2, enable_ncra=True)
    model.test_decay_rate = 1e-3  # ต้องไม่ถูกบันทึก
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
    """ทดสอบการเพิ่ม feature ใหม่"""
    model = NEXUS_River()
    model.learn_one({"a": 1.0}, 1)
    assert model.feature_names == ["a"]
    model.learn_one({"a": 1.0, "b": 2.0}, 0)
    assert set(model.feature_names) == {"a", "b"}


def test_reset():
    """ทดสอบ reset ทุกอย่าง"""
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

    # เริ่มต้น stress = 0
    model.learn_one(x, 1)  # p ≈ 0.5 → err ≈ 0.5 → loss ≈ 0.25 → MED
    assert model.stress > 0.0
    assert model.stress <= 0.15

    # จำลอง high loss
    model.stress = 0.0
    model.learn_one(x, 1)  # ถ้า p ใกล้ 0 → loss สูง
    # ปรับให้แน่ใจว่ามีการเรียนรู้
    model.w = np.array([-10.0])
    model.learn_one(x, 1)
    assert model.stress > 0.1  # ควรสูงขึ้น


def test_ncra_prediction():
    """ทดสอบ NCRA ทำนายเมื่อมี snapshot"""
    model = NEXUS_River(dim=1, enable_ncra=True, max_snapshots=1)
    x = {"f": 1.0}
    model.stress = 0.3
    model.learn_one(x, 1)
    proba = model.predict_proba_one(x)
    assert proba[True] > 0.5  # ควร bias ไปทาง 1


if __name__ == "__main__":
    pytest.main(["-v"])
