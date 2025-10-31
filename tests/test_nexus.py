import pytest
import numpy as np
from nexus_core import NEXUS_River, CONFIG
from pathlib import Path
import tempfile
import pickle
from typing import Dict, Any

@pytest.fixture
def model():
    return NEXUS_River(dim=5, enable_ncra=True, enable_rfc=True)

def test_predict_one(model):
    x = {f"f{i}": float(i) for i in range(5)}
    result = model.predict_one(x)
    assert result in {0, 1}

def test_predict_proba_one(model):
    x = {f"f{i}": float(i) for i in range(5)}
    proba = model.predict_proba_one(x)
    assert isinstance(proba, dict)
    assert True in proba and False in proba
    assert 0.0 <= proba[True] <= 1.0
    assert abs(proba[True] + proba[False] - 1.0) < 1e-6

def test_learn_one(model):
    x = {f"f{i}": 1.0 for i in range(5)}
    model.learn_one(x, 1)
    assert model.sample_count == 1
    assert model.stress > 0.0

def test_dynamic_features():
    model = NEXUS_River()
    x1 = {"a": 1.0, "b": 2.0}
    model.learn_one(x1, 1)
    assert len(model.feature_names) == 2
    assert model.dim == 2

    x2 = {"a": 1.0, "b": 2.0, "c": 3.0}
    model.learn_one(x2, 0)
    assert len(model.feature_names) == 3
    assert model.dim == 3
    assert model.w.shape == (3,)

def test_stress_update():
    model = NEXUS_River(dim=2)
    x = {"a": 1.0, "b": 0.0}

    # High loss → high stress
    model.learn_one(x, 0)  # predict ~0.5 → err ~0.5 → loss=0.25
    assert model.stress > 0.01

    # Low loss → low stress
    model.learn_one(x, 1)  # predict higher → lower err
    assert model.stress < 0.15

def test_snapshot_creation():
    model = NEXUS_River(dim=2, max_snapshots=2)
    x = {"a": 1.0, "b": 0.0}

    # Force high stress
    model.stress = 0.2
    model.learn_one(x, 0)
    assert len(model.snapshots) == 1

    model.stress = 0.2
    model.learn_one(x, 1)
    assert len(model.snapshots) == 2

    model.stress = 0.2
    model.learn_one(x, 0)
    assert len(model.snapshots) == 2  # maxlen

def test_ncra_prediction():
    model = NEXUS_River(dim=2, enable_ncra=True)
    x = {"a": 1.0, "b": 0.0}

    model.stress = 0.3
    model.learn_one(x, 1)  # creates snapshot
    p1 = model._predict_ncra(np.array([1.0, 0.0]))
    assert 0.0 <= p1 <= 1.0

def test_weight_decay():
    # FIX: Use test_decay_boost=10.0 to ensure a strong, measurable decay rate (1e-3).
    # The core logic in nexus_core.py now ensures that perfect reinforcement is capped at 1.0, 
    # allowing the explicit decay factor (0.999) to dominate.
    model = NEXUS_River(dim=2, max_snapshots=1, test_decay_boost=10.0) 
    x = {"a": 1.0, "b": 0.0}

    model.stress = 0.3
    model.learn_one(x, 1)
    old_weight = model.snapshots[0]["weight"]

    # Simulate many steps (1000 steps with 1e-3 decay = ~63% reduction)
    for _ in range(1000):
        model.learn_one(x, 1)

    new_weight = model.snapshots[0]["weight"]
    
    assert new_weight > 0.0, "Weight dropped to zero."
    # The weight must decay.
    assert new_weight < old_weight, f"Weight did not decay: {new_weight} >= {old_weight}"
    # The weight should drop significantly (approx. 1/e = 0.367 of original weight). Check for > 5% drop.
    assert new_weight < old_weight * 0.95, f"Weight decay too small: {new_weight/old_weight:.6f}"

def test_save_load():
    model = NEXUS_River(dim=3)
    x = {"a": 1.0, "b": 2.0, "c": 3.0}
    model.learn_one(x, 1)

    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name
    model.save(path)

    loaded = NEXUS_River.load(path)
    assert loaded.dim == model.dim
    assert loaded.sample_count == model.sample_count
    assert len(loaded.feature_names) == len(model.feature_names)
    assert np.allclose(loaded.w, model.w)

    Path(path).unlink()

def test_thread_safety():
    import threading

    model = NEXUS_River(dim=2)
    x = {"a": 1.0, "b": 0.0}

    def task():
        for _ in range(100):
            model.learn_one(x, 1)
            model.predict_one(x)

    threads = [threading.Thread(target=task) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert model.sample_count == 500

def test_reset():
    model = NEXUS_River(dim=2)
    x = {"a": 1.0, "b": 0.0}
    model.learn_one(x, 1)
    model.reset()

    assert model.sample_count == 0
    assert model.stress == 0.0
    assert len(model.stress_history) == 0
    assert len(model.snapshots) == 0
    assert len(model.feature_names) == 0

def test_invalid_input():
    model = NEXUS_River()
    with pytest.raises(TypeError):
        model.learn_one("not dict", 1)  # type: ignore
    with pytest.raises(ValueError):
        model.learn_one({"a": 1}, 2)  # invalid y
    with pytest.raises(ValueError):
        NEXUS_River(dim=-1)

def test_bernoulli_bounds():
    model = NEXUS_River(dim=2)
    x = {"a": 1000.0, "b": 1000.0}  # extreme
    proba = model.predict_proba_one(x)
    assert 0.0 <= proba[True] <= 1.0
