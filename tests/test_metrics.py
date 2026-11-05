"""
Data-free unit tests for the teaching simulator's metric helpers.
No dataset download required, so CI can run these on every push.

Run:  pytest -q
"""
import os
import sys

import numpy as np
import pytest

# make simulator/ importable without installing the package
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "simulator"))

from metrics_sim import sigmoid, simulate, ece, evaluate  # noqa: E402


def test_sigmoid_bounds_and_monotonic():
    xs = np.array([-50.0, -1.0, 0.0, 1.0, 50.0])
    s = sigmoid(xs)
    assert np.all((s >= 0) & (s <= 1))
    assert abs(sigmoid(0.0) - 0.5) < 1e-12
    assert np.all(np.diff(s) > 0)  # strictly increasing


def test_simulate_shapes_and_balance():
    y, s = simulate(n=10_000, prevalence=0.05, separation=2.0, seed=0)
    assert y.shape == s.shape == (10_000,)
    assert set(np.unique(y)).issubset({0, 1})
    # prevalence recovered to the rounding used inside simulate()
    assert y.sum() == round(10_000 * 0.05)
    assert np.all((s >= 0) & (s <= 1))


def test_simulate_is_reproducible_with_seed():
    a = simulate(n=5_000, prevalence=0.1, separation=1.5, seed=42)
    b = simulate(n=5_000, prevalence=0.1, separation=1.5, seed=42)
    assert np.array_equal(a[0], b[0]) and np.allclose(a[1], b[1])


@pytest.mark.parametrize("kwargs", [
    dict(prevalence=0.0),     # must be strictly in (0,1)
    dict(prevalence=1.0),
    dict(prevalence=-0.1),
    dict(separation=-1.0),    # must be >= 0 and finite
    dict(separation=float("inf")),
    dict(n=1),                # must be >= 2
    dict(n=10, prevalence=1e-6),  # single-class sample at this n
])
def test_simulate_input_validation(kwargs):
    base = dict(n=1_000, prevalence=0.05, separation=2.0, seed=0)
    base.update(kwargs)
    with pytest.raises(ValueError):
        simulate(**base)


def test_separation_sets_auroc_order():
    # higher separation => strictly higher AUROC (the whole point of the knob)
    aurocs = []
    for sep in (1.0, 2.0, 3.0):
        y, s = simulate(n=40_000, prevalence=0.1, separation=sep, seed=0)
        aurocs.append(evaluate(y, s)["AUROC"])
    assert aurocs[0] < aurocs[1] < aurocs[2]
    assert aurocs[2] > 0.95


def test_monotonic_remap_preserves_auroc_but_changes_calibration():
    y, s = simulate(n=50_000, prevalence=0.1, separation=2.0, seed=0)
    base = evaluate(y, s)
    sharp = evaluate(y, s ** 3)
    flat = evaluate(y, np.sqrt(s))
    # ranking is invariant under a strictly monotonic remap
    assert abs(base["AUROC"] - sharp["AUROC"]) < 1e-9
    assert abs(base["AUROC"] - flat["AUROC"]) < 1e-9
    # but calibration / fixed-threshold metrics are not
    eces = {round(base["ECE"], 6), round(sharp["ECE"], 6), round(flat["ECE"], 6)}
    assert len(eces) == 3


def test_ece_zero_for_perfect_scores():
    # scores that equal the labels are perfectly calibrated within each bin
    y = np.array([0, 0, 1, 1])
    s = np.array([0.0, 0.0, 1.0, 1.0])
    assert ece(y, s) == pytest.approx(0.0, abs=1e-12)


def test_evaluate_threshold_validation():
    y, s = simulate(n=1_000, prevalence=0.1, separation=2.0, seed=0)
    with pytest.raises(ValueError):
        evaluate(y, s, threshold=1.5)
    with pytest.raises(ValueError):
        evaluate(y, s, threshold=-0.01)


def test_confusion_cells_sum_to_n():
    y, s = simulate(n=10_000, prevalence=0.05, separation=2.0, seed=0)
    r = evaluate(y, s, threshold=0.5)
    assert r["TP"] + r["FP"] + r["FN"] + r["TN"] == 10_000
