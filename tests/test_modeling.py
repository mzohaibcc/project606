"""Independent behavioral checks of isolation, interpolation and calibration."""
import numpy as np
import pandas as pd
import pytest

from toxbench import modeling
from toxbench.modeling import SigmoidCalibrator, resample, tune_fit
from toxbench.runner import fingerprint_similarity


def binary_data():
    rng = np.random.default_rng(22)
    X = np.zeros((30, 12), dtype=np.uint8)
    X[:24, 8:] = rng.integers(0, 2, (24, 4))
    X[24:, :8] = 1
    X[24:, 8:] = [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0],
                  [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 1, 1]]
    return X, np.array([0] * 24 + [1] * 6), [f"id-{i:03}" for i in range(30)]


def test_cache_reuses_only_identical_training_content(tmp_path):
    X, y, ids = binary_data()
    a, b, first = resample(X, y, ids, "undersample", 7, tmp_path, "v1")
    c, d, second = resample(X, y, ids, "undersample", 7, tmp_path, "v1")
    assert not first["cache_hit"] and second["cache_hit"]
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(b, d)
    variants = [
        (X[:-1], y[:-1], ids[:-1], 7, "v1"),
        (X[::-1], y[::-1], ids[::-1], 7, "v1"),
        (X, y, ids, 8, "v1"),
        (X, y, ids, 7, "v2"),
        (X, y, ["changed"] + ids[1:], 7, "v1"),
        (1 - X, y, ids, 7, "v1"),
        (X, np.roll(y, 1), ids, 7, "v1"),
    ]
    keys = {first["cache_key"]}
    for features, labels, rowids, seed, namespace in variants:
        _, _, log = resample(features, labels, rowids, "undersample", seed, tmp_path, namespace)
        assert not log["cache_hit"]
        assert log["cache_key"] not in keys
        keys.add(log["cache_key"])


def test_smote_preserves_fractional_interpolation_and_inputs(tmp_path):
    X, y, ids = binary_data()
    original = X.copy()
    xr, yr, log = resample(X, y, ids, "smoteenn", 7, tmp_path, "toy")
    assert np.issubdtype(xr.dtype, np.floating)
    assert np.any((xr > 0) & (xr < 1))
    assert log["fractional_values"] > 0
    assert log["after_smote_expected"] == {"0": 24, "1": 24}
    assert log["before"] == {"0": 24, "1": 6}
    assert log["after"] == {str(i): int(np.sum(yr == i)) for i in [0, 1]}
    np.testing.assert_array_equal(X, original)


def test_smote_small_minority_adjusts_and_impossible_fold_fails(tmp_path):
    X, y, ids = binary_data()
    _, _, log = resample(X[:-3], y[:-3], ids[:-3], "smoteenn", 7, tmp_path, "small")
    assert log["adjustment"] == "SMOTE k_neighbors reduced from 5 to 2"
    with pytest.raises(ValueError, match="fewer than two"):
        resample(X[:-5], y[:-5], ids[:-5], "smoteenn", 7, tmp_path, "impossible")
    with pytest.raises(ValueError, match="only one class"):
        resample(X[:24], y[:24], ids[:24], "none", 7, tmp_path, "single")


def test_tuning_passes_only_current_fold_rows_to_resampling(tmp_path, monkeypatch):
    X, y, ids = binary_data()
    # Each fold has two minority and eight majority examples.
    frame = pd.DataFrame(dict(fp_index=np.arange(30), structure_id=ids,
                              label=y, cv_fold=np.arange(30) % 3))
    # Extra held-out feature rows must never enter tuning.
    extended_X = np.vstack([X, np.full((4, X.shape[1]), 99, dtype=np.uint8)])
    calls = []
    original = modeling.resample
    def spy(features, labels, row_ids, *args, **kwargs):
        calls.append((set(row_ids), labels.copy()))
        assert np.max(features) <= 1
        return original(features, labels, row_ids, *args, **kwargs)
    monkeypatch.setattr(modeling, "resample", spy)
    model, log = tune_fit(extended_X, frame, "logistic", "weights", {"logistic_C": [.1, 1]},
                          7, tmp_path, "isolation")
    assert len(calls) == 4
    for fold in range(3):
        expected = frame[frame.cv_fold != fold]
        assert calls[fold][0] == set(expected.structure_id)
        assert calls[fold][0].isdisjoint(frame[frame.cv_fold == fold].structure_id)
        np.testing.assert_array_equal(calls[fold][1], expected.label.to_numpy())
    assert calls[-1][0] == set(ids)
    assert model.class_weight == "balanced"
    assert log["class_weights"] == {"0": 30 / 48, "1": 30 / 12}
    assert log["cv_fit_count"] == 6 and log["final_fit_count"] == 1


def test_two_score_sigmoid_matches_smoothed_empirical_rates():
    scores = np.array([-2.] * 20 + [2.] * 20)
    y = np.array([1] * 3 + [0] * 17 + [1] * 15 + [0] * 5)
    calibrator = SigmoidCalibrator().fit(scores, y)
    # For two distinct scores a two-parameter increasing sigmoid can attain
    # the exact optimum: each group's mean Platt-smoothed target.
    positive_target = 19 / 20
    negative_target = 1 / 24
    expected = [(3 * positive_target + 17 * negative_target) / 20,
                (15 * positive_target + 5 * negative_target) / 20]
    np.testing.assert_allclose(calibrator.predict([-2., 2.]), expected, atol=1e-6)
    assert calibrator.slope_ > 0
    assert np.all(np.diff(calibrator.predict(np.linspace(-8, 8, 101))) > 0)
    assert calibrator.diagnostics_["calibration_n"] == 40
    assert calibrator.diagnostics_["calibration_positive"] == 18
    center = calibrator.center_
    calibrator.predict([1000., -1000.])
    assert calibrator.center_ == center  # Prediction must not refit scaling.


def test_reverse_signal_becomes_constant_and_single_class_calibration_fails():
    c = SigmoidCalibrator().fit([-2.] * 10 + [2.] * 10, [1] * 10 + [0] * 10)
    assert c.slope_ == pytest.approx(0, abs=1e-10)
    assert c.diagnostics_["constant_mapping"]
    np.testing.assert_allclose(c.predict([-100, 0, 100]), [.5, .5, .5], atol=1e-6)
    with pytest.raises(ValueError, match="both classes"):
        SigmoidCalibrator().fit([1., 2.], [0, 0])


def test_nearest_similarity_matches_manual_tanimoto():
    X = np.zeros((4, 2048), dtype=np.uint8)
    X[0, [0, 1]] = 1
    X[1, [2, 3]] = 1
    X[2, [0, 1, 2]] = 1
    X[3, [5]] = 1
    np.testing.assert_allclose(fingerprint_similarity(X, [0, 1], [2, 3]), [2 / 3, 0])
