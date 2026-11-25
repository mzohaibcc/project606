import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss

from toxbench.evaluation import (metrics, top_indices, reliability_table,
                                bootstrap_comparisons, _prepared, _weighted_metrics)


def test_metrics_match_sklearn_including_ties():
    rng = np.random.default_rng(68)
    y = rng.integers(2, size=109)
    p = np.round(rng.random(109), 1)
    ids = [f"id{i:04}" for i in range(109)]
    got = metrics(y, p, ids)
    assert got["average_precision"] == pytest.approx(average_precision_score(y, p))
    assert got["roc_auc"] == pytest.approx(roc_auc_score(y, p))
    assert got["brier"] == pytest.approx(brier_score_loss(y, p))
    assert got["log_loss"] == pytest.approx(log_loss(y, p))
    assert got["selection_count"] == 11


def test_tie_break_and_ceiling():
    p = np.ones(11) * .4
    ids = np.array(["z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j"])
    assert top_indices(p, ids).tolist() == [1, 2]
    y = np.zeros(11, dtype=int)
    y[1] = 1
    assert metrics(y, p, ids)["precision_top10"] == .5
    assert metrics(y, p, ids)["recall_top10"] == 1


def test_weighted_bootstrap_equals_explicit_replication():
    rng = np.random.default_rng(74)
    y = rng.integers(2, size=100)
    p = np.round(rng.random(100), 1)
    ids = np.array([str(i) for i in range(100)])
    counts = rng.integers(4, size=100)
    indices = np.repeat(np.arange(100), counts)
    actual = _weighted_metrics(y, _prepared(y, p, ids), counts)
    expected = metrics(y[indices], p[indices], ids[indices])
    for name in actual:
        assert actual[name] == pytest.approx(expected[name])


def test_reliability_endpoints_and_empty_bins():
    table = reliability_table([0, 1, 1], [0, .1, 1])
    assert len(table) == 10
    assert table["count"].sum() == 3
    assert table["count"].iloc[0] == 1
    assert table["count"].iloc[1] == 1
    assert table["count"].iloc[-1] == 1
    assert np.isnan(table.mean_prediction.iloc[2])


@pytest.mark.parametrize("split_kind", ["random", "scaffold"])
def test_identical_models_have_zero_paired_intervals(split_kind):
    data = pd.DataFrame(dict(structure_id=list("abcdefgh"), label=[0, 1, 0, 1, 1, 0, 0, 1],
                             scaffold=["A", "A", "B", "B", "C", "C", "", ""]))
    for c in ["raw", "sigmoid"]:
        for m in ["none", "weights", "undersample", "smoteenn"]:
            data[f"{m}_{c}"] = [.1, .8, .3, .7, .9, .1, .2, .4]
    result = bootstrap_comparisons(data, split_kind, n_bootstrap=80, seed=7)
    assert len(result) == 36
    assert (result.difference == 0).all()
    assert (result.ci_lower == 0).all()
    assert (result.ci_upper == 0).all()
    assert (result.valid_count + result.invalid_count == 80).all()
    assert (result.n_bootstrap_units == (4 if split_kind == "scaffold" else 8)).all()
    pd.testing.assert_frame_equal(result, bootstrap_comparisons(data, split_kind, n_bootstrap=80, seed=7))


def test_single_class_explicit_undefined_ranking():
    got = metrics([0, 0], [.1, .2], ["a", "b"])
    assert np.isnan(got["average_precision"])
    assert np.isnan(got["roc_auc"])
    assert np.isnan(got["recall_top10"])
    assert got["exploratory"]
    assert got["brier"] == pytest.approx(.025)


def test_invalid_predictions_rejected():
    with pytest.raises(ValueError):
        metrics([0, 1], [.1, np.nan], ["a", "b"])
    with pytest.raises(ValueError):
        metrics([0, 1], [-.1, .9], ["a", "b"])


def test_cluster_intervals_match_manual_whole_group_replication():
    data = pd.DataFrame(dict(structure_id=list("abcdefg"), label=[0, 1, 1, 0, 0, 1, 0],
                             scaffold=["A", "A", "A", "B", "B", "C", "C"],
                             none_raw=[.1, .8, .6, .3, .2, .9, .4],
                             weights_raw=[.2, .7, .8, .5, .1, .7, .3]))
    output = bootstrap_comparisons(data, "scaffold", n_bootstrap=41, seed=993)
    rng = np.random.default_rng(993)
    groups = [np.array([0, 1, 2]), np.array([3, 4]), np.array([5, 6])]
    deltas = []
    for _ in range(41):
        ix = np.concatenate([groups[g] for g in rng.integers(3, size=3)])
        sample = data.iloc[ix]
        deltas.append(np.mean((sample.label - sample.weights_raw)**2)
                      - np.mean((sample.label - sample.none_raw)**2))
    row = output.loc[output.metric == "brier"].iloc[0]
    assert row.ci_lower == pytest.approx(np.quantile(deltas, .025))
    assert row.ci_upper == pytest.approx(np.quantile(deltas, .975))
