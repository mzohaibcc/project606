"""Deterministic held-out metrics, paired bootstrap intervals, and figures.

Bootstrap intervals condition on the already fitted models and test partition.
They do not quantify model-refitting uncertainty and are not multiplicity adjusted.
"""
from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd

METRIC_NAMES = (
    "average_precision", "roc_auc", "brier", "log_loss",
    "precision_top10", "recall_top10",
)
METHODS = ("none", "weights", "undersample", "smoteenn")
CALIBRATIONS = ("raw", "sigmoid")


def _validate(y, p, ids):
    y = np.asarray(y)
    p = np.asarray(p, dtype=np.float64)
    ids = np.asarray(ids, dtype=str)
    if y.ndim != 1 or p.shape != y.shape or ids.shape != y.shape or not len(y):
        raise ValueError("Expected nonempty, aligned one-dimensional y, p, and ids")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError("Labels must be binary and probabilities finite")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must lie in [0, 1]")
    return y.astype(np.int64), p, ids


def top_indices(p, ids, fraction=0.1):
    """Descending probability, ascending structure ID, then occurrence index."""
    p, ids = np.asarray(p), np.asarray(ids, dtype=str)
    k = math.ceil(fraction * len(p))
    return np.lexsort((np.arange(len(p)), ids, -p))[:k]


def _prepared(y, p, ids):
    order = np.lexsort((np.arange(len(p)), ids, -p))
    pp = p[order]
    # Tie groups matter for AP and AUC; selection still uses the ID tie rule.
    ends = np.r_[np.flatnonzero(pp[1:] != pp[:-1]), len(pp) - 1]
    eps = np.finfo(np.float64).eps
    clipped = np.clip(p, eps, 1 - eps)
    losses = -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
    return order, ends, (p - y) ** 2, losses


def _weighted_metrics(y, prepared, counts):
    """Exact metrics for an explicitly replicated sample, without replication."""
    order, ends, squared, losses = prepared
    n = int(counts.sum())
    pos = int(np.dot(counts, y))
    neg = n - pos
    if n == 0:
        return {key: np.nan for key in METRIC_NAMES}
    w, yy = counts[order], y[order]
    cumulative_n = np.cumsum(w)
    cumulative_pos = np.cumsum(w * yy)
    tie_n, tie_pos = cumulative_n[ends], cumulative_pos[ends]
    positive_increment = np.diff(np.r_[0, tie_pos])
    negative_increment = np.diff(np.r_[0, tie_n - tie_pos])
    valid = tie_n > 0
    # Single-class AP and ROC-AUC are intentionally undefined for this benchmark.
    ap = (np.sum(positive_increment[valid] * tie_pos[valid] / tie_n[valid]) / pos
          if pos and neg else np.nan)
    positive_before = np.r_[0, tie_pos[:-1]]
    auc = (np.sum(negative_increment * (positive_before + .5 * positive_increment))
           / (pos * neg) if pos and neg else np.nan)
    k = math.ceil(.1 * n)
    before = np.r_[0, cumulative_n[:-1]]
    selected_weights = np.minimum(w, np.maximum(0, k - before))
    selected_pos = int(np.dot(selected_weights, yy))
    return {
        "average_precision": float(ap), "roc_auc": float(auc),
        "brier": float(np.dot(counts, squared) / n),
        "log_loss": float(np.dot(counts, losses) / n),
        "precision_top10": float(selected_pos / k),
        "recall_top10": float(selected_pos / pos) if pos else np.nan,
    }


def metrics(y, p, ids):
    """Metrics at observed prevalence with ceil(10% n) deterministic selection."""
    y, p, ids = _validate(y, p, ids)
    result = _weighted_metrics(y, _prepared(y, p, ids), np.ones(len(y), dtype=int))
    result.update(n=int(len(y)), n_positive=int(y.sum()), prevalence=float(y.mean()),
                  selection_count=math.ceil(.1 * len(y)), exploratory=bool(y.sum() < 20))
    return result


def reliability_table(y, p, n_bins=10):
    """Equal-width reliability bins, including empty bins and both endpoints."""
    y, p, _ = _validate(y, p, np.arange(len(y)))
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.minimum(np.searchsorted(edges, p, side="right") - 1, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        count = int(mask.sum())
        rows.append(dict(bin=b, lower=edges[b], upper=edges[b + 1], count=count,
                         positives=int(y[mask].sum()),
                         exploratory=bool(y[mask].sum() < 20),
                         mean_prediction=float(p[mask].mean()) if count else np.nan,
                         observed_fraction=float(y[mask].mean()) if count else np.nan))
    return pd.DataFrame(rows)


def bootstrap_comparisons(predictions, split_kind, n_bootstrap=1000, seed=1729):
    """Paired percentile CIs: method minus corresponding unbalanced baseline.

    Draw examples for random tests or sample G scaffold groups with replacement
    for scaffold tests. Every comparison shares exactly the same sampled counts.
    Empty scaffold values are one group. Number of examples can vary by draw for
    scaffold tests. Undefined metric draws are counted and omitted per metric.
    """
    if split_kind not in ("random", "scaffold") or n_bootstrap < 1:
        raise ValueError("split_kind must be random/scaffold and n_bootstrap positive")
    if predictions.structure_id.duplicated().any():
        raise ValueError("Prediction input must have one row per structure")
    y = predictions.label.to_numpy(dtype=int)
    ids = predictions.structure_id.to_numpy(dtype=str)
    cols = [f"{m}_{c}" for m in METHODS for c in CALIBRATIONS if f"{m}_{c}" in predictions]
    preparations = {}
    for col in cols:
        _, p, _ = _validate(y, predictions[col], ids)
        preparations[col] = _prepared(y, p, ids)
    comparisons = [(m, c) for m in METHODS[1:] for c in CALIBRATIONS
                   if f"{m}_{c}" in cols and f"none_{c}" in cols]
    ones = np.ones(len(y), dtype=np.int64)
    original = {col: _weighted_metrics(y, prep, ones) for col, prep in preparations.items()}
    values = {(m, c, metric): [] for m, c in comparisons for metric in METRIC_NAMES}
    rng = np.random.default_rng(seed)
    if split_kind == "scaffold":
        groups = predictions.scaffold.fillna("__EMPTY__").replace("", "__EMPTY__")
        group_index, group_names = pd.factorize(groups, sort=True)
        n_groups = len(group_names)
    else:
        n_groups = len(y)
    for _ in range(n_bootstrap):
        if split_kind == "scaffold":
            cluster_counts = np.bincount(rng.integers(n_groups, size=n_groups), minlength=n_groups)
            counts = cluster_counts[group_index]
        else:
            counts = np.bincount(rng.integers(len(y), size=len(y)), minlength=len(y))
        boot = {col: _weighted_metrics(y, prep, counts) for col, prep in preparations.items()}
        for method, calibration in comparisons:
            for metric in METRIC_NAMES:
                values[(method, calibration, metric)].append(
                    boot[f"{method}_{calibration}"][metric] - boot[f"none_{calibration}"][metric])
    rows = []
    for (method, calibration, metric), draws in values.items():
        draws = np.asarray(draws)
        valid = draws[np.isfinite(draws)]
        lo, hi = np.quantile(valid, [.025, .975]) if len(valid) else (np.nan, np.nan)
        rows.append(dict(method=method, calibration=calibration, metric=metric,
                         difference=original[f"{method}_{calibration}"][metric] - original[f"none_{calibration}"][metric],
                         ci_lower=float(lo), ci_upper=float(hi), valid_count=len(valid),
                         invalid_count=n_bootstrap - len(valid), n_bootstrap=n_bootstrap,
                         seed=seed, bootstrap_unit="scaffold" if split_kind == "scaffold" else "example",
                         n_bootstrap_units=n_groups, original_n=len(y), original_n_positive=int(y.sum()),
                         exploratory=bool(y.sum() < 20)))
    return pd.DataFrame(rows)


def create_figures(predictions, output_dir, title):
    """Write publication-resolution PNG/PDF PR and reliability count panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = dict(zip(METHODS, ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]))
    labels = dict(none="Unbalanced", weights="Class weights", undersample="Undersampling", smoteenn="SMOTEENN")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    y = predictions.label.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), constrained_layout=True)
    for ax, calibration in zip(axes, CALIBRATIONS):
        for method in METHODS:
            col = f"{method}_{calibration}"
            if col not in predictions:
                continue
            p = predictions[col].to_numpy()
            precision, recall, _ = precision_recall_curve(y, p)
            ap = metrics(y, p, predictions.structure_id)["average_precision"]
            ax.step(recall, precision, where="post", color=colors[method], lw=1.6,
                    label=f"{labels[method]} (AP {ap:.3f})")
        ax.axhline(y.mean(), color="#666666", ls="--", lw=1, label=f"Prevalence {y.mean():.3f}")
        ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02),
               title="Before calibration" if calibration == "raw" else "After sigmoid calibration")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(title)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"precision_recall.{extension}", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), constrained_layout=True,
                             gridspec_kw={"height_ratios": [2.5, 1]})
    reliability = []
    for j, calibration in enumerate(CALIBRATIONS):
        ax, count_ax = axes[0, j], axes[1, j]
        ax.plot([0, 1], [0, 1], color="#777777", ls="--", lw=1)
        for method in METHODS:
            col = f"{method}_{calibration}"
            if col not in predictions:
                continue
            bins = reliability_table(y, predictions[col])
            bins["method"], bins["calibration"] = method, calibration
            reliability.append(bins)
            present = bins["count"] > 0
            ax.plot(bins.loc[present, "mean_prediction"], bins.loc[present, "observed_fraction"],
                    marker="o", ms=3, lw=1.4, color=colors[method], label=labels[method])
            count_ax.step(np.r_[bins.lower.to_numpy(), 1.0], np.r_[bins["count"].to_numpy(), bins["count"].iloc[-1]],
                          where="post", color=colors[method], lw=1.4)
        ax.set(xlabel="Mean predicted probability", ylabel="Observed active fraction", xlim=(0, 1), ylim=(0, 1),
               title="Before calibration" if calibration == "raw" else "After sigmoid calibration")
        ax.legend(fontsize=8, loc="upper left")
        count_ax.set(xlabel="Predicted probability bin", ylabel="Bin count", xlim=(0, 1))
        count_ax.set_yscale("symlog", linthresh=1)
        count_ax.set_ylim(bottom=0)
    fig.suptitle(title + "\nReliability in ten fixed equal-width bins")
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"reliability.{extension}", dpi=300)
    plt.close(fig)
    pd.concat(reliability, ignore_index=True).to_csv(output_dir / "reliability.csv", index=False)

