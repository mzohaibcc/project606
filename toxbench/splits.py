"""Prespecified deterministic, method-independent partition and CV manifests."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split
from .data import ASSAYS, sha256_file, write_json

DEFAULT_SEEDS = [17, 42, 2026]
SPLIT_SETTINGS = {
    "version": 1,
    "fractions": [0.6, 0.2, 0.2],
    "scaffold_assignment": "sort groups by decreasing size times seeded uniform[0.5,1.5] priority; assign each to largest remaining absolute target capacity; ties partition order",
    "cv": "three folds, stratified shuffled for random; same label-independent group allocator with equal thirds for scaffold",
    "cv_seed_offset": 1009,
    "infeasible": "raise if any partition or any CV training/validation fold lacks a class; never retry seeds or move groups based on labels",
}


def assign_groups(groups, fractions, seed):
    """Allocate complete groups using sizes only. Empty scaffold is an ordinary group."""
    groups = np.asarray(groups, dtype=str)
    unique, counts = np.unique(groups, return_counts=True)
    rng = np.random.default_rng(seed)
    priorities = counts * rng.uniform(0.5, 1.5, len(unique))
    order = np.lexsort((unique, -priorities))
    targets = np.asarray(fractions) * len(groups)
    allocated = np.zeros(len(fractions), dtype=int)
    assignments = {}
    for position in order:
        selected = int(np.argmax(targets - allocated))
        assignments[unique[position]] = selected
        allocated[selected] += counts[position]
    return np.array([assignments[g] for g in groups], dtype=int)


def create_partition(table: pd.DataFrame, kind: str, seed: int) -> pd.DataFrame:
    table = table.sort_values("structure_id").reset_index(drop=True).copy()
    indices = np.arange(len(table))
    if kind == "random":
        train, holdout = train_test_split(indices, test_size=0.4, stratify=table.label, random_state=seed)
        calibration, test = train_test_split(holdout, test_size=0.5, stratify=table.label.iloc[holdout], random_state=seed + 1)
        table["partition"] = ""
        for name, subset in (("train", train), ("calibration", calibration), ("test", test)):
            table.loc[subset, "partition"] = name
    elif kind == "scaffold":
        assignment = assign_groups(table.scaffold, [0.6, 0.2, 0.2], seed)
        table["partition"] = np.array(["train", "calibration", "test"])[assignment]
    else:
        raise ValueError(f"Unknown partition kind {kind}")
    for name in ("train", "calibration", "test"):
        if table.loc[table.partition == name, "label"].nunique() != 2:
            raise ValueError(f"Infeasible {kind} seed {seed}: {name} lacks a class; split retained conceptually, no group relaxation or seed retries")
    table["cv_fold"] = -1
    train = np.flatnonzero(table.partition.to_numpy() == "train")
    if kind == "random":
        if table.iloc[train].label.value_counts().min() < 3:
            raise ValueError(f"Infeasible three-fold stratified CV for seed {seed}: fewer than 3 minority examples")
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + 1009)
        for fold, (_, validation) in enumerate(cv.split(train, table.label.iloc[train])):
            table.loc[train[validation], "cv_fold"] = fold
    else:
        table.loc[train, "cv_fold"] = assign_groups(table.scaffold.iloc[train], [1 / 3] * 3, seed + 1009)
    for fold in range(3):
        for validation in (True, False):
            mask = (table.partition == "train") & ((table.cv_fold == fold) if validation else (table.cv_fold != fold))
            if table.loc[mask, "label"].nunique() != 2:
                raise ValueError(f"Infeasible {kind} seed {seed}: CV fold {fold} {'validation' if validation else 'training'} lacks a class")
    return table


def check_partition(table: pd.DataFrame, kind: str):
    checks = {"unique_structures": bool(table.structure_id.is_unique),
              "all_rows_assigned": bool(table.partition.isin(["train", "calibration", "test"]).all()),
              "cv_train_only": bool((table.loc[table.partition != "train", "cv_fold"] == -1).all()),
              "all_training_rows_in_cv": bool(table.loc[table.partition == "train", "cv_fold"].isin([0, 1, 2]).all())}
    overlap = {}
    for left, right in [("train", "calibration"), ("train", "test"), ("calibration", "test")]:
        a, b = table[table.partition == left], table[table.partition == right]
        overlap[f"{left}_{right}_structures"] = len(set(a.structure_id) & set(b.structure_id))
        overlap[f"{left}_{right}_scaffolds"] = len(set(a.scaffold) & set(b.scaffold))
    checks["structure_disjoint"] = all(v == 0 for k, v in overlap.items() if k.endswith("structures"))
    if kind == "scaffold":
        checks["scaffold_disjoint"] = table.groupby("scaffold").partition.nunique().max() == 1
        checks["cv_scaffold_disjoint"] = table[table.partition == "train"].groupby("scaffold").cv_fold.nunique().max() == 1
        checks["empty_scaffold_unsplit"] = table.loc[table.scaffold == "__EMPTY__", "partition"].nunique() <= 1
    checks = {k: bool(v) for k, v in checks.items()}
    if not all(checks.values()):
        raise AssertionError(checks)
    return {"checks": checks, "overlap_counts": overlap}


def make_partitions(root: Path, seeds=DEFAULT_SEEDS) -> None:
    root = Path(root)
    out = root / "partitions"
    out.mkdir(parents=True, exist_ok=True)
    inputs = {"settings": SPLIT_SETTINGS, "seeds": list(seeds), "code_sha256": sha256_file(Path(__file__)),
              "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
              "tasks": {a: sha256_file(root / "data/processed" / f"{a}.csv") for a in ASSAYS}}
    cache_key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    state_path = out / "manifest.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["cache_key"] == cache_key and all((root / p).exists() and sha256_file(root / p) == h for p, h in state["outputs"].items()):
            return
    # Write the split design before materializing any partitions or evaluating models.
    write_json(out / "split_design.json", inputs)
    summary, verification, written = [], {}, []
    for assay in ASSAYS:
        table = pd.read_csv(root / "data/processed" / f"{assay}.csv")
        for kind in ("scaffold", "random"):
            for seed in seeds:
                key = f"{assay}/{kind}/{seed}"
                try:
                    result = create_partition(table, kind, seed)
                    verification[key] = check_partition(result, kind)
                    assert len(result) == len(table) and set(result.structure_id) == set(table.structure_id)
                except Exception as exc:
                    write_json(out / "infeasible.json", {"key": key, "reason": str(exc), "settings": SPLIT_SETTINGS})
                    raise
                path = out / assay / kind / f"{seed}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                result.to_csv(path, index=False)
                written.append(path)
                for partition, target in zip(("train", "calibration", "test"), (0.6, 0.2, 0.2)):
                    subset = result[result.partition == partition]
                    summary.append({"assay": assay, "split": kind, "seed": seed, "partition": partition, "cv_fold": -1,
                                    "n": len(subset), "positive": int(subset.label.sum()), "negative": int((subset.label == 0).sum()),
                                    "prevalence": float(subset.label.mean()), "scaffolds": subset.scaffold.nunique(), "fraction": len(subset) / len(table),
                                    "target_fraction": target, "deviation_percentage_points": 100 * (len(subset) / len(table) - target),
                                    "exploratory": int(subset.label.sum()) < 20})
                train = result[result.partition == "train"]
                for fold in range(3):
                    subset = train[train.cv_fold == fold]
                    summary.append({"assay": assay, "split": kind, "seed": seed, "partition": "cv_validation", "cv_fold": fold,
                                    "n": len(subset), "positive": int(subset.label.sum()), "negative": int((subset.label == 0).sum()),
                                    "prevalence": float(subset.label.mean()), "scaffolds": subset.scaffold.nunique(), "fraction": len(subset) / len(train),
                                    "target_fraction": 1 / 3, "deviation_percentage_points": 100 * (len(subset) / len(train) - 1 / 3),
                                    "exploratory": int(subset.label.sum()) < 20})
    summary_path = out / "partition_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    write_json(out / "verification.json", verification)
    written += [summary_path, out / "verification.json", out / "split_design.json"]
    write_json(state_path, {"cache_key": cache_key, "inputs": inputs, "outputs": {str(p.relative_to(root)): sha256_file(p) for p in written}})
