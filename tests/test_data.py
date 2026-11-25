import numpy as np
import pandas as pd
from toxbench.data import ASSAYS, clean_raw, standardize_smiles
from toxbench.splits import assign_groups, check_partition, create_partition


def test_standardization_salt_and_invalid():
    assert standardize_smiles("CC(=O)[O-].[Na+]")[0] == standardize_smiles("CC(=O)O")[0]
    assert standardize_smiles("CCO")[1] == "__EMPTY__"
    import pytest
    with pytest.raises(ValueError, match="no_organic_fragment"):
        standardize_smiles("[Na+].[Cl-]")
    with pytest.raises(ValueError, match="invalid_smiles"):
        standardize_smiles("not_a_smiles")


def test_missing_and_duplicate_resolution_are_assay_specific():
    raw = pd.DataFrame({"smiles": ["CCO", "OCC", "CCN", "NCC", "c1ccccc1", "bad"],
                        "NR-AR": [0, 0, 0, 1, np.nan, 1],
                        "NR-AhR": [1, np.nan, 1, 1, 0, 0],
                        "SR-MMP": [np.nan, 1, 0, 0, 1, 1]})
    structures, tasks, counts, exclusions, duplicates, valid = clean_raw(raw)
    assert len(structures) == 3 and len(valid) == 5
    assert len(tasks["NR-AR"]) == 1
    assert len(tasks["NR-AhR"]) == 3
    assert len(tasks["SR-MMP"]) == 3
    assert tasks["NR-AhR"].label.sum() == 2
    assert len(exclusions[exclusions.reason == "conflicting_duplicate_label"]) == 2
    assert set(duplicates[duplicates.action == "remove_conflict"].assay) == {"NR-AR"}
    for _, row in counts.iterrows():
        assert row.raw_observed == row.invalid_structure_observed_rows + row.conflicting_rows + row.consistent_duplicate_rows_merged + row.clean_structures


def toy_table():
    # Every scaffold contains both labels to guarantee group-only CV feasibility.
    return pd.DataFrame({"structure_id": [f"id{i:04}" for i in range(200)],
                         "scaffold": ["__EMPTY__" if i < 20 else f"g{i // 2}" for i in range(200)],
                         "label": [i % 2 for i in range(200)]})


def test_group_assignment_label_independent_and_reproducible():
    groups = toy_table().scaffold
    a = assign_groups(groups, [.6, .2, .2], 17)
    assert np.array_equal(a, assign_groups(groups, [.6, .2, .2], 17))
    for group in groups.unique():
        assert len(set(a[groups == group])) == 1
    assert set(a) == {0, 1, 2}


def test_partitions_cover_rows_and_cv_isolation():
    for kind in ("scaffold", "random"):
        t = create_partition(toy_table(), kind, 17)
        assert len(t) == 200
        assert all(check_partition(t, kind)["checks"].values())
        assert (t[t.partition != "train"].cv_fold == -1).all()
        assert t.equals(create_partition(toy_table(), kind, 17))
