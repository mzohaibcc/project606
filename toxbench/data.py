"""Deterministic, label-independent structural preprocessing for public Tox21.

Structures are used only as retrospective assay-benchmark features and grouping keys.
No synthetic feature vector is interpreted as a molecule.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

ASSAYS = ["NR-AR", "NR-AhR", "SR-MMP"]
SOURCE_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
LOADER_URL = "https://github.com/deepchem/deepchem/blob/master/deepchem/molnet/load_function/tox21_datasets.py"
SETTINGS = {
    "version": 1,
    "standardization": "RDKit Cleanup; largest carbon-containing fragment by heavy atom count, atom count, lexicographic isomeric SMILES; Uncharger; sanitize",
    "tautomer_canonicalization": False,
    "identity_stereochemistry": True,
    "scaffold_stereochemistry": False,
    "fingerprint_radius": 2,
    "fingerprint_bits": 2048,
    "fingerprint_chirality": False,
    "empty_scaffold": "__EMPTY__",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def standardize_smiles(smiles: str) -> tuple[str, str]:
    """Return canonical isomeric SMILES and non-stereochemical Murcko scaffold.

    Organic means contains at least one carbon atom. Ties in main-fragment
    selection are resolved by ascending canonical isomeric SMILES.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("missing_or_empty_smiles")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("invalid_smiles")
    mol = rdMolStandardize.Cleanup(mol)
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    organic = [m for m in fragments if any(a.GetAtomicNum() == 6 for a in m.GetAtoms())]
    if not organic:
        raise ValueError("no_organic_fragment")
    organic.sort(key=lambda m: (-m.GetNumHeavyAtoms(), -m.GetNumAtoms(), Chem.MolToSmiles(m, isomericSmiles=True)))
    mol = rdMolStandardize.Uncharger().uncharge(organic[0])
    Chem.SanitizeMol(mol)
    smiles_out = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return smiles_out, scaffold or SETTINGS["empty_scaffold"]


def clean_raw(raw: pd.DataFrame):
    """Pure audit transformation. Returns global structures, assay tables and logs."""
    raw = raw.reset_index(drop=True)
    required = {"smiles", *ASSAYS}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing required columns: {required - set(raw.columns)}")
    valid_rows, exclusions = [], []
    for row_id, row in raw.reset_index(drop=True).iterrows():
        try:
            canonical, scaffold = standardize_smiles(row["smiles"])
        except (ValueError, Chem.rdchem.MolSanitizeException) as exc:
            exclusions.append({"source_row": row_id, "assay": "ALL", "reason": str(exc), "raw_smiles": row["smiles"], "structure_id": ""})
            continue
        identifier = hashlib.sha256(canonical.encode()).hexdigest()
        valid_rows.append({"source_row": row_id, "raw_smiles": row["smiles"], "canonical_smiles": canonical,
                           "structure_id": identifier, "scaffold": scaffold, **{a: row[a] for a in ASSAYS}})
    valid = pd.DataFrame(valid_rows)
    if valid.empty:
        raise ValueError("No valid organic structures remain")
    structures = valid[["structure_id", "canonical_smiles", "scaffold"]].drop_duplicates("structure_id").sort_values("structure_id").reset_index(drop=True)
    structures["fp_index"] = np.arange(len(structures))
    fp_lookup = structures.set_index("structure_id")["fp_index"].to_dict()
    tasks, summary, duplicates = {}, [], []
    for assay in ASSAYS:
        vals = pd.to_numeric(raw[assay], errors="raise")
        observed_values = set(vals.dropna().unique())
        if not observed_values.issubset({0, 1}):
            raise ValueError(f"Unexpected nonbinary labels in {assay}: {observed_values}")
        # Record missing assay labels even when a row also has an invalid structure.
        for row_id in raw.index[raw[assay].isna()]:
            exclusions.append({"source_row": int(row_id), "assay": assay, "reason": "missing_label", "raw_smiles": raw.loc[row_id, "smiles"], "structure_id": ""})
        observed = valid.loc[valid[assay].notna()].copy()
        records = []
        conflicting_structures = 0
        conflicting_rows = 0
        merged_rows = 0
        for sid, group in observed.groupby("structure_id", sort=True):
            source_rows = group.source_row.astype(int).tolist()
            labels = group[assay].astype(int).unique()
            if len(labels) > 1:
                conflicting_structures += 1
                conflicting_rows += len(group)
                for _, row in group.iterrows():
                    exclusions.append({"source_row": int(row.source_row), "assay": assay, "reason": "conflicting_duplicate_label", "raw_smiles": row.raw_smiles, "structure_id": sid})
                duplicates.append({"assay": assay, "structure_id": sid, "source_rows": json.dumps(source_rows), "labels": json.dumps(group[assay].astype(int).tolist()), "action": "remove_conflict"})
                continue
            row = group.iloc[0]
            records.append({"structure_id": sid, "canonical_smiles": row.canonical_smiles, "scaffold": row.scaffold,
                            "label": int(labels[0]), "fp_index": int(fp_lookup[sid]), "source_rows": json.dumps(source_rows), "source_record_count": len(group)})
            if len(group) > 1:
                merged_rows += len(group) - 1
                duplicates.append({"assay": assay, "structure_id": sid, "source_rows": json.dumps(source_rows), "labels": json.dumps(group[assay].astype(int).tolist()), "action": "merge_consistent"})
        table = pd.DataFrame(records)
        tasks[assay] = table
        summary.append({"assay": assay, "raw_records": len(raw), "raw_observed": int(vals.notna().sum()), "raw_positives": int((vals == 1).sum()),
                        "raw_negatives": int((vals == 0).sum()), "raw_missing": int(vals.isna().sum()),
                        "invalid_structure_observed_rows": int(vals.notna().sum() - len(observed)),
                        "valid_structure_observed_rows": len(observed), "conflicting_structures": conflicting_structures,
                        "conflicting_rows": conflicting_rows, "consistent_duplicate_rows_merged": merged_rows,
                        "clean_structures": len(table), "clean_positives": int(table.label.sum()),
                        "clean_negatives": int((table.label == 0).sum()), "clean_prevalence": float(table.label.mean())})
    exclusion_columns = ["source_row", "assay", "reason", "raw_smiles", "structure_id"]
    duplicate_columns = ["assay", "structure_id", "source_rows", "labels", "action"]
    return structures, tasks, pd.DataFrame(summary), pd.DataFrame(exclusions, columns=exclusion_columns), pd.DataFrame(duplicates, columns=duplicate_columns), valid


def prepare(root: Path) -> None:
    root = Path(root)
    raw_dir, proc, audit = root / "data/raw", root / "data/processed", root / "data/audit"
    for directory in (raw_dir, proc, audit):
        directory.mkdir(parents=True, exist_ok=True)
    raw_path, source_path = raw_dir / "tox21.csv.gz", raw_dir / "source.json"
    if not raw_path.exists():
        temporary = raw_path.with_suffix(".download")
        with urlopen(SOURCE_URL, timeout=120) as response, temporary.open("wb") as out:
            out.write(response.read())
        os.replace(temporary, raw_path)
        write_json(source_path, {"url": SOURCE_URL, "deepchem_loader_url": LOADER_URL,
                                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                                "sha256": sha256_file(raw_path), "bytes": raw_path.stat().st_size})
    digest = sha256_file(raw_path)
    if source_path.exists():
        provenance = json.loads(source_path.read_text())
        if provenance["sha256"] != digest:
            raise ValueError("Raw file checksum differs from recorded source.json; refusing changed input")
    else:
        # An explicitly prepopulated raw file can be used offline, but not claimed downloaded.
        provenance = {"url": SOURCE_URL, "deepchem_loader_url": LOADER_URL, "downloaded_at_utc": None,
                      "provenance_note": "File pre-existed; download date unverified", "sha256": digest, "bytes": raw_path.stat().st_size}
        write_json(source_path, provenance)
    key_inputs = {"raw_sha256": digest, "code_sha256": sha256_file(Path(__file__)), "settings": SETTINGS,
                  "rdkit_version": rdBase.rdkitVersion, "numpy_version": np.__version__, "pandas_version": pd.__version__}
    cache_key = hashlib.sha256(json.dumps(key_inputs, sort_keys=True).encode()).hexdigest()
    manifest_path = proc / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["cache_key"] == cache_key and all((root / p).exists() and sha256_file(root / p) == h for p, h in old["outputs"].items()):
            return
    raw = pd.read_csv(raw_path)
    structures, tasks, summary, exclusions, duplicates, valid = clean_raw(raw)
    provenance.update({"raw_records": len(raw), "columns": raw.columns.tolist(), "assay_label_columns": [c for c in raw if c.startswith(("NR-", "SR-"))]})
    write_json(source_path, provenance)
    structures.to_csv(proc / "structures.csv", index=False)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    fingerprints = np.stack([generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in structures.canonical_smiles]).astype(np.uint8)
    np.save(proc / "fingerprints.npy", fingerprints, allow_pickle=False)
    for assay, table in tasks.items():
        table.to_csv(proc / f"{assay}.csv", index=False)
    summary.to_csv(audit / "assay_counts.csv", index=False)
    exclusions.to_csv(audit / "exclusions.csv", index=False)
    duplicates.to_csv(audit / "duplicate_groups.csv", index=False)
    valid.to_csv(audit / "standardized_source_rows.csv", index=False)
    write_json(audit / "structure_counts.json", {"raw_records": len(raw), "valid_structure_rows": len(valid),
               "invalid_or_nonorganic_structure_rows": len(raw) - len(valid), "unique_standardized_structures": len(structures),
               "global_duplicate_rows": len(valid) - len(structures), "empty_scaffold_unique_structures": int((structures.scaffold == "__EMPTY__").sum()),
               "exclusion_log_note": "A row may have both a global structure exclusion and assay-specific missing-label entries; do not sum overlapping reasons."})
    paths = [p for d in (proc, audit) for p in d.iterdir() if p.is_file() and p != manifest_path]
    write_json(manifest_path, {"cache_key": cache_key, "inputs": key_inputs,
               "outputs": {str(p.relative_to(root)): sha256_file(p) for p in paths}})
