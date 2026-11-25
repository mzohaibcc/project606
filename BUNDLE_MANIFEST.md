# Submission bundle manifest — `project606_submission.zip`

**Project:** Class Balancing and Calibration for Toxicity Prediction on New Chemical Scaffolds (CSC606)
**Built:** 25 November 2026
**Archive:** `project606_submission.zip` (64.97 MB compressed, 130.35 MB uncompressed, 1,099 files, single top-level directory `project606/`)

Everything in the archive sits under one root folder, `project606/`, and keeps the paths used
throughout `docs/report.md`, so every file reference in the report resolves inside the extracted tree.

---

## 1. What is included

| Path in archive | Files | Uncompressed | What it is |
| --- | ---: | ---: | --- |
| `project606/toxbench/` | 7 | 0.05 MB | the pipeline package (`data`, `splits`, `modeling`, `evaluation`, `postprocess`, `runner`) |
| `project606/tests/` | 3 | 0.01 MB | the pytest suite (20 tests, all passing at build time) |
| `project606/configs/` | 2 | < 0.01 MB | `full.json` (the reported run) and `smoke.json` |
| `project606/docs/` | 8 | 0.16 MB | `report.md` (the final report), `proposal.txt`, `data_methods.md`, `implementation_plan.md`, `references.md` / `.json`, and the two independent audits `data_verification.json` (37 checks) and `results_verification.json` (63 checks) |
| `project606/notebook/` | 1 | 0.09 MB | `toxbench_colab.ipynb`, 49 cells, committed **unexecuted** with no stored outputs |
| `project606/partitions/` | 22 | 16.34 MB | the 18 partition manifests (`<assay>/<split>/<seed>.csv`) plus `partition_summary.csv`, `split_design.json`, `manifest.json`, `verification.json` |
| `project606/data/processed/` | 6 | 18.9 MB | cleaned per-assay tables, `structures.csv`, `fingerprints.npy`, `manifest.json` |
| `project606/data/audit/` | 5 | 1.66 MB | `assay_counts.csv`, `exclusions.csv`, `duplicate_groups.csv`, `standardized_source_rows.csv`, `structure_counts.json` |
| `project606/data/raw/source.json` | 1 | < 0.01 MB | provenance of the raw input: URL, SHA-256, byte count, download timestamp |
| `project606/data/raw/tox21.csv.gz` | 1 | 0.12 MB | the raw DeepChem Tox21 input — **included**, see §3 |
| `project606/requirements.txt`, `requirements-lock.txt` | 2 | < 0.01 MB | declared and fully pinned dependencies |
| `project606/.gitignore` | 1 | < 0.01 MB | the repository ignore rules |
| `project606/outputs/full/` top level | 10 | 0.62 MB | `metrics.csv` (432 rows), `summary.csv` (144), `bootstrap_differences.csv` (1,296), `similarity_metrics.csv` (1,152), `alignment_similarity_audit.csv` (36), plus `config.json`, `environment.json`, `execution_status.json`, `evaluation_status.json`, `invocations.jsonl` |
| `project606/outputs/full/evaluation/` | 288 | 40.58 MB | all 36 evaluation groups: `predictions_wide.csv`, `bootstrap.csv`, `bootstrap_key.txt`, `reliability.csv`, `precision_recall.{png,pdf}`, `reliability.{png,pdf}` |
| `project606/outputs/full/runs/` | 738 | 51.80 MB | for each of the 144 runs: `predictions.csv`, `calibration_predictions.csv`, `metrics.csv`, `training.json`, `complete.json`; plus the 18 per-partition `similarity.npz` files |
| `project606/outputs/*.log` | 3 | 0.01 MB | `preparation.log`, `full.log`, `evaluate.log` — the console logs of the three pipeline stages that produced the shipped artefacts |
| `project606/BUNDLE_MANIFEST.md` | 1 | < 0.01 MB | a copy of this file, so the archive describes itself |

---

## 2. What is excluded, and why

| Excluded | Size | Why |
| --- | ---: | --- |
| `.venv/` | 394 MB | installed third-party packages. Reconstructible exactly from the bundled `requirements-lock.txt` (fully pinned); shipping a platform-specific virtualenv adds no information and would not work on another machine anyway. |
| All `__pycache__/` directories and `*.pyc` | small | compiled bytecode caches, regenerated automatically by the interpreter. Also ignored by `.gitignore`. |
| `.pytest_cache/` | small | pytest's own run cache; regenerated on the next test run. Ignored by `.gitignore`. |
| `tmp/` | small | scratch working directory, no reported result depends on it. Ignored by `.gitignore`. |
| `outputs/notebook_demo/` | 14 MB | output of the Colab notebook's own small demonstration run. Regenerated from scratch every time the notebook is executed, and no number in `docs/report.md` comes from it. |
| `outputs/smoke/` and `outputs/smoke.log` | 11 MB | the early smoke-test run, entirely superseded by the full run in `outputs/full/`. `configs/smoke.json` is included so it can be re-run if anyone wants it. |
| `outputs/full/cache/` | 50 MB | cached resampled training folds, keyed by content hash. Pure derived data, regenerable by re-running the pipeline, and no reported metric reads from it. Ignored by `.gitignore`. |
| `outputs/full/runs/**/model.joblib` (144 files) | 122 MB | the fitted model binaries — see the note below. |

### Note on the 144 `model.joblib` files

**The committed `predictions.csv` and `training.json` fully determine every reported result.**
Each run directory keeps the model's test-set scores (`predictions.csv`), the calibration-partition
scores (`calibration_predictions.csv`) and the per-run metrics (`metrics.csv`). `training.json` holds
the training record — the selected hyperparameters and their CV fold scores, every grid trial, the
resampler logs, the CV and final fit counts, the class weights, the fitted sigmoid parameters
(slope, intercept, centre, scale, calibration-partition size and positives) and the
calibration/training/test content-hash identifiers. `complete.json` holds the run's namespace,
partition sizes, training prevalence, wall-clock seconds, fit counts and completion timestamp.
Every table in `docs/report.md` is computed from these files, and the independent results audit
(`docs/results_verification.json`) re-scored all 144 models from exactly these stored predictions
against scikit-learn 1.9.1, with a largest deviation of 3.33e-16 over 2,592 metric comparisons.

The models themselves are regenerable by re-running the pipeline from the bundled data, partitions
and pinned requirements — `python -m toxbench.runner run --config configs/full.json --output
outputs/full` (both flags are defaults in `toxbench/runner.py`, so bare `run` is equivalent).
**Excluding the binaries therefore costs archive size, not verifiability:** every number in the
report can be re-derived from what is in this bundle without them.

---

## 3. Decision on `data/raw/tox21.csv.gz`

**Decision: included.**

Reasoning:

- It is only **122,925 bytes (0.12 MB)** — 0.09% of the uncompressed bundle — so including it costs
  essentially nothing against the 65 MB archive.
- Including it makes the bundle **self-contained and reproducible offline**: the whole pipeline, from
  raw CSV through cleaning, fingerprints, partitions, training and evaluation, can be re-run from the
  archive alone, with no network access and no dependency on an external S3 bucket staying up.
- Its integrity is verifiable rather than assumed. `data/raw/source.json` records the source URL
  (`https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz`), the DeepChem loader it
  comes from, the byte count and the SHA-256
  `45d09792492ce049039dd24aa27b07fc79ce20c573187d4d90bcd178c0c0d360`. **That hash and byte count were
  re-verified against the bundled file at build time and matched.**
- It is the public DeepChem-hosted Tox21 input, downloaded from the URL above; nothing in it is
  private to this project.

The alternative — ship only `source.json` and let a reader re-download — would have saved 0.12 MB and
introduced a dependency on an external URL, which is the wrong trade for a submission that is meant to
be checkable years from now.

---

## 4. Items the brief did not cover, and what was done with them

These were not on either the include or the exclude list, so the decision is recorded here explicitly:

- **`outputs/full/invocations.jsonl` — included.** It is a JSON-lines run-accounting record for
  `outputs/full/` (86 bytes), of a piece with the other top-level status files.
- **`outputs/preparation.log`, `outputs/full.log`, `outputs/evaluate.log` — included** (8 KB total).
  They are the console provenance of the three stages that produced the shipped artefacts.
  **`outputs/smoke.log` — excluded**, together with the rest of the superseded smoke run.
- **`scripts/` — nothing to include.** The directory exists in the working tree but is empty, so it
  contributes no files; empty directories are not stored in the archive.
- **`BUNDLE_MANIFEST.md` — included** inside the archive as well as at the project root, so the
  bundle documents itself after extraction.
- **`project606_submission.zip` itself — excluded** from its own contents (`*.zip` is an excluded
  suffix in the build allowlist, and is also in `.gitignore`).

---

## 5. Verification performed on the archive

- Built from an explicit **allowlist** with Python's `zipfile` (deflate, level 9), which stores no
  `__MACOSX/` or `._*` resource-fork entries.
- Extracted into a fresh empty directory outside the project; `unzip -t` reported no errors.
- Every extracted file's SHA-256 was compared against the source file in the project — all 1,099 matched.
- Expected counts confirmed in the extracted tree: 144 × 5 per-run files (720), 18 `similarity.npz`,
  36 × 8 evaluation files (288).
- Leak scan over the extracted tree for every excluded pattern (`.venv`, `__pycache__`,
  `.pytest_cache`, `tmp/`, `notebook_demo`, `smoke`, `outputs/full/cache`, `model.joblib`, `*.pyc`,
  `.DS_Store`, `*.zip`, `__MACOSX`, `._*`) found **zero** matches.
- Every file path cited in `docs/report.md` was confirmed present in the extracted tree.
- The bundled notebook was confirmed to contain **no stored outputs** and no execution counts
  (49 cells, 32 of them code).
- The bundled `docs/report.md` was independently cross-checked against the bundled source CSVs and
  evaluation files: **2,644 table cells, 4,309 numeric/text tokens, 0 mismatches**, covering all 33
  captioned table blocks (Tables 2.1–6.5). A further 66 prose and headline claims were recomputed
  from the same sources and **all 66 were confirmed**.

### Report revision carried by this archive

**This bundle carries the corrected `docs/report.md` (1,020 lines, 650 table rows, 101,561 bytes).**

An earlier cross-check found four prose figures that had been derived from already-rounded display
values rather than from full precision. All four were corrected by the report's author before this
archive was built, and each correction was re-verified against the committed CSVs at build time:

| § | Now reads | Recomputed from the committed data |
| --- | --- | --- |
| 5.1 (line 802) | ratio **4.16** → 2.31 | 0.661749 / 0.158901 = **4.1645**; now agrees with Table 5.1 |
| 5.2 | the four deltas span **+0.0463 to +0.0618** | the four favourable forest/weights/scaffold AP deltas are +0.0463, +0.0496, +0.0505, +0.0618 |
| 8 | variability **more than doubled (0.0379 → 0.0921, a factor of 2.43)** | mean AP SD 0.037876 → 0.092150, ratio **2.4330** |
| 2.3 | **41.9–44.7%** of each assay, **69.9–74.6%** of each training partition | per assay 41.9 / 43.4 / 44.7% and 69.9 / 72.4 / 74.6%; matches Table 2.3 and §7.1 |

After the corrections the whole consistency check was re-run against the bundled report: **0 table
mismatches and 0 failed prose claims. There are no open items.**
