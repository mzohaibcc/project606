# Class Balancing and Calibration for Toxicity Prediction on New Chemical Scaffolds

**Final report — CSC606 Machine Learning individual project**
Muhammad Zohaib Chaudhary · 25 November 2026

---

## 1. Question, scope, and what was actually run

### 1.1 Research question

**Which balancing methods help models find more active compounds on new chemical scaffolds, while keeping their predicted probabilities reliable?**

The question has two halves that are reported separately and must not be conflated. *Detection* is measured by average precision, ROC AUC, and precision and recall among the top-ranked 10% of test compounds — the same fixed screening budget for every method. *Reliability of the predicted probabilities* is measured by Brier score, log loss and reliability curves, before and after sigmoid calibration fitted on a separate calibration partition.

### 1.2 Scope of the claim

Tox21 assay activity is an experimental endpoint. An active result in one of these assays is **not** a determination that a chemical is harmful to people. Every conclusion in this report concerns model performance, probability calibration, and the limits of this evaluation — nothing further.

### 1.3 What was actually run

| Factor | Levels | Count |
| --- | --- | --- |
| Assay | NR-AR, NR-AhR, SR-MMP | 3 |
| Split kind | scaffold (primary), random (reference) | 2 |
| Partition seed | 17, 42, 2026 | 3 |
| Classifier | regularised logistic regression, random forest (200 trees) | 2 |
| Balancing | none, inverse-frequency class weights, random undersampling, SMOTEENN | 4 |
| **Fitted final models** | | **144** |

All 144 models completed; there were no failed configurations (`outputs/full/execution_status.json`). Model selection used 1,728 cross-validation fits (three training-only folds across the four-entry hyperparameter grid for each of the 144 runs) plus 144 final refits. Total model fitting time was 1597.3 seconds across all runs, averaging 20.32 s per forest model and 1.86 s per logistic model.

Each model is scored in three states, giving the 432 rows of `outputs/full/metrics.csv`:

- **raw** — the model's own scores (logistic `decision_function`, forest positive-class probability);
- **sigmoid** — those scores mapped through a nonnegative-slope Platt sigmoid fitted **only** on the held-out calibration partition;
- **constant** — a reference model that predicts the training prevalence for every compound.

Hyperparameters were chosen by mean average precision over three training-only CV folds (logistic `C` in {0.01, 0.1, 1, 10}; forest maximum depth 10 or unrestricted crossed with minimum leaf size 1 or 5). Resampling and class weights were fitted inside each fold. Neither the calibration partition nor the test partition informed any selection.

### 1.4 Verification status

Two independent audits were completed before this report was written.

| Audit | Record | Checks | Result |
| --- | --- | --- | --- |
| Data preparation, fingerprints, partitions | `docs/data_verification.json` | 37 | all passed |
| Modelling, calibration, evaluation, bootstrap | `docs/results_verification.json` | 63 named checks | **PASS** |

The modelling audit re-scored all 144 models against scikit-learn 1.9.1 from their own stored predictions (largest deviation over 2,592 metric comparisons: 3.33e-16), reloaded every stored model and regenerated its predictions bit-exactly, re-derived the leakage invariants from all 18 partition manifests, and rejoined the full bootstrap accounting to the metrics table. One recorded boolean is false — `sklearn_recomputed_raw_and_sigmoid_ap_auc_bit_identical` — and is a deliberately stricter variant of the ranking-invariance check; the audit's finding F2 attributes the residual 1.11e-16 to float64 summation order and confirms the project's own stored values agree exactly. No test-driven selection was found.

### 1.5 How to read the numbers in this report

These conventions apply throughout and are not restated in every table.

- **Sign convention for deltas.** Every delta is *balancing method minus the unbalanced `none` baseline*, computed within the same assay, split kind, partition seed, classifier and calibration state. For **average precision, ROC AUC, precision and recall a positive delta is favourable**. For **Brier score and log loss a negative delta is favourable**, because both are losses.
- **Intervals are descriptive.** The reported intervals are 95% percentile intervals from 1,000 paired bootstrap replicates, drawn conditional on the already-fitted models and the current test partition. Random-split partitions resample test examples; scaffold-split partitions resample whole scaffold groups, so the number of resampling units is the number of test scaffold groups, not the number of compounds. These intervals do **not** capture retraining or split uncertainty, and they are **not** adjusted for the large number of comparisons made here. Every comparison in this report is therefore descriptive, and no population-level claim is made from any of them. An interval that excludes zero is reported as exactly that — an interval that excludes zero in this one test partition — and never as a formal hypothesis-test outcome.
- **Three seeds are not independent replication.** The ± values in the results tables are standard deviations over the three partition seeds. They describe split and fit variation across overlapping test sets drawn from the same cleaned assay table. They are not standard errors and they do not establish reproducibility across datasets.
- **Exploratory subsets.** Following the pre-registered rule, any subset containing fewer than 20 positive compounds is exploratory and is labelled so wherever it appears.

---

## 2. Data and partitions

### 2.1 Cleaning and per-assay counts

The input is the public Tox21 CSV linked by DeepChem's official loader. Of 7,831 downloaded records, 8 failed SMILES parsing and 80 contained no carbon-bearing fragment, leaving 7,743 structurally eligible source records and 7,524 unique standardised structures. Each assay was then cleaned independently: a missing label excludes the record only from that assay, duplicate structures with consistent labels are merged, and structures with conflicting labels are removed from that assay entirely.

**Table 2.1 — Per-assay cleaning (`data/audit/assay_counts.csv`)**

| Assay | Raw records | Observed labels | Raw positives | Missing labels | Rows dropped: invalid structure | Structures dropped: conflicting labels | Duplicate rows merged | Clean structures | Clean positives | Clean prevalence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NR-AR | 7831 | 7265 | 309 | 566 | 87 | 8 | 204 | 6957 | 298 | 0.0428 |
| NR-AhR | 7831 | 6549 | 768 | 1282 | 82 | 10 | 168 | 6278 | 748 | 0.1191 |
| SR-MMP | 7831 | 5810 | 918 | 2021 | 70 | 7 | 145 | 5578 | 888 | 0.1592 |

The three assays span a useful range of class imbalance: roughly 1 active in 23 compounds for NR-AR, 1 in 8 for NR-AhR, and 1 in 6 for SR-MMP.

### 2.2 Realised partitions

Each assay was split independently into approximately 60% training, 20% calibration and 20% test, under three fixed seeds. Random partitions are stratified; scaffold partitions use only non-stereochemical Bemis–Murcko scaffold membership and group size, never activity labels, and no scaffold group is ever split across partitions.

**Table 2.2 — Realised partitions (`partitions/partition_summary.csv`)**

| Assay | Split | Partition | n | Positives (min–max over seeds) | Prevalence (min–max) | Realised fraction | Deviation (pp) | Scaffold groups (min–max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NR-AR | random | train | 4174 | 179–179 | 0.0429–0.0429 | 0.599971 | -0.0029 | 1408–1433 |
| NR-AR | random | calibration | 1391 | 59–59 | 0.0424–0.0424 | 0.199943 | -0.0057 | 576–584 |
| NR-AR | random | test | 1392 | 60–60 | 0.0431–0.0431 | 0.200086 | +0.0086 | 564–573 |
| NR-AR | scaffold | train | 4174 | 132–167 | 0.0316–0.0400 | 0.599971 | -0.0029 | 699–708 |
| NR-AR | scaffold | calibration | 1392 | 65–98 | 0.0467–0.0704 | 0.200086 | +0.0086 | 703–708 |
| NR-AR | scaffold | test | 1391 | 62–96 | 0.0446–0.0690 | 0.199943 | -0.0057 | 693–701 |
| NR-AhR | random | train | 3766 | 449–449 | 0.1192–0.1192 | 0.599873 | -0.0127 | 1251–1269 |
| NR-AhR | random | calibration | 1256 | 149–150 | 0.1186–0.1194 | 0.200064 | +0.0064 | 496–514 |
| NR-AhR | random | test | 1256 | 149–150 | 0.1186–0.1194 | 0.200064 | +0.0064 | 525–535 |
| NR-AhR | scaffold | train | 3767 | 331–368 | 0.0879–0.0977 | 0.600032 | +0.0032 | 622–624 |
| NR-AhR | scaffold | calibration | 1256 | 172–235 | 0.1369–0.1871 | 0.200064 | +0.0064 | 624–629 |
| NR-AhR | scaffold | test | 1255 | 170–208 | 0.1355–0.1657 | 0.199904 | -0.0096 | 627–632 |
| SR-MMP | random | train | 3346 | 533–533 | 0.1593–0.1593 | 0.599857 | -0.0143 | 1068–1099 |
| SR-MMP | random | calibration | 1116 | 177–178 | 0.1586–0.1595 | 0.200072 | +0.0072 | 426–457 |
| SR-MMP | random | test | 1116 | 177–178 | 0.1586–0.1595 | 0.200072 | +0.0072 | 438–446 |
| SR-MMP | scaffold | train | 3347 | 384–421 | 0.1147–0.1258 | 0.600036 | +0.0036 | 535–547 |
| SR-MMP | scaffold | calibration | 1116 | 247–270 | 0.2213–0.2419 | 0.200072 | +0.0072 | 535–549 |
| SR-MMP | scaffold | test | 1115 | 220–238 | 0.1973–0.2135 | 0.199892 | -0.0108 | 541–543 |

Realised fractions land within 0.015 percentage points of the requested 60/20/20 in every case. Because scaffold allocation is label-independent, the prevalence of the scaffold partitions is free to drift: the NR-AhR scaffold calibration partition ranges from 13.69% to 18.71% active across seeds, and the NR-AR scaffold test partition from 4.46% to 6.90%. **Test prevalence therefore differs between the two split kinds, and because average precision has prevalence as its floor, average precision is not directly comparable across split kinds without that correction.** Section 5 reports the prevalence-normalised ratio alongside the raw values.

Partition isolation was verified on all 18 manifests: zero structure overlap between training, calibration and test in every partition, and for the scaffold design, zero scaffold overlap as well. Under random splitting, scaffold overlap between training and test is expected and is quantified at 146–224 shared scaffold groups per partition (`partitions/verification.json`).

### 2.3 The scaffold-split caveat

This is the most important structural limitation of the primary experiment and it is stated here rather than buried in Section 7.

The scaffold allocator processes groups by decreasing size times a seeded priority and assigns each to the partition with the largest remaining absolute capacity. Under this fixed, label-independent rule, the two giant scaffold groups — `__EMPTY__` (every acyclic compound, pooled into one group by design) and plain benzene `c1ccccc1` — are the first groups placed and always land in training, for every assay and every seed.

**Table 2.3 — Giant scaffold groups (derived from the per-assay scaffold partition manifests, `partitions/<assay>/scaffold/<seed>.csv`)**

| Assay | Scaffold groups | `__EMPTY__` size | Benzene `c1ccccc1` size | Both as % of assay | Partition both land in | Both as % of train | Remaining pool | Test as % of remaining pool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NR-AR | 2104 | 1559 | 1359 | 41.9% | train (all 3 seeds) | 69.9% | 4039 | 34.4% |
| NR-AhR | 1878 | 1496 | 1230 | 43.4% | train (all 3 seeds) | 72.4% | 3552 | 35.3% |
| SR-MMP | 1625 | 1354 | 1142 | 44.7% | train (all 3 seeds) | 74.6% | 3082 | 36.2% |

The consequence is explicit: **the scaffold test set is the remaining scaffold distribution, not a uniform draw over all scaffold groups.** 41.9–44.7% of each assay's compounds sit in two groups that are guaranteed to be training data; those two groups make up 69.9–74.6% of the training partition. The held-out 20% is drawn from the other ~1,600–2,100 scaffold groups, which are smaller and structurally more peripheral. Scaffold test performance in this report should be read as performance on that remaining distribution of smaller scaffold groups, and not as an unbiased estimate of performance on an arbitrary new scaffold.

### 2.4 How unfamiliar are the test compounds?

Scaffold separation does not by itself guarantee that test fingerprints are dissimilar to training fingerprints: two molecules with different Bemis–Murcko scaffolds can still share most of their Morgan bits.

**Table 2.4 — Nearest-neighbour Tanimoto similarity of each test compound to its closest training compound (`outputs/full/alignment_similarity_audit.csv`)**

| Assay | Split | Test n | Test positives (min–max) | Mean NN Tanimoto (min–max over seeds) | Median NN Tanimoto (min–max) |
| --- | --- | --- | --- | --- | --- |
| NR-AR | random | 1392 | 60–60 | 0.5625–0.5697 | 0.5500–0.5614 |
| NR-AR | scaffold | 1391 | 62–96 | 0.4177–0.4239 | 0.3978–0.4038 |
| NR-AhR | random | 1256 | 149–150 | 0.5503–0.5550 | 0.5385–0.5448 |
| NR-AhR | scaffold | 1255 | 170–208 | 0.4089–0.4224 | 0.3878–0.4000 |
| SR-MMP | random | 1116 | 177–178 | 0.5555–0.5613 | 0.5385–0.5455 |
| SR-MMP | scaffold | 1115 | 220–238 | 0.4086–0.4103 | 0.3846–0.3889 |

The scaffold split does shift the test set toward lower similarity — mean nearest-neighbour Tanimoto drops from about 0.55–0.57 to about 0.41–0.42 — but the scaffold test compounds remain moderately similar to training compounds on average. Section 6 uses fixed similarity bins to look inside this distribution.

---

## 3. Detection results

All detection metrics in this section are reported in the **raw** state. This is not a presentational choice: as established in Section 4.1, average precision and ROC AUC are bit-identical between the raw and sigmoid states for all 144 models, and the top-10% precision and recall deltas are likewise identical, so there is only one set of ranking results to report.

### 3.1 Per-assay results, seed mean ± SD over the three partition seeds

Each table is one assay and one split kind. The ± values are standard deviations across the three partition seeds (every `_count` column in `outputs/full/summary.csv` is 3).

**Table 3.1 — NR-AR, random split** (test prevalence 0.0431)

| Classifier | Balancing | Average precision | ROC AUC | Precision @ top 10% | Recall @ top 10% |
| --- | --- | --- | --- | --- | --- |
| forest | none (baseline) | 0.550 ± 0.063 | 0.759 ± 0.034 | 0.226 ± 0.029 | 0.528 ± 0.067 |
| forest | weights | 0.557 ± 0.068 | 0.778 ± 0.028 | 0.238 ± 0.025 | 0.556 ± 0.059 |
| forest | undersample | 0.556 ± 0.076 | 0.777 ± 0.036 | 0.240 ± 0.022 | 0.561 ± 0.051 |
| forest | SMOTEENN | 0.552 ± 0.068 | 0.763 ± 0.033 | 0.245 ± 0.029 | 0.572 ± 0.067 |
| logistic | none (baseline) | 0.537 ± 0.069 | 0.746 ± 0.022 | 0.229 ± 0.026 | 0.533 ± 0.060 |
| logistic | weights | 0.542 ± 0.074 | 0.754 ± 0.014 | 0.236 ± 0.026 | 0.550 ± 0.060 |
| logistic | undersample | 0.544 ± 0.080 | 0.730 ± 0.046 | 0.236 ± 0.033 | 0.550 ± 0.076 |
| logistic | SMOTEENN | 0.537 ± 0.070 | 0.758 ± 0.024 | 0.236 ± 0.026 | 0.550 ± 0.060 |

**Table 3.2 — NR-AR, scaffold split** (test prevalence 0.0446–0.0690 across seeds)

| Classifier | Balancing | Average precision | ROC AUC | Precision @ top 10% | Recall @ top 10% |
| --- | --- | --- | --- | --- | --- |
| forest | none (baseline) | 0.497 ± 0.149 | 0.741 ± 0.077 | 0.298 ± 0.151 | 0.537 ± 0.136 |
| forest | weights | 0.493 ± 0.154 | 0.758 ± 0.067 | 0.293 ± 0.155 | 0.526 ± 0.141 |
| forest | undersample | 0.502 ± 0.168 | 0.764 ± 0.070 | 0.293 ± 0.155 | 0.526 ± 0.141 |
| forest | SMOTEENN | 0.491 ± 0.152 | 0.759 ± 0.051 | 0.305 ± 0.151 | 0.550 ± 0.128 |
| logistic | none (baseline) | 0.394 ± 0.121 | 0.724 ± 0.065 | 0.248 ± 0.127 | 0.447 ± 0.120 |
| logistic | weights | 0.447 ± 0.190 | 0.740 ± 0.063 | 0.295 ± 0.153 | 0.531 ± 0.137 |
| logistic | undersample | 0.385 ± 0.217 | 0.748 ± 0.062 | 0.281 ± 0.165 | 0.500 ± 0.163 |
| logistic | SMOTEENN | 0.389 ± 0.238 | 0.734 ± 0.062 | 0.286 ± 0.156 | 0.512 ± 0.146 |

**Table 3.3 — NR-AhR, random split** (test prevalence 0.1186–0.1194)

| Classifier | Balancing | Average precision | ROC AUC | Precision @ top 10% | Recall @ top 10% |
| --- | --- | --- | --- | --- | --- |
| forest | none (baseline) | 0.566 ± 0.037 | 0.865 ± 0.008 | 0.603 ± 0.021 | 0.509 ± 0.017 |
| forest | weights | 0.588 ± 0.022 | 0.880 ± 0.011 | 0.601 ± 0.005 | 0.507 ± 0.003 |
| forest | undersample | 0.529 ± 0.016 | 0.866 ± 0.006 | 0.540 ± 0.027 | 0.455 ± 0.024 |
| forest | SMOTEENN | 0.503 ± 0.020 | 0.864 ± 0.003 | 0.524 ± 0.021 | 0.442 ± 0.019 |
| logistic | none (baseline) | 0.586 ± 0.030 | 0.875 ± 0.005 | 0.590 ± 0.012 | 0.498 ± 0.011 |
| logistic | weights | 0.581 ± 0.031 | 0.876 ± 0.006 | 0.577 ± 0.012 | 0.487 ± 0.010 |
| logistic | undersample | 0.544 ± 0.031 | 0.864 ± 0.004 | 0.556 ± 0.021 | 0.469 ± 0.019 |
| logistic | SMOTEENN | 0.535 ± 0.029 | 0.866 ± 0.008 | 0.550 ± 0.005 | 0.464 ± 0.002 |

**Table 3.4 — NR-AhR, scaffold split** (test prevalence 0.1355–0.1657)

| Classifier | Balancing | Average precision | ROC AUC | Precision @ top 10% | Recall @ top 10% |
| --- | --- | --- | --- | --- | --- |
| forest | none (baseline) | 0.469 ± 0.059 | 0.788 ± 0.021 | 0.513 ± 0.087 | 0.331 ± 0.022 |
| forest | weights | 0.510 ± 0.046 | 0.822 ± 0.008 | 0.556 ± 0.048 | 0.361 ± 0.010 |
| forest | undersample | 0.468 ± 0.041 | 0.805 ± 0.007 | 0.529 ± 0.052 | 0.344 ± 0.015 |
| forest | SMOTEENN | 0.442 ± 0.049 | 0.802 ± 0.022 | 0.497 ± 0.048 | 0.323 ± 0.013 |
| logistic | none (baseline) | 0.514 ± 0.043 | 0.819 ± 0.006 | 0.587 ± 0.069 | 0.381 ± 0.004 |
| logistic | weights | 0.502 ± 0.059 | 0.816 ± 0.008 | 0.585 ± 0.064 | 0.379 ± 0.017 |
| logistic | undersample | 0.477 ± 0.042 | 0.811 ± 0.007 | 0.561 ± 0.057 | 0.365 ± 0.033 |
| logistic | SMOTEENN | 0.474 ± 0.045 | 0.802 ± 0.013 | 0.537 ± 0.064 | 0.349 ± 0.025 |

**Table 3.5 — SR-MMP, random split** (test prevalence 0.1586–0.1595)

| Classifier | Balancing | Average precision | ROC AUC | Precision @ top 10% | Recall @ top 10% |
| --- | --- | --- | --- | --- | --- |
| forest | none (baseline) | 0.662 ± 0.010 | 0.870 ± 0.013 | 0.729 ± 0.021 | 0.461 ± 0.012 |
| forest | weights | 0.644 ± 0.014 | 0.871 ± 0.016 | 0.723 ± 0.000 | 0.457 ± 0.001 |
| forest | undersample | 0.621 ± 0.003 | 0.860 ± 0.009 | 0.690 ± 0.019 | 0.436 ± 0.013 |
| forest | SMOTEENN | 0.578 ± 0.031 | 0.853 ± 0.013 | 0.670 ± 0.018 | 0.423 ± 0.010 |
| logistic | none (baseline) | 0.636 ± 0.014 | 0.865 ± 0.011 | 0.729 ± 0.010 | 0.461 ± 0.005 |
| logistic | weights | 0.638 ± 0.011 | 0.868 ± 0.010 | 0.714 ± 0.015 | 0.451 ± 0.008 |
| logistic | undersample | 0.573 ± 0.005 | 0.849 ± 0.010 | 0.673 ± 0.010 | 0.425 ± 0.006 |
| logistic | SMOTEENN | 0.566 ± 0.038 | 0.841 ± 0.001 | 0.658 ± 0.040 | 0.415 ± 0.025 |

**Table 3.6 — SR-MMP, scaffold split** (test prevalence 0.1973–0.2135)

| Classifier | Balancing | Average precision | ROC AUC | Precision @ top 10% | Recall @ top 10% |
| --- | --- | --- | --- | --- | --- |
| forest | none (baseline) | 0.477 ± 0.053 | 0.752 ± 0.030 | 0.613 ± 0.084 | 0.297 ± 0.029 |
| forest | weights | 0.516 ± 0.058 | 0.785 ± 0.034 | 0.613 ± 0.069 | 0.297 ± 0.022 |
| forest | undersample | 0.497 ± 0.072 | 0.773 ± 0.023 | 0.622 ± 0.091 | 0.301 ± 0.034 |
| forest | SMOTEENN | 0.465 ± 0.038 | 0.760 ± 0.013 | 0.560 ± 0.031 | 0.272 ± 0.005 |
| logistic | none (baseline) | 0.504 ± 0.051 | 0.790 ± 0.017 | 0.607 ± 0.045 | 0.295 ± 0.011 |
| logistic | weights | 0.508 ± 0.062 | 0.795 ± 0.021 | 0.604 ± 0.055 | 0.293 ± 0.018 |
| logistic | undersample | 0.439 ± 0.046 | 0.752 ± 0.018 | 0.521 ± 0.040 | 0.253 ± 0.010 |
| logistic | SMOTEENN | 0.467 ± 0.057 | 0.760 ± 0.025 | 0.551 ± 0.052 | 0.267 ± 0.015 |

**Table 3.7 — Constant-prevalence reference model** (predicts the training prevalence for every compound; seed means)

| Assay | Split | Test prevalence (seed mean) | AP | ROC AUC | Precision @10% | Recall @10% | Brier | Log loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NR-AR | random | 0.0431 | 0.0431 | 0.5000 | 0.0595 | 0.1389 | 0.0412 | 0.1777 |
| NR-AR | scaffold | 0.0537 | 0.0537 | 0.5000 | 0.0476 | 0.0891 | 0.0512 | 0.2147 |
| NR-AhR | random | 0.1189 | 0.1189 | 0.5000 | 0.1111 | 0.0937 | 0.1048 | 0.3647 |
| NR-AhR | scaffold | 0.1548 | 0.1548 | 0.5000 | 0.1190 | 0.0765 | 0.1348 | 0.4509 |
| SR-MMP | random | 0.1589 | 0.1589 | 0.5000 | 0.1756 | 0.1109 | 0.1337 | 0.4378 |
| SR-MMP | scaffold | 0.2069 | 0.2069 | 0.5000 | 0.1637 | 0.0796 | 0.1717 | 0.5405 |

The constant model's average precision equals test prevalence and its ROC AUC is exactly 0.5 by construction, both confirmed by the audit. Every fitted model is far above this floor on both measures.

### 3.2 Paired bootstrap deltas against the unbalanced baseline

Each cell below is the method-minus-`none` difference on that exact test partition, followed by its 95% percentile bootstrap interval from 1,000 paired replicates. **Positive is favourable** for both metrics in this subsection. These are descriptive, unadjusted, within-partition intervals; see Section 1.5.

**Table 3.8 — Average precision, Δ (method − none) with 95% percentile bootstrap interval, by partition seed**

| Assay | Split | Classifier | Balancing | seed 17 | seed 42 | seed 2026 |
| --- | --- | --- | --- | --- | --- | --- |
| NR-AR | random | forest | weights | +0.010 [-0.011, +0.036] | +0.013 [-0.009, +0.037] | +0.001 [-0.031, +0.032] |
| NR-AR | random | forest | undersample | +0.021 [-0.001, +0.052] | +0.013 [-0.011, +0.039] | -0.015 [-0.052, +0.021] |
| NR-AR | random | forest | SMOTEENN | -0.001 [-0.027, +0.026] | +0.009 [-0.014, +0.032] | -0.001 [-0.034, +0.032] |
| NR-AR | random | logistic | weights | +0.024 [-0.003, +0.057] | +0.004 [-0.005, +0.013] | -0.013 [-0.033, +0.006] |
| NR-AR | random | logistic | undersample | +0.027 [-0.016, +0.075] | +0.010 [-0.015, +0.035] | -0.017 [-0.054, +0.018] |
| NR-AR | random | logistic | SMOTEENN | +0.018 [-0.008, +0.044] | -0.005 [-0.028, +0.014] | -0.013 [-0.045, +0.023] |
| NR-AR | scaffold | forest | weights | -0.010 [-0.034, +0.009] | -0.003 [-0.033, +0.023] | +0.001 [-0.016, +0.059] |
| NR-AR | scaffold | forest | undersample | +0.006 [-0.030, +0.042] | -0.017 [-0.051, +0.012] | +0.025 [-0.026, +0.042] |
| NR-AR | scaffold | forest | SMOTEENN | -0.014 [-0.080, +0.041] | -0.003 [-0.030, +0.024] | -0.003 [-0.042, +0.038] |
| NR-AR | scaffold | logistic | weights | -0.020 [-0.129, +0.065] | +0.038 [-0.063, +0.134] | +0.140 [+0.034, +0.213] |
| NR-AR | scaffold | logistic | undersample | -0.083 [-0.210, +0.039] | -0.053 [-0.213, +0.134] | +0.107 [-0.033, +0.289] |
| NR-AR | scaffold | logistic | SMOTEENN | -0.148 [-0.260, -0.020] | -0.003 [-0.140, +0.153] | +0.135 [+0.031, +0.219] |
| NR-AhR | random | forest | weights | +0.042 [-0.004, +0.080] | +0.006 [-0.028, +0.041] | +0.018 [-0.016, +0.054] |
| NR-AhR | random | forest | undersample | -0.008 [-0.057, +0.033] | -0.061 [-0.113, -0.003] | -0.044 [-0.099, +0.006] |
| NR-AhR | random | forest | SMOTEENN | -0.009 [-0.060, +0.034] | -0.122 [-0.178, -0.057] | -0.057 [-0.116, +0.003] |
| NR-AhR | random | logistic | weights | -0.005 [-0.022, +0.013] | -0.007 [-0.021, +0.008] | -0.003 [-0.018, +0.010] |
| NR-AhR | random | logistic | undersample | -0.049 [-0.093, -0.005] | -0.024 [-0.065, +0.013] | -0.055 [-0.092, -0.023] |
| NR-AhR | random | logistic | SMOTEENN | -0.049 [-0.078, -0.014] | -0.054 [-0.087, -0.015] | -0.052 [-0.087, -0.015] |
| NR-AhR | scaffold | forest | weights | +0.012 [-0.019, +0.045] | +0.050 [+0.006, +0.094] | +0.062 [+0.013, +0.113] |
| NR-AhR | scaffold | forest | undersample | -0.024 [-0.059, +0.018] | +0.014 [-0.040, +0.061] | +0.007 [-0.044, +0.057] |
| NR-AhR | scaffold | forest | SMOTEENN | -0.075 [-0.129, -0.011] | -0.019 [-0.073, +0.035] | +0.013 [-0.041, +0.074] |
| NR-AhR | scaffold | logistic | weights | -0.001 [-0.030, +0.029] | -0.032 [-0.056, -0.001] | -0.003 [-0.023, +0.016] |
| NR-AhR | scaffold | logistic | undersample | -0.007 [-0.050, +0.032] | -0.036 [-0.069, +0.001] | -0.070 [-0.099, -0.041] |
| NR-AhR | scaffold | logistic | SMOTEENN | -0.040 [-0.076, -0.003] | -0.043 [-0.080, +0.000] | -0.038 [-0.061, -0.013] |
| SR-MMP | random | forest | weights | -0.015 [-0.044, +0.015] | -0.028 [-0.055, -0.003] | -0.009 [-0.038, +0.021] |
| SR-MMP | random | forest | undersample | -0.030 [-0.075, +0.017] | -0.041 [-0.085, +0.010] | -0.049 [-0.087, -0.009] |
| SR-MMP | random | forest | SMOTEENN | -0.079 [-0.134, -0.021] | -0.115 [-0.172, -0.053] | -0.057 [-0.102, -0.010] |
| SR-MMP | random | logistic | weights | +0.007 [-0.005, +0.020] | +0.000 [-0.010, +0.010] | -0.002 [-0.014, +0.012] |
| SR-MMP | random | logistic | undersample | -0.055 [-0.082, -0.024] | -0.072 [-0.114, -0.033] | -0.063 [-0.093, -0.033] |
| SR-MMP | random | logistic | SMOTEENN | -0.100 [-0.135, -0.063] | -0.056 [-0.090, -0.022] | -0.056 [-0.098, -0.020] |
| SR-MMP | scaffold | forest | weights | +0.019 [-0.015, +0.055] | +0.046 [+0.001, +0.085] | +0.050 [+0.022, +0.075] |
| SR-MMP | scaffold | forest | undersample | +0.044 [-0.023, +0.108] | -0.009 [-0.053, +0.029] | +0.024 [-0.020, +0.058] |
| SR-MMP | scaffold | forest | SMOTEENN | -0.000 [-0.058, +0.059] | -0.002 [-0.067, +0.066] | -0.033 [-0.080, +0.017] |
| SR-MMP | scaffold | logistic | weights | +0.030 [+0.004, +0.056] | -0.008 [-0.024, +0.008] | -0.010 [-0.024, +0.008] |
| SR-MMP | scaffold | logistic | undersample | -0.047 [-0.096, +0.007] | -0.058 [-0.087, -0.031] | -0.088 [-0.134, -0.047] |
| SR-MMP | scaffold | logistic | SMOTEENN | -0.029 [-0.069, +0.015] | -0.043 [-0.081, -0.007] | -0.037 [-0.082, -0.002] |

**Table 3.9 — Recall at the top 10%, Δ (method − none) with 95% percentile bootstrap interval, by partition seed**

| Assay | Split | Classifier | Balancing | seed 17 | seed 42 | seed 2026 |
| --- | --- | --- | --- | --- | --- | --- |
| NR-AR | random | forest | weights | +0.033 [-0.036, +0.093] | +0.017 [-0.053, +0.073] | +0.033 [+0.000, +0.100] |
| NR-AR | random | forest | undersample | +0.033 [-0.032, +0.097] | +0.017 [-0.053, +0.074] | +0.050 [-0.013, +0.107] |
| NR-AR | random | forest | SMOTEENN | +0.017 [-0.045, +0.077] | +0.050 [-0.019, +0.103] | +0.067 [+0.015, +0.137] |
| NR-AR | random | logistic | weights | +0.017 [-0.036, +0.073] | +0.017 [+0.000, +0.051] | +0.017 [-0.020, +0.054] |
| NR-AR | random | logistic | undersample | +0.017 [-0.037, +0.083] | +0.033 [-0.023, +0.095] | +0.000 [-0.065, +0.078] |
| NR-AR | random | logistic | SMOTEENN | +0.017 [-0.040, +0.065] | +0.017 [-0.019, +0.054] | +0.017 [-0.020, +0.056] |
| NR-AR | scaffold | forest | weights | -0.032 [-0.100, +0.039] | +0.000 [-0.045, +0.059] | +0.000 [-0.011, +0.045] |
| NR-AR | scaffold | forest | undersample | -0.032 [-0.104, +0.057] | +0.000 [-0.052, +0.069] | +0.000 [+0.000, +0.030] |
| NR-AR | scaffold | forest | SMOTEENN | -0.016 [-0.079, +0.047] | +0.045 [+0.000, +0.111] | +0.010 [-0.023, +0.045] |
| NR-AR | scaffold | logistic | weights | +0.000 [-0.074, +0.070] | +0.136 [+0.031, +0.207] | +0.115 [+0.019, +0.288] |
| NR-AR | scaffold | logistic | undersample | -0.016 [-0.116, +0.079] | +0.061 [-0.067, +0.172] | +0.115 [+0.021, +0.277] |
| NR-AR | scaffold | logistic | SMOTEENN | -0.032 [-0.107, +0.061] | +0.121 [+0.017, +0.214] | +0.104 [+0.011, +0.254] |
| NR-AhR | random | forest | weights | +0.020 [-0.019, +0.079] | -0.020 [-0.062, +0.027] | -0.007 [-0.051, +0.046] |
| NR-AhR | random | forest | undersample | -0.047 [-0.092, +0.006] | -0.040 [-0.087, +0.031] | -0.073 [-0.136, -0.015] |
| NR-AhR | random | forest | SMOTEENN | -0.027 [-0.086, +0.023] | -0.087 [-0.149, -0.019] | -0.087 [-0.146, -0.013] |
| NR-AhR | random | logistic | weights | -0.013 [-0.045, +0.030] | -0.013 [-0.037, +0.021] | -0.007 [-0.048, +0.023] |
| NR-AhR | random | logistic | undersample | -0.013 [-0.074, +0.026] | -0.027 [-0.071, +0.022] | -0.047 [-0.094, +0.000] |
| NR-AhR | random | logistic | SMOTEENN | -0.027 [-0.066, +0.019] | -0.047 [-0.101, +0.007] | -0.027 [-0.082, +0.014] |
| NR-AhR | scaffold | forest | weights | +0.005 [-0.040, +0.061] | +0.065 [-0.006, +0.108] | +0.020 [-0.038, +0.065] |
| NR-AhR | scaffold | forest | undersample | +0.005 [-0.043, +0.065] | +0.047 [-0.024, +0.089] | -0.015 [-0.070, +0.034] |
| NR-AhR | scaffold | forest | SMOTEENN | -0.038 [-0.099, +0.031] | +0.024 [-0.045, +0.083] | -0.010 [-0.051, +0.040] |
| NR-AhR | scaffold | logistic | weights | -0.019 [-0.053, +0.026] | +0.006 [-0.080, +0.034] | +0.010 [-0.028, +0.052] |
| NR-AhR | scaffold | logistic | undersample | +0.000 [-0.042, +0.039] | +0.012 [-0.070, +0.053] | -0.059 [-0.092, -0.016] |
| NR-AhR | scaffold | logistic | SMOTEENN | -0.058 [-0.101, +0.004] | -0.024 [-0.098, +0.011] | -0.015 [-0.050, +0.021] |
| SR-MMP | random | forest | weights | +0.011 [-0.027, +0.040] | -0.011 [-0.049, +0.018] | -0.011 [-0.055, +0.034] |
| SR-MMP | random | forest | undersample | +0.000 [-0.042, +0.047] | -0.028 [-0.077, +0.022] | -0.045 [-0.086, +0.010] |
| SR-MMP | random | forest | SMOTEENN | -0.034 [-0.087, +0.016] | -0.045 [-0.098, +0.000] | -0.034 [-0.088, +0.017] |
| SR-MMP | random | logistic | weights | -0.011 [-0.033, +0.022] | -0.011 [-0.035, +0.017] | -0.006 [-0.027, +0.024] |
| SR-MMP | random | logistic | undersample | -0.028 [-0.062, +0.016] | -0.040 [-0.089, -0.005] | -0.039 [-0.067, +0.005] |
| SR-MMP | random | logistic | SMOTEENN | -0.068 [-0.110, -0.018] | -0.017 [-0.066, +0.018] | -0.051 [-0.086, +0.006] |
| SR-MMP | scaffold | forest | weights | -0.004 [-0.037, +0.029] | +0.009 [-0.030, +0.048] | -0.004 [-0.040, +0.040] |
| SR-MMP | scaffold | forest | undersample | +0.021 [-0.046, +0.063] | +0.000 [-0.049, +0.036] | -0.008 [-0.038, +0.031] |
| SR-MMP | scaffold | forest | SMOTEENN | -0.038 [-0.076, +0.014] | +0.005 [-0.049, +0.071] | -0.042 [-0.090, +0.010] |
| SR-MMP | scaffold | logistic | weights | +0.021 [-0.013, +0.048] | -0.009 [-0.031, +0.019] | -0.017 [-0.042, +0.008] |
| SR-MMP | scaffold | logistic | undersample | -0.043 [-0.088, +0.004] | -0.041 [-0.073, -0.005] | -0.042 [-0.086, -0.008] |
| SR-MMP | scaffold | logistic | SMOTEENN | -0.013 [-0.057, +0.025] | -0.036 [-0.076, +0.010] | -0.034 [-0.079, -0.004] |

**Table 3.10 — ROC AUC, Δ summarised over the three seeds**

| Assay | Split | Classifier | Balancing | Mean Δ | Δ range over 3 seeds | Intervals excluding 0, favourable | Intervals excluding 0, unfavourable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NR-AR | random | forest | weights | +0.0192 | [+0.0128, +0.0233] | 0/3 | 0/3 |
| NR-AR | random | forest | undersample | +0.0180 | [+0.0024, +0.0406] | 1/3 | 0/3 |
| NR-AR | random | forest | SMOTEENN | +0.0043 | [-0.0045, +0.0149] | 0/3 | 0/3 |
| NR-AR | random | logistic | weights | +0.0079 | [-0.0103, +0.0188] | 0/3 | 0/3 |
| NR-AR | random | logistic | undersample | -0.0158 | [-0.0559, +0.0344] | 0/3 | 0/3 |
| NR-AR | random | logistic | SMOTEENN | +0.0117 | [-0.0008, +0.0286] | 0/3 | 0/3 |
| NR-AR | scaffold | forest | weights | +0.0178 | [+0.0063, +0.0363] | 0/3 | 0/3 |
| NR-AR | scaffold | forest | undersample | +0.0231 | [+0.0128, +0.0369] | 0/3 | 0/3 |
| NR-AR | scaffold | forest | SMOTEENN | +0.0186 | [-0.0046, +0.0621] | 1/3 | 0/3 |
| NR-AR | scaffold | logistic | weights | +0.0156 | [+0.0027, +0.0266] | 0/3 | 0/3 |
| NR-AR | scaffold | logistic | undersample | +0.0232 | [-0.0024, +0.0463] | 0/3 | 0/3 |
| NR-AR | scaffold | logistic | SMOTEENN | +0.0095 | [-0.0160, +0.0328] | 0/3 | 0/3 |
| NR-AhR | random | forest | weights | +0.0149 | [+0.0119, +0.0196] | 3/3 | 0/3 |
| NR-AhR | random | forest | undersample | +0.0010 | [-0.0037, +0.0041] | 0/3 | 0/3 |
| NR-AhR | random | forest | SMOTEENN | -0.0008 | [-0.0087, +0.0102] | 0/3 | 0/3 |
| NR-AhR | random | logistic | weights | +0.0013 | [-0.0008, +0.0049] | 0/3 | 0/3 |
| NR-AhR | random | logistic | undersample | -0.0111 | [-0.0162, -0.0037] | 0/3 | 2/3 |
| NR-AhR | random | logistic | SMOTEENN | -0.0088 | [-0.0125, -0.0021] | 0/3 | 1/3 |
| NR-AhR | scaffold | forest | weights | +0.0348 | [+0.0062, +0.0499] | 2/3 | 0/3 |
| NR-AhR | scaffold | forest | undersample | +0.0177 | [-0.0127, +0.0360] | 2/3 | 0/3 |
| NR-AhR | scaffold | forest | SMOTEENN | +0.0141 | [-0.0336, +0.0390] | 2/3 | 1/3 |
| NR-AhR | scaffold | logistic | weights | -0.0030 | [-0.0071, -0.0002] | 0/3 | 0/3 |
| NR-AhR | scaffold | logistic | undersample | -0.0079 | [-0.0174, +0.0046] | 0/3 | 1/3 |
| NR-AhR | scaffold | logistic | SMOTEENN | -0.0168 | [-0.0221, -0.0085] | 0/3 | 2/3 |
| SR-MMP | random | forest | weights | +0.0012 | [-0.0013, +0.0036] | 0/3 | 0/3 |
| SR-MMP | random | forest | undersample | -0.0105 | [-0.0135, -0.0054] | 0/3 | 0/3 |
| SR-MMP | random | forest | SMOTEENN | -0.0176 | [-0.0185, -0.0170] | 0/3 | 1/3 |
| SR-MMP | random | logistic | weights | +0.0032 | [+0.0016, +0.0057] | 1/3 | 0/3 |
| SR-MMP | random | logistic | undersample | -0.0158 | [-0.0172, -0.0136] | 0/3 | 3/3 |
| SR-MMP | random | logistic | SMOTEENN | -0.0243 | [-0.0353, -0.0157] | 0/3 | 2/3 |
| SR-MMP | scaffold | forest | weights | +0.0333 | [+0.0231, +0.0398] | 2/3 | 0/3 |
| SR-MMP | scaffold | forest | undersample | +0.0212 | [+0.0084, +0.0398] | 0/3 | 0/3 |
| SR-MMP | scaffold | forest | SMOTEENN | +0.0080 | [-0.0141, +0.0242] | 0/3 | 0/3 |
| SR-MMP | scaffold | logistic | weights | +0.0050 | [+0.0017, +0.0109] | 0/3 | 0/3 |
| SR-MMP | scaffold | logistic | undersample | -0.0385 | [-0.0531, -0.0309] | 0/3 | 2/3 |
| SR-MMP | scaffold | logistic | SMOTEENN | -0.0307 | [-0.0397, -0.0226] | 0/3 | 2/3 |

**Table 3.11 — Precision at the top 10%, Δ summarised over the three seeds**

| Assay | Split | Classifier | Balancing | Mean Δ | Δ range over 3 seeds | Intervals excluding 0, favourable | Intervals excluding 0, unfavourable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NR-AR | random | forest | weights | +0.0119 | [+0.0071, +0.0143] | 0/3 | 0/3 |
| NR-AR | random | forest | undersample | +0.0143 | [+0.0071, +0.0214] | 0/3 | 0/3 |
| NR-AR | random | forest | SMOTEENN | +0.0190 | [+0.0071, +0.0286] | 1/3 | 0/3 |
| NR-AR | random | logistic | weights | +0.0071 | [+0.0071, +0.0071] | 0/3 | 0/3 |
| NR-AR | random | logistic | undersample | +0.0071 | [+0.0000, +0.0143] | 0/3 | 0/3 |
| NR-AR | random | logistic | SMOTEENN | +0.0071 | [+0.0071, +0.0071] | 0/3 | 0/3 |
| NR-AR | scaffold | forest | weights | -0.0048 | [-0.0143, +0.0000] | 0/3 | 0/3 |
| NR-AR | scaffold | forest | undersample | -0.0048 | [-0.0143, +0.0000] | 0/3 | 0/3 |
| NR-AR | scaffold | forest | SMOTEENN | +0.0071 | [-0.0071, +0.0214] | 0/3 | 0/3 |
| NR-AR | scaffold | logistic | weights | +0.0476 | [+0.0000, +0.0786] | 2/3 | 0/3 |
| NR-AR | scaffold | logistic | undersample | +0.0333 | [-0.0071, +0.0786] | 1/3 | 0/3 |
| NR-AR | scaffold | logistic | SMOTEENN | +0.0381 | [-0.0143, +0.0714] | 2/3 | 0/3 |
| NR-AhR | random | forest | weights | -0.0026 | [-0.0238, +0.0238] | 0/3 | 0/3 |
| NR-AhR | random | forest | undersample | -0.0635 | [-0.0873, -0.0476] | 0/3 | 1/3 |
| NR-AhR | random | forest | SMOTEENN | -0.0794 | [-0.1032, -0.0317] | 0/3 | 2/3 |
| NR-AhR | random | logistic | weights | -0.0132 | [-0.0159, -0.0079] | 0/3 | 0/3 |
| NR-AhR | random | logistic | undersample | -0.0344 | [-0.0556, -0.0159] | 0/3 | 0/3 |
| NR-AhR | random | logistic | SMOTEENN | -0.0397 | [-0.0556, -0.0317] | 0/3 | 0/3 |
| NR-AhR | scaffold | forest | weights | +0.0423 | [+0.0079, +0.0873] | 0/3 | 0/3 |
| NR-AhR | scaffold | forest | undersample | +0.0159 | [-0.0238, +0.0635] | 0/3 | 0/3 |
| NR-AhR | scaffold | forest | SMOTEENN | -0.0159 | [-0.0635, +0.0317] | 0/3 | 0/3 |
| NR-AhR | scaffold | logistic | weights | -0.0026 | [-0.0317, +0.0159] | 0/3 | 0/3 |
| NR-AhR | scaffold | logistic | undersample | -0.0265 | [-0.0952, +0.0159] | 0/3 | 1/3 |
| NR-AhR | scaffold | logistic | SMOTEENN | -0.0503 | [-0.0952, -0.0238] | 0/3 | 0/3 |
| SR-MMP | random | forest | weights | -0.0060 | [-0.0179, +0.0179] | 0/3 | 0/3 |
| SR-MMP | random | forest | undersample | -0.0387 | [-0.0714, +0.0000] | 0/3 | 0/3 |
| SR-MMP | random | forest | SMOTEENN | -0.0595 | [-0.0714, -0.0536] | 0/3 | 0/3 |
| SR-MMP | random | logistic | weights | -0.0149 | [-0.0179, -0.0089] | 0/3 | 0/3 |
| SR-MMP | random | logistic | undersample | -0.0565 | [-0.0625, -0.0446] | 0/3 | 1/3 |
| SR-MMP | random | logistic | SMOTEENN | -0.0714 | [-0.1071, -0.0268] | 0/3 | 1/3 |
| SR-MMP | scaffold | forest | weights | -0.0000 | [-0.0089, +0.0179] | 0/3 | 0/3 |
| SR-MMP | scaffold | forest | undersample | +0.0089 | [-0.0179, +0.0446] | 0/3 | 0/3 |
| SR-MMP | scaffold | forest | SMOTEENN | -0.0536 | [-0.0893, +0.0089] | 0/3 | 0/3 |
| SR-MMP | scaffold | logistic | weights | -0.0030 | [-0.0357, +0.0446] | 0/3 | 0/3 |
| SR-MMP | scaffold | logistic | undersample | -0.0863 | [-0.0893, -0.0804] | 0/3 | 2/3 |
| SR-MMP | scaffold | logistic | SMOTEENN | -0.0565 | [-0.0714, -0.0268] | 0/3 | 1/3 |

### 3.3 What the deltas add up to

**Table 3.12 — Tally over all 18 comparisons per method per split kind (3 assays × 2 classifiers × 3 seeds)**

| Metric | Balancing | Split | Comparisons | Δ favourable in sign | Intervals excluding 0, favourable | Intervals excluding 0, unfavourable | Mean Δ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Average precision | weights | random | 18 | 10 | 0 | 1 | +0.0023 |
| Average precision | weights | scaffold | 18 | 10 | 6 | 1 | +0.0201 |
| Average precision | undersample | random | 18 | 4 | 0 | 7 | -0.0284 |
| Average precision | undersample | scaffold | 18 | 7 | 0 | 3 | -0.0148 |
| Average precision | SMOTEENN | random | 18 | 2 | 0 | 10 | -0.0445 |
| Average precision | SMOTEENN | scaffold | 18 | 2 | 1 | 6 | -0.0213 |
| ROC AUC | weights | random | 18 | 14 | 4 | 0 | +0.0079 |
| ROC AUC | weights | scaffold | 18 | 15 | 4 | 0 | +0.0172 |
| ROC AUC | undersample | random | 18 | 6 | 1 | 5 | -0.0057 |
| ROC AUC | undersample | scaffold | 18 | 11 | 2 | 3 | +0.0065 |
| ROC AUC | SMOTEENN | random | 18 | 5 | 0 | 4 | -0.0059 |
| ROC AUC | SMOTEENN | scaffold | 18 | 7 | 3 | 5 | +0.0004 |
| Precision @10% | weights | random | 18 | 8 | 0 | 0 | -0.0029 |
| Precision @10% | weights | scaffold | 18 | 9 | 2 | 0 | +0.0133 |
| Precision @10% | undersample | random | 18 | 5 | 0 | 2 | -0.0286 |
| Precision @10% | undersample | scaffold | 18 | 6 | 1 | 3 | -0.0099 |
| Precision @10% | SMOTEENN | random | 18 | 6 | 1 | 3 | -0.0373 |
| Precision @10% | SMOTEENN | scaffold | 18 | 6 | 2 | 1 | -0.0218 |
| Recall @10% | weights | random | 18 | 8 | 0 | 0 | +0.0030 |
| Recall @10% | weights | scaffold | 18 | 9 | 2 | 0 | +0.0167 |
| Recall @10% | undersample | random | 18 | 5 | 0 | 2 | -0.0154 |
| Recall @10% | undersample | scaffold | 18 | 6 | 1 | 3 | +0.0003 |
| Recall @10% | SMOTEENN | random | 18 | 6 | 1 | 3 | -0.0203 |
| Recall @10% | SMOTEENN | scaffold | 18 | 6 | 2 | 1 | -0.0026 |

Three patterns are visible and are consistent across metrics.

1. **The two resampling methods reduce detection more often than they improve it, and most clearly on the random split.** SMOTEENN's average precision delta is unfavourable in sign in 16 of its 18 random-split comparisons, with 10 intervals excluding zero unfavourably and none favourably; its mean delta is −0.0445. Random undersampling is milder but points the same way (7 of 18 intervals excluding zero unfavourably on the random split, none favourably). The damage is concentrated in the two higher-prevalence assays: across the 12 NR-AhR and SR-MMP random-split comparisons, SMOTEENN's average-precision delta is negative in all 12, ranging from −0.009 to −0.122.

2. **Inverse-frequency class weighting is the only method that does not systematically cost detection.** Across all 36 average-precision comparisons it has 6 intervals excluding zero favourably against 2 unfavourably, and a mean delta of +0.0112. On ROC AUC it is favourable in sign in 29 of 36 comparisons with 8 intervals excluding zero favourably and **zero** unfavourably.

3. **What gains exist are concentrated in one place: class weighting with the random forest on the scaffold split.** In those 9 comparisons the average precision delta is favourable in sign 7 times, 4 intervals exclude zero favourably and none unfavourably, and the mean delta is +0.0253. Those four are NR-AhR seeds 42 (+0.050 [+0.006, +0.094]) and 2026 (+0.062 [+0.013, +0.113]), and SR-MMP seeds 42 (+0.046 [+0.001, +0.085]) and 2026 (+0.050 [+0.022, +0.075]). The corresponding seed-mean gains are NR-AhR 0.469 → 0.510 and SR-MMP 0.477 → 0.516 average precision. On NR-AR, the smallest and most imbalanced assay, the same combination shows no gain (seed means 0.497 → 0.493, no interval excluding zero).

Two counter-observations keep this from being read as more than it is. The one unfavourable class-weighting interval on the scaffold split is NR-AhR/logistic/seed 42 (−0.032 [−0.056, −0.001]). And the largest single favourable average-precision delta anywhere in the study — NR-AR/scaffold/logistic/seed 2026, +0.140 [+0.034, +0.213] — comes from the group with by far the widest seed-to-seed spread (logistic scaffold NR-AR average precision 0.394 ± 0.121 for `none`, 0.447 ± 0.190 for weights); the same comparison is −0.020 on seed 17 and +0.038 on seed 42. That instability, not the point estimate, is the reportable finding there.

### 3.4 Figures

Precision–recall curves for each of the 36 assay/split/seed/classifier groups are committed alongside the metrics. Representative paths:

- `outputs/full/evaluation/NR-AhR/scaffold/42/forest/precision_recall.png` — the group with the clearest class-weighting gain.
- `outputs/full/evaluation/SR-MMP/random/17/forest/precision_recall.png` — a group where SMOTEENN visibly costs precision across the useful recall range.
- `outputs/full/evaluation/NR-AR/scaffold/2026/logistic/precision_recall.png` — the high-variance NR-AR scaffold logistic group discussed above.

The full set follows the pattern `outputs/full/evaluation/<assay>/<split_kind>/<seed>/<classifier>/precision_recall.png`.

---

## 4. Calibration results

### 4.1 Calibration cannot change ranking here, and it did not

Before reporting any calibration effect, one result must be stated plainly because it determines what calibration is allowed to be credited with.

The sigmoid mapping fitted in this project is constrained to a nonnegative slope. A monotone non-decreasing map cannot reorder compounds, so it cannot change any metric that depends only on the ranking. This holds exactly in the stored results: across all 144 models, the maximum absolute difference between the raw and sigmoid states is **exactly 0.0** for average precision and **exactly 0.0** for ROC AUC, and the paired bootstrap deltas for average precision, ROC AUC, precision at 10% and recall at 10% are identical to 0.0 between the two states. The audit additionally confirmed zero strict rank reversals over all test pairs in all 144 models, and recorded a fitted slope range of 0.510 to 1.887 with no zero-slope or constant-mapping case.

**Calibration is therefore never presented in this report as improving average precision, ROC AUC, or any other ranking measure. Its effects belong entirely to Brier score, log loss and the reliability curves.**

One related detail, for completeness. Two of the 144 models — `NR-AhR/scaffold/42/forest/none` and `SR-MMP/scaffold/17/forest/none` — record `rank_order_identical: false`. This is **not** a rank reversal. The audit (finding F1) established that in each case the sigmoid maps two or three compounds whose raw forest probabilities differ by 1–2 units in the last place of float64 onto a single float64 value, and that every such collapsed group consists entirely of negatives. Average precision and ROC AUC are mathematically unchanged; the flag is a strict permutation-equality test that is sensitive to float64 tie structure carrying no ranking information.

### 4.2 Brier score and log loss, before and after calibration

**Table 4.1 — Brier score and log loss, seed mean ± SD, in all three states**

| Assay | Split | Classifier | Balancing | Brier raw | Brier sigmoid | Brier constant | Log loss raw | Log loss sigmoid | Log loss constant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NR-AR | random | forest | none (baseline) | 0.0246 ± 0.0028 | 0.0231 ± 0.0028 | 0.0412 | 0.1924 ± 0.1410 | 0.1099 ± 0.0116 | 0.1777 |
| NR-AR | random | forest | weights | 0.0917 ± 0.0154 | 0.0237 ± 0.0042 | 0.0412 | 0.3454 ± 0.0467 | 0.1135 ± 0.0159 | 0.1777 |
| NR-AR | random | forest | undersample | 0.1253 ± 0.0140 | 0.0250 ± 0.0043 | 0.0412 | 0.4267 ± 0.0304 | 0.1194 ± 0.0140 | 0.1777 |
| NR-AR | random | forest | SMOTEENN | 0.0500 ± 0.0143 | 0.0244 ± 0.0031 | 0.0412 | 0.2212 ± 0.0540 | 0.1163 ± 0.0113 | 0.1777 |
| NR-AR | random | logistic | none (baseline) | 0.0253 ± 0.0029 | 0.0252 ± 0.0036 | 0.0412 | 0.1258 ± 0.0120 | 0.1217 ± 0.0096 | 0.1777 |
| NR-AR | random | logistic | weights | 0.1062 ± 0.0103 | 0.0248 ± 0.0034 | 0.0412 | 0.3784 ± 0.0234 | 0.1201 ± 0.0095 | 0.1777 |
| NR-AR | random | logistic | undersample | 0.2014 ± 0.0136 | 0.0253 ± 0.0041 | 0.0412 | 0.5939 ± 0.0276 | 0.1227 ± 0.0142 | 0.1777 |
| NR-AR | random | logistic | SMOTEENN | 0.1299 ± 0.0244 | 0.0258 ± 0.0028 | 0.0412 | 0.4258 ± 0.0550 | 0.1244 ± 0.0084 | 0.1777 |
| NR-AR | scaffold | forest | none (baseline) | 0.0355 ± 0.0045 | 0.0314 ± 0.0024 | 0.0512 | 0.1509 ± 0.0108 | 0.1404 ± 0.0057 | 0.2147 |
| NR-AR | scaffold | forest | weights | 0.0589 ± 0.0388 | 0.0317 ± 0.0014 | 0.0512 | 0.2445 ± 0.1205 | 0.1416 ± 0.0047 | 0.2147 |
| NR-AR | scaffold | forest | undersample | 0.1645 ± 0.0137 | 0.0341 ± 0.0021 | 0.0512 | 0.5153 ± 0.0296 | 0.1491 ± 0.0074 | 0.2147 |
| NR-AR | scaffold | forest | SMOTEENN | 0.0403 ± 0.0013 | 0.0327 ± 0.0015 | 0.0512 | 0.1953 ± 0.0110 | 0.1438 ± 0.0047 | 0.2147 |
| NR-AR | scaffold | logistic | none (baseline) | 0.0429 ± 0.0073 | 0.0452 ± 0.0116 | 0.0512 | 0.1781 ± 0.0182 | 0.1827 ± 0.0292 | 0.2147 |
| NR-AR | scaffold | logistic | weights | 0.0857 ± 0.0348 | 0.0363 ± 0.0010 | 0.0512 | 0.3150 ± 0.0932 | 0.1578 ± 0.0050 | 0.2147 |
| NR-AR | scaffold | logistic | undersample | 0.2397 ± 0.0686 | 0.0425 ± 0.0043 | 0.0512 | 0.7691 ± 0.2509 | 0.1725 ± 0.0062 | 0.2147 |
| NR-AR | scaffold | logistic | SMOTEENN | 0.1132 ± 0.0305 | 0.0405 ± 0.0015 | 0.0512 | 0.4585 ± 0.0687 | 0.1681 ± 0.0012 | 0.2147 |
| NR-AhR | random | forest | none (baseline) | 0.0757 ± 0.0046 | 0.0727 ± 0.0030 | 0.1048 | 0.2684 ± 0.0142 | 0.2535 ± 0.0085 | 0.3647 |
| NR-AhR | random | forest | weights | 0.0833 ± 0.0007 | 0.0706 ± 0.0020 | 0.1048 | 0.2824 ± 0.0008 | 0.2431 ± 0.0069 | 0.3647 |
| NR-AhR | random | forest | undersample | 0.1605 ± 0.0076 | 0.0764 ± 0.0021 | 0.1048 | 0.4779 ± 0.0188 | 0.2578 ± 0.0041 | 0.3647 |
| NR-AhR | random | forest | SMOTEENN | 0.1078 ± 0.0091 | 0.0776 ± 0.0014 | 0.1048 | 0.3624 ± 0.0419 | 0.2610 ± 0.0026 | 0.3647 |
| NR-AhR | random | logistic | none (baseline) | 0.0716 ± 0.0010 | 0.0712 ± 0.0013 | 0.1048 | 0.2472 ± 0.0016 | 0.2452 ± 0.0022 | 0.3647 |
| NR-AhR | random | logistic | weights | 0.1162 ± 0.0028 | 0.0715 ± 0.0012 | 0.1048 | 0.3700 ± 0.0066 | 0.2458 ± 0.0015 | 0.3647 |
| NR-AhR | random | logistic | undersample | 0.1632 ± 0.0062 | 0.0758 ± 0.0018 | 0.1048 | 0.4916 ± 0.0132 | 0.2571 ± 0.0043 | 0.3647 |
| NR-AhR | random | logistic | SMOTEENN | 0.1843 ± 0.0183 | 0.0754 ± 0.0007 | 0.1048 | 0.5878 ± 0.0474 | 0.2567 ± 0.0028 | 0.3647 |
| NR-AhR | scaffold | forest | none (baseline) | 0.1096 ± 0.0092 | 0.1071 ± 0.0062 | 0.1348 | 0.3599 ± 0.0215 | 0.3555 ± 0.0152 | 0.4509 |
| NR-AhR | scaffold | forest | weights | 0.1895 ± 0.0124 | 0.1021 ± 0.0067 | 0.1348 | 0.5607 ± 0.0283 | 0.3338 ± 0.0191 | 0.4509 |
| NR-AhR | scaffold | forest | undersample | 0.2239 ± 0.0141 | 0.1063 ± 0.0074 | 0.1348 | 0.6382 ± 0.0321 | 0.3464 ± 0.0195 | 0.4509 |
| NR-AhR | scaffold | forest | SMOTEENN | 0.1567 ± 0.0431 | 0.1090 ± 0.0083 | 0.1348 | 0.4809 ± 0.1039 | 0.3538 ± 0.0246 | 0.4509 |
| NR-AhR | scaffold | logistic | none (baseline) | 0.1009 ± 0.0072 | 0.1007 ± 0.0071 | 0.1348 | 0.3366 ± 0.0207 | 0.3343 ± 0.0200 | 0.4509 |
| NR-AhR | scaffold | logistic | weights | 0.1809 ± 0.0339 | 0.1015 ± 0.0056 | 0.1348 | 0.5372 ± 0.0819 | 0.3365 ± 0.0168 | 0.4509 |
| NR-AhR | scaffold | logistic | undersample | 0.2209 ± 0.0127 | 0.1040 ± 0.0073 | 0.1348 | 0.6383 ± 0.0312 | 0.3422 ± 0.0169 | 0.4509 |
| NR-AhR | scaffold | logistic | SMOTEENN | 0.2385 ± 0.0298 | 0.1047 ± 0.0068 | 0.1348 | 0.7036 ± 0.0513 | 0.3457 ± 0.0208 | 0.4509 |
| SR-MMP | random | forest | none (baseline) | 0.0874 ± 0.0002 | 0.0849 ± 0.0018 | 0.1337 | 0.3233 ± 0.0352 | 0.2897 ± 0.0004 | 0.4378 |
| SR-MMP | random | forest | weights | 0.0957 ± 0.0039 | 0.0857 ± 0.0013 | 0.1337 | 0.3329 ± 0.0317 | 0.2902 ± 0.0077 | 0.4378 |
| SR-MMP | random | forest | undersample | 0.1484 ± 0.0073 | 0.0907 ± 0.0012 | 0.1337 | 0.4638 ± 0.0343 | 0.3049 ± 0.0066 | 0.4378 |
| SR-MMP | random | forest | SMOTEENN | 0.1184 ± 0.0059 | 0.0946 ± 0.0010 | 0.1337 | 0.3889 ± 0.0386 | 0.3136 ± 0.0065 | 0.4378 |
| SR-MMP | random | logistic | none (baseline) | 0.0888 ± 0.0011 | 0.0876 ± 0.0011 | 0.1337 | 0.3001 ± 0.0050 | 0.2970 ± 0.0059 | 0.4378 |
| SR-MMP | random | logistic | weights | 0.1207 ± 0.0055 | 0.0873 ± 0.0008 | 0.1337 | 0.3872 ± 0.0144 | 0.2948 ± 0.0057 | 0.4378 |
| SR-MMP | random | logistic | undersample | 0.1590 ± 0.0072 | 0.0949 ± 0.0009 | 0.1337 | 0.4888 ± 0.0181 | 0.3159 ± 0.0063 | 0.4378 |
| SR-MMP | random | logistic | SMOTEENN | 0.1786 ± 0.0142 | 0.0958 ± 0.0033 | 0.1337 | 0.5909 ± 0.0186 | 0.3201 ± 0.0045 | 0.4378 |
| SR-MMP | scaffold | forest | none (baseline) | 0.1447 ± 0.0021 | 0.1378 ± 0.0043 | 0.1717 | 0.4527 ± 0.0050 | 0.4391 ± 0.0132 | 0.5405 |
| SR-MMP | scaffold | forest | weights | 0.1782 ± 0.0324 | 0.1320 ± 0.0067 | 0.1717 | 0.5365 ± 0.0776 | 0.4178 ± 0.0222 | 0.5405 |
| SR-MMP | scaffold | forest | undersample | 0.2241 ± 0.0063 | 0.1358 ± 0.0036 | 0.1717 | 0.6397 ± 0.0137 | 0.4255 ± 0.0109 | 0.5405 |
| SR-MMP | scaffold | forest | SMOTEENN | 0.1720 ± 0.0043 | 0.1396 ± 0.0018 | 0.1717 | 0.5246 ± 0.0095 | 0.4385 ± 0.0040 | 0.5405 |
| SR-MMP | scaffold | logistic | none (baseline) | 0.1357 ± 0.0021 | 0.1327 ± 0.0015 | 0.1717 | 0.4363 ± 0.0199 | 0.4186 ± 0.0044 | 0.5405 |
| SR-MMP | scaffold | logistic | weights | 0.1715 ± 0.0036 | 0.1322 ± 0.0035 | 0.1717 | 0.5215 ± 0.0124 | 0.4158 ± 0.0094 | 0.5405 |
| SR-MMP | scaffold | logistic | undersample | 0.2435 ± 0.0102 | 0.1427 ± 0.0021 | 0.1717 | 0.7303 ± 0.0474 | 0.4438 ± 0.0070 | 0.5405 |
| SR-MMP | scaffold | logistic | SMOTEENN | 0.2648 ± 0.0144 | 0.1397 ± 0.0011 | 0.1717 | 0.7571 ± 0.0224 | 0.4371 ± 0.0058 | 0.5405 |

The pattern is uniform and large. In the raw state, every balancing method inflates both losses relative to the unbalanced baseline, often severely: NR-AR random logistic undersampling reaches a Brier score of 0.2014 against the baseline's 0.0253, and a log loss of 0.5939 against 0.1258. After sigmoid calibration the same configuration sits at 0.0253 and 0.1227 — indistinguishable from the calibrated baseline.

A blunt summary of how badly raw balanced probabilities behave:

**Table 4.2 — Models whose Brier score is no better than the constant-prevalence reference**

| Balancing | Models (of 36) | Raw Brier worse than constant baseline | Sigmoid Brier worse than constant baseline |
| --- | --- | --- | --- |
| none (baseline) | 36 | 0 | 0 |
| weights | 36 | 22 | 0 |
| undersample | 36 | 36 | 0 |
| SMOTEENN | 36 | 26 | 0 |

60 of 144 models beat the constant-prevalence reference on Brier score in the raw state. The 84 that do not are **all** balanced models — every one of the 36 undersampling models, 26 of the 36 SMOTEENN models and 22 of the 36 class-weighting models produce raw probabilities worse, as probabilities, than simply predicting the training prevalence for every compound. None of the 36 unbalanced models falls into this category. After sigmoid calibration, **144 of 144** beat the constant reference.

### 4.3 Calibration deltas against the unbalanced baseline

**Negative is favourable for both metrics in this subsection**, since Brier score and log loss are losses.

**Table 4.3 — Brier score, Δ (method − none), raw state versus sigmoid state**

| Assay | Split | Classifier | Balancing | Mean Δ raw | Δ raw range | Unfavourable intervals, raw | Mean Δ sigmoid | Δ sigmoid range | Unfavourable intervals, sigmoid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NR-AR | random | forest | weights | +0.0671 | [+0.0503, +0.0862] | 3/3 | +0.0006 | [-0.0005, +0.0022] | 1/3 |
| NR-AR | random | forest | undersample | +0.1007 | [+0.0884, +0.1186] | 3/3 | +0.0019 | [+0.0004, +0.0033] | 2/3 |
| NR-AR | random | forest | SMOTEENN | +0.0254 | [+0.0154, +0.0420] | 3/3 | +0.0013 | [+0.0009, +0.0018] | 1/3 |
| NR-AR | random | logistic | weights | +0.0809 | [+0.0711, +0.0958] | 3/3 | -0.0004 | [-0.0022, +0.0009] | 1/3 |
| NR-AR | random | logistic | undersample | +0.1761 | [+0.1649, +0.1950] | 3/3 | +0.0001 | [-0.0017, +0.0020] | 1/3 |
| NR-AR | random | logistic | SMOTEENN | +0.1046 | [+0.0819, +0.1347] | 3/3 | +0.0006 | [-0.0012, +0.0017] | 2/3 |
| NR-AR | scaffold | forest | weights | +0.0233 | [-0.0041, +0.0690] | 2/3 | +0.0003 | [-0.0010, +0.0011] | 1/3 |
| NR-AR | scaffold | forest | undersample | +0.1290 | [+0.1085, +0.1423] | 3/3 | +0.0027 | [+0.0023, +0.0029] | 2/3 |
| NR-AR | scaffold | forest | SMOTEENN | +0.0048 | [-0.0011, +0.0083] | 2/3 | +0.0013 | [+0.0006, +0.0026] | 1/3 |
| NR-AR | scaffold | logistic | weights | +0.0428 | [+0.0189, +0.0728] | 3/3 | -0.0089 | [-0.0221, -0.0007] | 0/3 |
| NR-AR | scaffold | logistic | undersample | +0.1968 | [+0.1107, +0.2495] | 3/3 | -0.0027 | [-0.0110, +0.0028] | 0/3 |
| NR-AR | scaffold | logistic | SMOTEENN | +0.0703 | [+0.0513, +0.0972] | 3/3 | -0.0047 | [-0.0161, +0.0033] | 1/3 |
| NR-AhR | random | forest | weights | +0.0077 | [+0.0018, +0.0116] | 2/3 | -0.0021 | [-0.0036, -0.0009] | 0/3 |
| NR-AhR | random | forest | undersample | +0.0848 | [+0.0712, +0.0946] | 3/3 | +0.0037 | [+0.0022, +0.0048] | 0/3 |
| NR-AhR | random | forest | SMOTEENN | +0.0321 | [+0.0165, +0.0411] | 3/3 | +0.0049 | [+0.0015, +0.0087] | 1/3 |
| NR-AhR | random | logistic | weights | +0.0446 | [+0.0415, +0.0482] | 3/3 | +0.0003 | [+0.0002, +0.0004] | 0/3 |
| NR-AhR | random | logistic | undersample | +0.0916 | [+0.0864, +0.0985] | 3/3 | +0.0046 | [+0.0039, +0.0052] | 3/3 |
| NR-AhR | random | logistic | SMOTEENN | +0.1127 | [+0.0904, +0.1250] | 3/3 | +0.0042 | [+0.0034, +0.0047] | 3/3 |
| NR-AhR | scaffold | forest | weights | +0.0799 | [+0.0699, +0.0879] | 3/3 | -0.0050 | [-0.0074, -0.0018] | 0/3 |
| NR-AhR | scaffold | forest | undersample | +0.1142 | [+0.1104, +0.1218] | 3/3 | -0.0008 | [-0.0025, +0.0017] | 0/3 |
| NR-AhR | scaffold | forest | SMOTEENN | +0.0470 | [+0.0205, +0.0881] | 3/3 | +0.0018 | [-0.0036, +0.0087] | 1/3 |
| NR-AhR | scaffold | logistic | weights | +0.0800 | [+0.0404, +0.1029] | 3/3 | +0.0008 | [-0.0004, +0.0029] | 1/3 |
| NR-AhR | scaffold | logistic | undersample | +0.1200 | [+0.1143, +0.1282] | 3/3 | +0.0033 | [-0.0006, +0.0075] | 1/3 |
| NR-AhR | scaffold | logistic | SMOTEENN | +0.1376 | [+0.1042, +0.1600] | 3/3 | +0.0041 | [+0.0029, +0.0048] | 3/3 |
| SR-MMP | random | forest | weights | +0.0084 | [+0.0059, +0.0130] | 2/3 | +0.0008 | [-0.0013, +0.0041] | 1/3 |
| SR-MMP | random | forest | undersample | +0.0611 | [+0.0551, +0.0694] | 3/3 | +0.0058 | [+0.0029, +0.0077] | 2/3 |
| SR-MMP | random | forest | SMOTEENN | +0.0310 | [+0.0272, +0.0380] | 3/3 | +0.0097 | [+0.0079, +0.0126] | 3/3 |
| SR-MMP | random | logistic | weights | +0.0319 | [+0.0264, +0.0392] | 3/3 | -0.0003 | [-0.0008, -0.0001] | 0/3 |
| SR-MMP | random | logistic | undersample | +0.0702 | [+0.0627, +0.0792] | 3/3 | +0.0072 | [+0.0056, +0.0080] | 3/3 |
| SR-MMP | random | logistic | SMOTEENN | +0.0899 | [+0.0742, +0.1037] | 3/3 | +0.0082 | [+0.0063, +0.0111] | 3/3 |
| SR-MMP | scaffold | forest | weights | +0.0335 | [-0.0006, +0.0572] | 2/3 | -0.0058 | [-0.0080, -0.0028] | 0/3 |
| SR-MMP | scaffold | forest | undersample | +0.0794 | [+0.0740, +0.0847] | 3/3 | -0.0021 | [-0.0039, -0.0010] | 0/3 |
| SR-MMP | scaffold | forest | SMOTEENN | +0.0273 | [+0.0232, +0.0302] | 2/3 | +0.0018 | [-0.0012, +0.0058] | 0/3 |
| SR-MMP | scaffold | logistic | weights | +0.0358 | [+0.0335, +0.0399] | 3/3 | -0.0005 | [-0.0033, +0.0010] | 0/3 |
| SR-MMP | scaffold | logistic | undersample | +0.1078 | [+0.1004, +0.1212] | 3/3 | +0.0100 | [+0.0078, +0.0131] | 3/3 |
| SR-MMP | scaffold | logistic | SMOTEENN | +0.1291 | [+0.1137, +0.1450] | 3/3 | +0.0070 | [+0.0062, +0.0080] | 3/3 |

**Table 4.4 — Log loss, Δ (method − none), raw state versus sigmoid state**

| Assay | Split | Classifier | Balancing | Mean Δ raw | Δ raw range | Unfavourable intervals, raw | Mean Δ sigmoid | Δ sigmoid range | Unfavourable intervals, sigmoid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NR-AR | random | forest | weights | +0.1530 | [-0.0575, +0.2853] | 2/3 | +0.0036 | [+0.0003, +0.0091] | 1/3 |
| NR-AR | random | forest | undersample | +0.2343 | [+0.0666, +0.3536] | 2/3 | +0.0095 | [+0.0074, +0.0131] | 2/3 |
| NR-AR | random | forest | SMOTEENN | +0.0288 | [-0.1633, +0.1670] | 2/3 | +0.0064 | [+0.0061, +0.0067] | 2/3 |
| NR-AR | random | logistic | weights | +0.2526 | [+0.2231, +0.2917] | 3/3 | -0.0016 | [-0.0053, +0.0006] | 0/3 |
| NR-AR | random | logistic | undersample | +0.4681 | [+0.4386, +0.5131] | 3/3 | +0.0010 | [-0.0026, +0.0074] | 1/3 |
| NR-AR | random | logistic | SMOTEENN | +0.3000 | [+0.2400, +0.3720] | 3/3 | +0.0027 | [-0.0026, +0.0062] | 1/3 |
| NR-AR | scaffold | forest | weights | +0.0936 | [+0.0037, +0.2295] | 2/3 | +0.0012 | [-0.0003, +0.0025] | 0/3 |
| NR-AR | scaffold | forest | undersample | +0.3643 | [+0.3212, +0.3972] | 3/3 | +0.0087 | [+0.0073, +0.0107] | 1/3 |
| NR-AR | scaffold | forest | SMOTEENN | +0.0443 | [+0.0292, +0.0641] | 2/3 | +0.0034 | [+0.0023, +0.0042] | 0/3 |
| NR-AR | scaffold | logistic | weights | +0.1369 | [+0.0775, +0.2221] | 3/3 | -0.0249 | [-0.0607, -0.0024] | 0/3 |
| NR-AR | scaffold | logistic | undersample | +0.5910 | [+0.3090, +0.8263] | 3/3 | -0.0103 | [-0.0360, +0.0087] | 0/3 |
| NR-AR | scaffold | logistic | SMOTEENN | +0.2804 | [+0.2077, +0.3624] | 3/3 | -0.0147 | [-0.0457, +0.0089] | 0/3 |
| NR-AhR | random | forest | weights | +0.0140 | [+0.0041, +0.0293] | 1/3 | -0.0105 | [-0.0116, -0.0086] | 0/3 |
| NR-AhR | random | forest | undersample | +0.2095 | [+0.1826, +0.2385] | 3/3 | +0.0043 | [+0.0013, +0.0093] | 0/3 |
| NR-AhR | random | forest | SMOTEENN | +0.0940 | [+0.0404, +0.1367] | 3/3 | +0.0074 | [-0.0006, +0.0167] | 0/3 |
| NR-AhR | random | logistic | weights | +0.1229 | [+0.1170, +0.1320] | 3/3 | +0.0005 | [-0.0013, +0.0016] | 0/3 |
| NR-AhR | random | logistic | undersample | +0.2445 | [+0.2308, +0.2597] | 3/3 | +0.0119 | [+0.0088, +0.0151] | 2/3 |
| NR-AhR | random | logistic | SMOTEENN | +0.3407 | [+0.2849, +0.3724] | 3/3 | +0.0115 | [+0.0071, +0.0138] | 3/3 |
| NR-AhR | scaffold | forest | weights | +0.2008 | [+0.1779, +0.2193] | 3/3 | -0.0218 | [-0.0333, -0.0041] | 0/3 |
| NR-AhR | scaffold | forest | undersample | +0.2782 | [+0.2677, +0.2964] | 3/3 | -0.0091 | [-0.0182, +0.0073] | 0/3 |
| NR-AhR | scaffold | forest | SMOTEENN | +0.1210 | [+0.0545, +0.2203] | 3/3 | -0.0017 | [-0.0206, +0.0253] | 1/3 |
| NR-AhR | scaffold | logistic | weights | +0.2007 | [+0.1047, +0.2516] | 3/3 | +0.0022 | [+0.0001, +0.0062] | 0/3 |
| NR-AhR | scaffold | logistic | undersample | +0.3017 | [+0.2866, +0.3191] | 3/3 | +0.0079 | [-0.0037, +0.0176] | 1/3 |
| NR-AhR | scaffold | logistic | SMOTEENN | +0.3670 | [+0.3165, +0.4052] | 3/3 | +0.0114 | [+0.0082, +0.0139] | 3/3 |
| SR-MMP | random | forest | weights | +0.0095 | [-0.0048, +0.0244] | 1/3 | +0.0005 | [-0.0053, +0.0097] | 0/3 |
| SR-MMP | random | forest | undersample | +0.1404 | [+0.1285, +0.1501] | 3/3 | +0.0152 | [+0.0080, +0.0220] | 1/3 |
| SR-MMP | random | forest | SMOTEENN | +0.0656 | [+0.0464, +0.0775] | 1/3 | +0.0239 | [+0.0180, +0.0314] | 2/3 |
| SR-MMP | random | logistic | weights | +0.0871 | [+0.0772, +0.1038] | 3/3 | -0.0022 | [-0.0031, -0.0005] | 0/3 |
| SR-MMP | random | logistic | undersample | +0.1887 | [+0.1764, +0.2100] | 3/3 | +0.0189 | [+0.0171, +0.0212] | 3/3 |
| SR-MMP | random | logistic | SMOTEENN | +0.2908 | [+0.2747, +0.3021] | 3/3 | +0.0231 | [+0.0157, +0.0327] | 3/3 |
| SR-MMP | scaffold | forest | weights | +0.0838 | [-0.0041, +0.1378] | 2/3 | -0.0213 | [-0.0297, -0.0111] | 0/3 |
| SR-MMP | scaffold | forest | undersample | +0.1869 | [+0.1810, +0.1973] | 3/3 | -0.0135 | [-0.0213, -0.0080] | 0/3 |
| SR-MMP | scaffold | forest | SMOTEENN | +0.0718 | [+0.0690, +0.0774] | 3/3 | -0.0006 | [-0.0075, +0.0125] | 0/3 |
| SR-MMP | scaffold | logistic | weights | +0.0852 | [+0.0600, +0.1077] | 2/3 | -0.0027 | [-0.0096, +0.0013] | 0/3 |
| SR-MMP | scaffold | logistic | undersample | +0.2939 | [+0.2544, +0.3536] | 3/3 | +0.0252 | [+0.0204, +0.0345] | 2/3 |
| SR-MMP | scaffold | logistic | SMOTEENN | +0.3208 | [+0.2962, +0.3531] | 3/3 | +0.0186 | [+0.0149, +0.0224] | 2/3 |

The magnitudes collapse once calibration is applied. Over all 108 Brier comparisons the largest raw delta is +0.2495 and the mean absolute delta is 0.0745; in the sigmoid state the largest absolute delta is 0.0221 and the mean absolute delta is 0.0038. For log loss the corresponding figures are +0.8263 and 0.2061 in the raw state against 0.0607 and 0.0114 after calibration. In other words, calibration on a separate, uncontaminated partition removes most — not all — of the probability distortion that balancing introduces.

What survives calibration is small and mixed in direction. Several groups retain intervals excluding zero unfavourably after calibration (for example SR-MMP/scaffold/logistic/undersampling, mean Brier delta +0.0100 with 3 of 3 intervals excluding zero unfavourably), and a few retain small favourable ones (SR-MMP/scaffold/forest/weights, mean Brier delta −0.0058 with 2 of 3 intervals excluding zero favourably). Neither direction is consistent enough across assays, classifiers and seeds to support a general statement.

### 4.4 What the reliability curves show

The committed reliability data (`outputs/full/evaluation/<assay>/<split_kind>/<seed>/<classifier>/reliability.csv`, plotted as `reliability.png`) place predictions into ten fixed bins of width 0.1 and record, per bin, the count, the number of positives, the mean predicted probability and the observed active fraction. The two statistics below are derived from those files: the **weighted gap** is the count-weighted mean of |mean predicted − observed fraction| over the non-empty bins, pooled across the three seeds, and the **signed bias** is the same average without the absolute value, so a positive value means the model predicts higher probabilities than the observed rate.

**Table 4.5 — Reliability gap per group, pooled over seeds**

| Assay | Split | Classifier | Balancing | Weighted gap, raw | Weighted gap, sigmoid | Signed bias, raw | Signed bias, sigmoid |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NR-AR | random | forest | none (baseline) | 0.0139 | 0.0042 | -0.0021 | +0.0006 |
| NR-AR | random | forest | weights | 0.2450 | 0.0067 | +0.2432 | +0.0005 |
| NR-AR | random | forest | undersample | 0.3033 | 0.0109 | +0.3025 | -0.0011 |
| NR-AR | random | forest | SMOTEENN | 0.1332 | 0.0083 | +0.1305 | -0.0003 |
| NR-AR | random | logistic | none (baseline) | 0.0147 | 0.0084 | +0.0025 | -0.0001 |
| NR-AR | random | logistic | weights | 0.2641 | 0.0072 | +0.2630 | -0.0003 |
| NR-AR | random | logistic | undersample | 0.4125 | 0.0110 | +0.4095 | -0.0013 |
| NR-AR | random | logistic | SMOTEENN | 0.2814 | 0.0087 | +0.2814 | +0.0001 |
| NR-AR | scaffold | forest | none (baseline) | 0.0138 | 0.0105 | -0.0117 | +0.0034 |
| NR-AR | scaffold | forest | weights | 0.1200 | 0.0144 | +0.1149 | +0.0035 |
| NR-AR | scaffold | forest | undersample | 0.3517 | 0.0213 | +0.3510 | +0.0026 |
| NR-AR | scaffold | forest | SMOTEENN | 0.0651 | 0.0162 | +0.0605 | +0.0028 |
| NR-AR | scaffold | logistic | none (baseline) | 0.0291 | 0.0248 | +0.0115 | -0.0034 |
| NR-AR | scaffold | logistic | weights | 0.1573 | 0.0202 | +0.1547 | +0.0009 |
| NR-AR | scaffold | logistic | undersample | 0.3570 | 0.0231 | +0.3570 | +0.0033 |
| NR-AR | scaffold | logistic | SMOTEENN | 0.1746 | 0.0241 | +0.1613 | +0.0017 |
| NR-AhR | random | forest | none (baseline) | 0.0315 | 0.0159 | +0.0043 | +0.0058 |
| NR-AhR | random | forest | weights | 0.0879 | 0.0152 | +0.0879 | +0.0025 |
| NR-AhR | random | forest | undersample | 0.2506 | 0.0171 | +0.2506 | +0.0033 |
| NR-AhR | random | forest | SMOTEENN | 0.1331 | 0.0145 | +0.1331 | +0.0020 |
| NR-AhR | random | logistic | none (baseline) | 0.0210 | 0.0117 | +0.0104 | +0.0033 |
| NR-AhR | random | logistic | weights | 0.1649 | 0.0127 | +0.1649 | +0.0033 |
| NR-AhR | random | logistic | undersample | 0.2600 | 0.0184 | +0.2600 | +0.0033 |
| NR-AhR | random | logistic | SMOTEENN | 0.2384 | 0.0120 | +0.2384 | +0.0020 |
| NR-AhR | scaffold | forest | none (baseline) | 0.0354 | 0.0259 | -0.0117 | +0.0072 |
| NR-AhR | scaffold | forest | weights | 0.2867 | 0.0221 | +0.2855 | +0.0067 |
| NR-AhR | scaffold | forest | undersample | 0.3312 | 0.0250 | +0.3309 | +0.0096 |
| NR-AhR | scaffold | forest | SMOTEENN | 0.1999 | 0.0271 | +0.1998 | +0.0087 |
| NR-AhR | scaffold | logistic | none (baseline) | 0.0253 | 0.0198 | -0.0138 | +0.0057 |
| NR-AhR | scaffold | logistic | weights | 0.2491 | 0.0180 | +0.2491 | +0.0075 |
| NR-AhR | scaffold | logistic | undersample | 0.3189 | 0.0250 | +0.3189 | +0.0093 |
| NR-AhR | scaffold | logistic | SMOTEENN | 0.3138 | 0.0201 | +0.3138 | +0.0072 |
| SR-MMP | random | forest | none (baseline) | 0.0311 | 0.0227 | -0.0080 | -0.0010 |
| SR-MMP | random | forest | weights | 0.0833 | 0.0194 | +0.0799 | +0.0003 |
| SR-MMP | random | forest | undersample | 0.2106 | 0.0257 | +0.2099 | -0.0009 |
| SR-MMP | random | forest | SMOTEENN | 0.1221 | 0.0223 | +0.1217 | -0.0026 |
| SR-MMP | random | logistic | none (baseline) | 0.0317 | 0.0205 | +0.0019 | -0.0001 |
| SR-MMP | random | logistic | weights | 0.1535 | 0.0212 | +0.1535 | -0.0007 |
| SR-MMP | random | logistic | undersample | 0.2326 | 0.0217 | +0.2326 | +0.0000 |
| SR-MMP | random | logistic | SMOTEENN | 0.2126 | 0.0210 | +0.2063 | -0.0017 |
| SR-MMP | scaffold | forest | none (baseline) | 0.0568 | 0.0268 | -0.0448 | +0.0122 |
| SR-MMP | scaffold | forest | weights | 0.1909 | 0.0261 | +0.1890 | +0.0107 |
| SR-MMP | scaffold | forest | undersample | 0.2935 | 0.0306 | +0.2905 | +0.0112 |
| SR-MMP | scaffold | forest | SMOTEENN | 0.1812 | 0.0278 | +0.1812 | +0.0150 |
| SR-MMP | scaffold | logistic | none (baseline) | 0.0526 | 0.0256 | -0.0296 | +0.0127 |
| SR-MMP | scaffold | logistic | weights | 0.1756 | 0.0307 | +0.1756 | +0.0131 |
| SR-MMP | scaffold | logistic | undersample | 0.2801 | 0.0311 | +0.2801 | +0.0090 |
| SR-MMP | scaffold | logistic | SMOTEENN | 0.3312 | 0.0280 | +0.3312 | +0.0179 |

**Table 4.6 — The same statistics averaged by balancing method (48 group-level values, 12 per method)**

| Balancing | Mean weighted gap, raw | Mean weighted gap, sigmoid | Mean signed bias, raw | Mean signed bias, sigmoid |
| --- | --- | --- | --- | --- |
| none (baseline) | 0.0297 | 0.0181 | -0.0076 | +0.0039 |
| weights | 0.1815 | 0.0178 | +0.1801 | +0.0040 |
| undersample | 0.3002 | 0.0217 | +0.2995 | +0.0040 |
| SMOTEENN | 0.1989 | 0.0192 | +0.1966 | +0.0044 |

The curves tell a single consistent story.

- **Raw balanced models are systematically over-confident toward the positive class.** The signed bias is positive and large for all three balancing methods in the raw state (mean +0.1801 for class weights, +0.2995 for undersampling, +0.1966 for SMOTEENN), meaning the reliability curve sits well below the diagonal: a bin whose compounds are assigned around 0.5 contains far fewer than half actives. The unbalanced model is nearly unbiased in the raw state (mean signed bias −0.0076).
- **Sigmoid calibration moves every group onto the diagonal.** The weighted gap falls for **48 of 48** groups, and the post-calibration gap is between 0.0042 and 0.0311 everywhere, with signed bias within ±0.018. The large method-to-method differences in the raw state (mean weighted gap 0.0297 for `none` against 0.3002 for undersampling) essentially disappear after calibration (0.0181 against 0.0217).
- **The residual post-calibration gap is larger on scaffold splits than on random splits** — in all 24 matched assay × classifier × method cells, mean weighted gap 0.0235 against 0.0149. This is consistent with calibration being fitted on a scaffold calibration partition whose prevalence differs from the scaffold test partition's (Table 2.2), so the fitted sigmoid is slightly mismatched to the test distribution.

A concrete example makes the shape of the effect visible. This is NR-AhR, scaffold split, seed 17, random forest — the figure is `outputs/full/evaluation/NR-AhR/scaffold/17/forest/reliability.png`.

**Table 4.7 — Reliability bins for one group, illustrating the raw distortion and its repair**

| Balancing | State | Bin | n | Positives | Mean predicted | Observed active fraction | Exploratory (<20 positives) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none (baseline) | raw | [0.0, 0.1) | 405 | 14 | 0.0649 | 0.0346 | yes |
| none (baseline) | raw | [0.1, 0.2) | 463 | 43 | 0.1410 | 0.0929 | no |
| none (baseline) | raw | [0.2, 0.3) | 271 | 81 | 0.2398 | 0.2989 | no |
| none (baseline) | raw | [0.3, 0.4) | 102 | 56 | 0.3404 | 0.5490 | no |
| none (baseline) | raw | [0.4, 0.5) | 14 | 14 | 0.4341 | 1.0000 | yes |
| weights | raw | [0.0, 0.1) | 2 | 0 | 0.0814 | 0.0000 | yes |
| weights | raw | [0.1, 0.2) | 74 | 3 | 0.1563 | 0.0405 | yes |
| weights | raw | [0.2, 0.3) | 151 | 3 | 0.2560 | 0.0199 | yes |
| weights | raw | [0.3, 0.4) | 239 | 9 | 0.3537 | 0.0377 | yes |
| weights | raw | [0.4, 0.5) | 259 | 22 | 0.4484 | 0.0849 | no |
| weights | raw | [0.5, 0.6) | 237 | 41 | 0.5476 | 0.1730 | no |
| weights | raw | [0.6, 0.7) | 174 | 59 | 0.6477 | 0.3391 | no |
| weights | raw | [0.7, 0.8) | 88 | 44 | 0.7444 | 0.5000 | no |
| weights | raw | [0.8, 0.9) | 31 | 27 | 0.8254 | 0.8710 | no |
| weights | sigmoid | [0.0, 0.1) | 658 | 33 | 0.0418 | 0.0502 | no |
| weights | sigmoid | [0.1, 0.2) | 256 | 33 | 0.1460 | 0.1289 | no |
| weights | sigmoid | [0.2, 0.3) | 117 | 33 | 0.2465 | 0.2821 | no |
| weights | sigmoid | [0.3, 0.4) | 97 | 36 | 0.3484 | 0.3711 | no |
| weights | sigmoid | [0.4, 0.5) | 48 | 18 | 0.4433 | 0.3750 | yes |
| weights | sigmoid | [0.5, 0.6) | 43 | 25 | 0.5454 | 0.5814 | no |
| weights | sigmoid | [0.6, 0.7) | 31 | 25 | 0.6479 | 0.8065 | no |
| weights | sigmoid | [0.7, 0.8) | 5 | 5 | 0.7177 | 1.0000 | yes |

Reading the table: the unbalanced raw model confines its predictions to the 0–0.5 range and tracks the observed rate reasonably within that range. The class-weighted raw model spreads predictions across the whole 0–0.9 range, but a bin predicted at 0.448 contains only 8.5% actives and a bin predicted at 0.648 contains 33.9% — the curve is far below the diagonal. After the sigmoid, the same model's bins agree closely with the observed rates. Note that the ranking of compounds is identical in the second and third blocks of the table; only the numbers attached to them changed.

### 4.5 Cases where calibration made things worse

The proposal commits to reporting these, and they exist.

**Table 4.8 — All models where the sigmoid state is worse than the raw state on Brier score or log loss**

| Assay | Split | Seed | Classifier | Balancing | Brier raw | Brier sigmoid | Log loss raw | Log loss sigmoid | Measure made worse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NR-AR | random | 17 | logistic | none (baseline) | 0.02690 | 0.02749 | 0.13550 | 0.12513 | Brier only |
| NR-AR | random | 2026 | logistic | none (baseline) | 0.02704 | 0.02705 | 0.12955 | 0.12914 | Brier only |
| NR-AR | scaffold | 42 | logistic | none (baseline) | 0.04110 | 0.04124 | 0.17812 | 0.17523 | Brier only |
| NR-AR | scaffold | 2026 | logistic | none (baseline) | 0.05097 | 0.05827 | 0.19619 | 0.21499 | **both** |
| NR-AhR | scaffold | 17 | logistic | none (baseline) | 0.10671 | 0.10677 | 0.35413 | 0.35275 | Brier only |
| NR-AhR | scaffold | 42 | forest | none (baseline) | 0.09904 | 0.10032 | 0.33516 | 0.34083 | **both** |
| NR-AhR | scaffold | 42 | logistic | none (baseline) | 0.09278 | 0.09281 | 0.31378 | 0.31314 | Brier only |
| NR-AhR | scaffold | 2026 | forest | none (baseline) | 0.11404 | 0.11244 | 0.37090 | 0.37122 | log loss only |

8 of 144 models are worse after calibration on at least one of the two measures — 7 on Brier score, 3 on log loss, 2 on both — and **all of them are unbalanced (`none`) models** — the models that were already close to well calibrated in the raw state, and therefore had nothing to gain and a little to lose from fitting a two-parameter map on a finite calibration partition. The largest single degradation is NR-AR/scaffold/seed 2026/logistic, where Brier rises from 0.05097 to 0.05827 and log loss from 0.19619 to 0.21499; this is the same NR-AR scaffold logistic group that shows the largest seed-to-seed instability in Section 3. No balanced model was made worse by calibration on either measure.

---

## 5. Scaffold versus random: does any benefit survive the harder split?

**Table 5.1 — Scaffold minus random, paired within assay, classifier and balancing method (seed means)**

| Assay | Classifier | Balancing | AP random | AP scaffold | Δ AP (scaffold − random) | AP SD random | AP SD scaffold | AP / prevalence random | AP / prevalence scaffold | Δ ROC AUC | Δ recall @10% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NR-AR | forest | none (baseline) | 0.550 | 0.497 | -0.052 | 0.063 | 0.149 | 12.75 | 9.26 | -0.018 | +0.009 |
| NR-AR | forest | weights | 0.557 | 0.493 | -0.064 | 0.068 | 0.154 | 12.93 | 9.19 | -0.020 | -0.029 |
| NR-AR | forest | undersample | 0.556 | 0.502 | -0.054 | 0.076 | 0.168 | 12.90 | 9.35 | -0.013 | -0.035 |
| NR-AR | forest | SMOTEENN | 0.552 | 0.491 | -0.061 | 0.068 | 0.152 | 12.80 | 9.14 | -0.004 | -0.022 |
| NR-AR | logistic | none (baseline) | 0.537 | 0.394 | -0.143 | 0.069 | 0.121 | 12.46 | 7.35 | -0.022 | -0.086 |
| NR-AR | logistic | weights | 0.542 | 0.447 | -0.095 | 0.074 | 0.190 | 12.57 | 8.33 | -0.014 | -0.019 |
| NR-AR | logistic | undersample | 0.544 | 0.385 | -0.159 | 0.080 | 0.217 | 12.61 | 7.17 | +0.017 | -0.050 |
| NR-AR | logistic | SMOTEENN | 0.537 | 0.389 | -0.148 | 0.070 | 0.238 | 12.46 | 7.24 | -0.024 | -0.038 |
| NR-AhR | forest | none (baseline) | 0.566 | 0.469 | -0.097 | 0.037 | 0.059 | 4.76 | 3.03 | -0.077 | -0.178 |
| NR-AhR | forest | weights | 0.588 | 0.510 | -0.078 | 0.022 | 0.046 | 4.95 | 3.30 | -0.057 | -0.146 |
| NR-AhR | forest | undersample | 0.529 | 0.468 | -0.061 | 0.016 | 0.041 | 4.45 | 3.02 | -0.061 | -0.112 |
| NR-AhR | forest | SMOTEENN | 0.503 | 0.442 | -0.061 | 0.020 | 0.049 | 4.23 | 2.85 | -0.062 | -0.119 |
| NR-AhR | logistic | none (baseline) | 0.586 | 0.514 | -0.072 | 0.030 | 0.043 | 4.93 | 3.32 | -0.056 | -0.117 |
| NR-AhR | logistic | weights | 0.581 | 0.502 | -0.079 | 0.031 | 0.059 | 4.89 | 3.24 | -0.060 | -0.107 |
| NR-AhR | logistic | undersample | 0.544 | 0.477 | -0.067 | 0.031 | 0.042 | 4.57 | 3.08 | -0.053 | -0.104 |
| NR-AhR | logistic | SMOTEENN | 0.535 | 0.474 | -0.061 | 0.029 | 0.045 | 4.50 | 3.06 | -0.064 | -0.116 |
| SR-MMP | forest | none (baseline) | 0.662 | 0.477 | -0.185 | 0.010 | 0.053 | 4.16 | 2.31 | -0.118 | -0.164 |
| SR-MMP | forest | weights | 0.644 | 0.516 | -0.128 | 0.014 | 0.058 | 4.05 | 2.49 | -0.086 | -0.160 |
| SR-MMP | forest | undersample | 0.621 | 0.497 | -0.125 | 0.003 | 0.072 | 3.91 | 2.40 | -0.087 | -0.135 |
| SR-MMP | forest | SMOTEENN | 0.578 | 0.465 | -0.112 | 0.031 | 0.038 | 3.64 | 2.25 | -0.093 | -0.151 |
| SR-MMP | logistic | none (baseline) | 0.636 | 0.504 | -0.133 | 0.014 | 0.051 | 4.01 | 2.43 | -0.075 | -0.166 |
| SR-MMP | logistic | weights | 0.638 | 0.508 | -0.131 | 0.011 | 0.062 | 4.02 | 2.45 | -0.073 | -0.158 |
| SR-MMP | logistic | undersample | 0.573 | 0.439 | -0.134 | 0.005 | 0.046 | 3.61 | 2.12 | -0.097 | -0.172 |
| SR-MMP | logistic | SMOTEENN | 0.566 | 0.467 | -0.099 | 0.038 | 0.057 | 3.56 | 2.26 | -0.081 | -0.148 |

### 5.1 The scaffold split is harder, without exception

Across all 24 assay × classifier × method cells:

- average precision is lower on the scaffold split in **24/24** cells, mean change −0.0999;
- the prevalence-normalised ratio (average precision ÷ test prevalence), which corrects for the higher prevalence of the scaffold test partitions, is lower in **24/24** cells — so the drop is not an artefact of differing prevalence, it is larger than the raw numbers suggest;
- ROC AUC is lower, mean change −0.0541;
- recall within the fixed top-10% screening budget is lower, mean change −0.1051;
- the across-seed standard deviation of average precision is larger in **24/24** cells, rising from a mean of 0.0379 on random splits to 0.0921 on scaffold splits.

This matches the proposal's stated expectation that performance on unfamiliar scaffolds would be weaker *and* less stable, and it is the most robust single result in the study. The effect is largest on SR-MMP (forest, unbalanced: average precision 0.662 → 0.477, ratio 4.16 → 2.31) and on NR-AR with logistic regression (0.537 → 0.394).

### 5.2 Does any balancing benefit survive?

The honest answer is: **one narrow benefit survives, and it is actually clearer on the scaffold split than on the random split, but it does not hold across all three assays.**

- **Class weighting with the random forest.** On the random split this combination is essentially neutral (mean average-precision delta +0.0040, no interval excluding zero favourably, one unfavourably). On the scaffold split it is the strongest result in the study: mean delta +0.0253, 4 of 9 intervals excluding zero favourably, none unfavourably, and the gain appears in both NR-AhR and SR-MMP. It does not appear in NR-AR.
- **Class weighting with logistic regression.** Neutral on the random split (mean delta +0.0006), positive on average on the scaffold split (+0.0149) but driven almost entirely by the single unstable NR-AR/seed 2026 comparison; 2 intervals excluding zero favourably against 1 unfavourably.
- **Random undersampling and SMOTEENN.** Both are harmful on average under both split kinds. Their scaffold-split averages are less negative than their random-split averages (undersampling −0.0148 versus −0.0284; SMOTEENN −0.0213 versus −0.0445), but "less harmful on the harder split" is not a benefit surviving, and neither method accumulates favourable intervals anywhere near the number of unfavourable ones.

Two cautions attach to the surviving benefit. First, Section 2.3: the scaffold test set is the remaining scaffold distribution after two giant groups are placed in training, so "scaffold split" here does not mean a uniform draw over new scaffolds. Second, those four deltas span +0.0463 to +0.0618 average precision, on two of three assays with three overlapping seeds, against unadjusted intervals; that is a direction worth following up, not an established effect.

---

## 6. Similarity analysis

Nearest-neighbour Tanimoto thresholds were fixed before the results were viewed, in four bins: [0.0, 0.4), [0.4, 0.6), [0.6, 0.8) and [0.8, 1.0]. Each test compound is assigned to the bin containing its maximum Tanimoto similarity to any training compound, and metrics are recomputed within each bin (`outputs/full/similarity_metrics.csv`).

### 6.1 Bin composition

**Table 6.1 — Compounds and positives per similarity bin**

| Assay | Split | Tanimoto bin | n (min–max over seeds) | Positives (min–max) | Prevalence (seed mean) | Seeds flagged exploratory |
| --- | --- | --- | --- | --- | --- | --- |
| NR-AR | random | [0.0, 0.4) | 267–300 | 7–11 | 0.0318 | 3/3 |
| NR-AR | random | [0.4, 0.6) | 534–536 | 14–16 | 0.0274 | 3/3 |
| NR-AR | random | [0.6, 0.8) | 401–431 | 27–32 | 0.0712 | 0/3 |
| NR-AR | random | [0.8, 1.0] | 156–172 | 6–8 | 0.0433 | 3/3 |
| NR-AR | scaffold | [0.0, 0.4) | 666–696 | 21–33 | 0.0376 | 0/3 |
| NR-AR | scaffold | [0.4, 0.6) | 530–554 | 25–42 | 0.0595 | 0/3 |
| NR-AR | scaffold | [0.6, 0.8) | 150–159 | 12–21 | 0.1096 | 2/3 |
| NR-AR | scaffold | [0.8, 1.0] | 12–25 | 0–1 | 0.0133 | 3/3 |
| NR-AhR | random | [0.0, 0.4) | 260–295 | 25–36 | 0.1142 | 0/3 |
| NR-AhR | random | [0.4, 0.6) | 485–535 | 66–73 | 0.1344 | 0/3 |
| NR-AhR | random | [0.6, 0.8) | 330–348 | 33–46 | 0.1129 | 0/3 |
| NR-AhR | random | [0.8, 1.0] | 124–131 | 7–14 | 0.0838 | 3/3 |
| NR-AhR | scaffold | [0.0, 0.4) | 627–659 | 82–102 | 0.1435 | 0/3 |
| NR-AhR | scaffold | [0.4, 0.6) | 467–489 | 68–90 | 0.1700 | 0/3 |
| NR-AhR | scaffold | [0.6, 0.8) | 108–138 | 16–24 | 0.1642 | 2/3 |
| NR-AhR | scaffold | [0.8, 1.0] | 13–23 | 1–3 | 0.1134 | 3/3 |
| SR-MMP | random | [0.0, 0.4) | 232–254 | 32–43 | 0.1575 | 0/3 |
| SR-MMP | random | [0.4, 0.6) | 426–445 | 64–77 | 0.1587 | 0/3 |
| SR-MMP | random | [0.6, 0.8) | 290–321 | 43–52 | 0.1581 | 0/3 |
| SR-MMP | random | [0.8, 1.0] | 121–128 | 18–25 | 0.1646 | 2/3 |
| SR-MMP | scaffold | [0.0, 0.4) | 579–590 | 89–103 | 0.1669 | 0/3 |
| SR-MMP | scaffold | [0.4, 0.6) | 402–428 | 104–113 | 0.2654 | 0/3 |
| SR-MMP | scaffold | [0.6, 0.8) | 95–109 | 17–26 | 0.2071 | 1/3 |
| SR-MMP | scaffold | [0.8, 1.0] | 13–15 | 1–3 | 0.1470 | 3/3 |

The bins are far from balanced, and the imbalance differs sharply by split kind. Under random splitting, most test compounds sit in the two middle bins and few are far from training data. Under scaffold splitting, the mass shifts down: the [0.0, 0.4) bin holds 579–696 compounds, while the high-similarity [0.8, 1.0] bin collapses to 12–25 compounds. Every scaffold-split [0.8, 1.0] bin is **exploratory** in every seed, containing 0–3 positives.

### 6.2 Average precision across the bins

**Table 6.2 — NR-AR, average precision by similarity bin (seed mean)**

Random split:

| Classifier | Balancing | [0.0, 0.4) | [0.4, 0.6) | [0.6, 0.8) | [0.8, 1.0] |
| --- | --- | --- | --- | --- | --- |
| forest | none (baseline) | 0.157 | 0.456 | 0.686 | 0.677 |
| forest | weights | 0.155 | 0.458 | 0.688 | 0.727 |
| forest | undersample | 0.130 | 0.452 | 0.695 | 0.716 |
| forest | SMOTEENN | 0.150 | 0.453 | 0.696 | 0.696 |
| logistic | none (baseline) | 0.137 | 0.408 | 0.685 | 0.710 |
| logistic | weights | 0.125 | 0.431 | 0.683 | 0.724 |
| logistic | undersample | 0.123 | 0.434 | 0.698 | 0.707 |
| logistic | SMOTEENN | 0.115 | 0.430 | 0.685 | 0.686 |

Scaffold split:

| Classifier | Balancing | [0.0, 0.4) | [0.4, 0.6) | [0.6, 0.8) | [0.8, 1.0] |
| --- | --- | --- | --- | --- | --- |
| forest | none (baseline) | 0.260 | 0.566 | 0.716 | 0.071 (1/3 seeds) |
| forest | weights | 0.264 | 0.561 | 0.722 | 0.091 (1/3 seeds) |
| forest | undersample | 0.279 | 0.571 | 0.723 | 0.083 (1/3 seeds) |
| forest | SMOTEENN | 0.261 | 0.559 | 0.706 | 0.167 (1/3 seeds) |
| logistic | none (baseline) | 0.180 | 0.470 | 0.663 | 0.125 (1/3 seeds) |
| logistic | weights | 0.238 | 0.537 | 0.652 | 0.167 (1/3 seeds) |
| logistic | undersample | 0.226 | 0.515 | 0.547 | 0.500 (1/3 seeds) |
| logistic | SMOTEENN | 0.217 | 0.511 | 0.557 | 0.200 (1/3 seeds) |

**Table 6.3 — NR-AhR, average precision by similarity bin (seed mean)**

Random split:

| Classifier | Balancing | [0.0, 0.4) | [0.4, 0.6) | [0.6, 0.8) | [0.8, 1.0] |
| --- | --- | --- | --- | --- | --- |
| forest | none (baseline) | 0.407 | 0.568 | 0.686 | 0.701 |
| forest | weights | 0.429 | 0.596 | 0.684 | 0.717 |
| forest | undersample | 0.418 | 0.529 | 0.627 | 0.685 |
| forest | SMOTEENN | 0.409 | 0.465 | 0.625 | 0.684 |
| logistic | none (baseline) | 0.431 | 0.602 | 0.690 | 0.699 |
| logistic | weights | 0.438 | 0.601 | 0.679 | 0.725 |
| logistic | undersample | 0.453 | 0.570 | 0.592 | 0.649 |
| logistic | SMOTEENN | 0.438 | 0.544 | 0.637 | 0.658 |

Scaffold split:

| Classifier | Balancing | [0.0, 0.4) | [0.4, 0.6) | [0.6, 0.8) | [0.8, 1.0] |
| --- | --- | --- | --- | --- | --- |
| forest | none (baseline) | 0.364 | 0.540 | 0.591 | 0.697 |
| forest | weights | 0.423 | 0.559 | 0.646 | 0.692 |
| forest | undersample | 0.406 | 0.512 | 0.570 | 0.576 |
| forest | SMOTEENN | 0.373 | 0.498 | 0.555 | 0.632 |
| logistic | none (baseline) | 0.457 | 0.547 | 0.640 | 0.628 |
| logistic | weights | 0.458 | 0.527 | 0.627 | 0.682 |
| logistic | undersample | 0.432 | 0.508 | 0.556 | 0.630 |
| logistic | SMOTEENN | 0.429 | 0.504 | 0.595 | 0.598 |

**Table 6.4 — SR-MMP, average precision by similarity bin (seed mean)**

Random split:

| Classifier | Balancing | [0.0, 0.4) | [0.4, 0.6) | [0.6, 0.8) | [0.8, 1.0] |
| --- | --- | --- | --- | --- | --- |
| forest | none (baseline) | 0.445 | 0.636 | 0.756 | 0.827 |
| forest | weights | 0.468 | 0.621 | 0.741 | 0.788 |
| forest | undersample | 0.428 | 0.580 | 0.716 | 0.836 |
| forest | SMOTEENN | 0.382 | 0.524 | 0.678 | 0.803 |
| logistic | none (baseline) | 0.440 | 0.629 | 0.740 | 0.855 |
| logistic | weights | 0.429 | 0.629 | 0.750 | 0.882 |
| logistic | undersample | 0.417 | 0.571 | 0.639 | 0.810 |
| logistic | SMOTEENN | 0.392 | 0.554 | 0.651 | 0.859 |

Scaffold split:

| Classifier | Balancing | [0.0, 0.4) | [0.4, 0.6) | [0.6, 0.8) | [0.8, 1.0] |
| --- | --- | --- | --- | --- | --- |
| forest | none (baseline) | 0.403 | 0.549 | 0.599 | 0.622 |
| forest | weights | 0.422 | 0.600 | 0.625 | 0.526 |
| forest | undersample | 0.403 | 0.571 | 0.626 | 0.574 |
| forest | SMOTEENN | 0.388 | 0.550 | 0.548 | 0.352 |
| logistic | none (baseline) | 0.383 | 0.628 | 0.626 | 0.481 |
| logistic | weights | 0.388 | 0.631 | 0.645 | 0.406 |
| logistic | undersample | 0.341 | 0.546 | 0.651 | 0.354 |
| logistic | SMOTEENN | 0.358 | 0.587 | 0.535 | 0.369 |

### 6.3 What the bins show

- **Performance rises with similarity to the training set, in every single series.** Across all 48 assay × split × classifier × method series, average precision in the [0.6, 0.8) bin exceeds average precision in the [0.0, 0.4) bin **48 out of 48 times**, and the increase is monotone across the first three bins in 45 of 48. ROC AUC, which does not depend on bin prevalence, shows the same ordering in 47 of 48 series. The low-similarity bin is genuinely the hard regime: for NR-AR random splits the unbalanced forest scores 0.157 average precision in [0.0, 0.4) against 0.686 in [0.6, 0.8).
- **This explains part of why the scaffold split looks harder.** The scaffold split does not create a fundamentally different problem so much as it loads the test set with more low-similarity compounds — Section 2.4 showed mean nearest-neighbour similarity dropping from ~0.56 to ~0.42, and Table 6.1 shows the resulting redistribution of test mass into the lowest bin.
- **The top bin behaves differently on scaffold splits, and should not be over-read.** Average precision in [0.8, 1.0] exceeds [0.6, 0.8) in 22 of 24 random-split series but only 8 of 24 scaffold-split series. Every scaffold [0.8, 1.0] bin is exploratory (12–25 compounds, 0–3 positives), so these values are dominated by sampling noise. The apparently dramatic NR-AR/scaffold/logistic/undersampling value of 0.500 in that bin rests on a single positive compound.
- **Balancing does not change the similarity gradient.** Class weighting with the forest on the scaffold split improves average precision in the low-similarity bin for NR-AhR (0.364 → 0.423) and SR-MMP (0.403 → 0.422), consistent with the aggregate result in Section 3, but no method flattens the gradient or lifts the [0.0, 0.4) bin anywhere near the level of the higher-similarity bins.

### 6.4 The 32 undefined values

`similarity_metrics.csv` contains 32 missing average-precision values and the same number of missing ROC AUC values. These are **not missing data and not a computation failure.**

**Table 6.5 — The source of every undefined value**

| Assay | Split | Seed | Tanimoto bin | Compounds in bin | Positives in bin | Rows affected |
| --- | --- | --- | --- | --- | --- | --- |
| NR-AR | scaffold | 17 | [0.8, 1.0] | 12 | 0 | 16 |
| NR-AR | scaffold | 2026 | [0.8, 1.0] | 15 | 0 | 16 |

Both affected bins contain **zero positive compounds**. Average precision and ROC AUC are mathematically undefined for a single-class subset, so they are correctly recorded as undefined by design. Each of the two bins contributes 16 rows (2 classifiers × 4 balancing methods × 2 calibration states), giving 32 in total. Where a bin's seed mean in Tables 6.2–6.4 is computed from fewer than three seeds, the number of contributing seeds is marked in the cell.

---

## 7. Limitations

These are ordered roughly by how much they constrain the conclusions.

**7.1 The scaffold test set is not a uniform draw over scaffolds.** As set out in Section 2.3, the fixed size-based allocator places the `__EMPTY__` and benzene groups — together 41.9–44.7% of each assay, and 69.9–74.6% of each training partition — into training every time. The scaffold result therefore describes the remaining distribution of smaller scaffold groups. This is a deliberate consequence of a pre-registered, label-independent allocation rule that never splits a group, but it means "performance on new scaffolds" in this report has a narrower meaning than the phrase suggests.

**7.2 Three seeds are not independent replication.** The three partition seeds draw overlapping test sets from the same cleaned assay table. The ± values throughout are split/fit variation, not sampling error over datasets, and they do not support reproducibility claims. The scaffold cross-validation folds are additionally distorted by the same giant groups, which dominate two of the three training folds; scaffold CV fractions deviate by as much as 7.91 percentage points from equal thirds (`docs/data_methods.md`). One NR-AR scaffold CV validation fold contains 14 positives and one NR-AhR fold contains 18, for every seed — both exploratory by the pre-registered threshold, which makes hyperparameter selection on those assays less stable than the grid search implies.

**7.3 The intervals are conditional and unadjusted.** Every bootstrap interval in this report conditions on the already-fitted model and the current test partition. It captures test-set sampling variability only — not retraining, not resampling of the split, not generalisation to other assays. The report makes several hundred such comparisons with no multiplicity adjustment. Intervals excluding zero are reported as a descriptive feature of this particular test partition and nothing more.

**7.4 The two classifiers' intervals within a partition are not independent evidence (audit finding F3).** The bootstrap RNG seed is derived at `toxbench/postprocess.py:32` as `sha256(f'{assay}/{kind}/{seed}')`, which **excludes the classifier**. Within a given assay, split kind and partition seed, the logistic regression group and the random forest group therefore draw the **identical 1,000 resample index vectors**. This is consistent with the pre-registered rule and the pairing within each classifier is correct, so the intervals themselves are valid. But when this report observes that a result appears for both classifiers in the same partition, those two intervals are computed on the same resamples and must not be treated as two independent confirmations. A separate naming hazard: the `seed` column in `outputs/full/bootstrap_differences.csv` holds the **partition seed**, not the bootstrap RNG seed, which `postprocess.py:33` overwrites.

**7.5 Small positive counts.** NR-AR is the limiting case: 298 positives in 6,957 cleaned structures, giving test partitions with 60–96 positives. Several similarity bins fall below the 20-positive exploratory threshold, and two scaffold [0.8, 1.0] bins contain no positives at all. Conclusions about NR-AR, and about any bin flagged exploratory, carry correspondingly less weight.

**7.6 Narrow design space.** One fingerprint (2,048-bit Morgan radius 2, no chirality), one calibration family (a nonnegative-slope Platt sigmoid), four-entry hyperparameter grids, two classifiers, three assays. Results may not transfer to other descriptors, to isotonic or beta calibration, to larger grids, or to the nine Tox21 assays not studied. The audit also notes (finding F4) that no two grid entries ever tied on cross-validated mean average precision, so the documented "ties go to the first grid entry" rule rests on reading the code, not on an observed event.

**7.7 Chemistry and label limitations.** Tautomers are not canonicalised, so residual tautomer-related near-duplicates may remain across partitions. Tanimoto similarity on binary Morgan fingerprints is a crude proxy for chemical relatedness, and the Bemis–Murcko scaffold is a coarse grouping — most visibly in the single `__EMPTY__` group that pools every acyclic compound. Tox21 labels themselves carry assay noise that this study does not model.

**7.8 Literature context is not a validated baseline.** Reference [1] (Idakwo et al., 2020) is fully verified and motivates the inclusion of SMOTEENN, but its curation, descriptors and tuning differ from this project, so its results are motivation and not a comparable baseline. Reference [2] (Santiago-Gonzalez et al., 2025) is verified only as bibliographic metadata and abstract; its detailed methods, including its leakage prevention and evaluation procedure, were not verified here. Reference [3] (van den Goorbergh et al., 2022) is fully verified; its finding that resampling distorts logistic-regression probabilities without improving discrimination is directionally consistent with Sections 3 and 4 of this report, but its evidence comes from simulations and a clinical case study, not from chemical fingerprints or tree ensembles, so the agreement is suggestive rather than confirmatory. **Reference [4] (Kartic et al., 2026, ToxBench) is verified only as bibliographic metadata and publisher abstract; the proposal's specific assertion about that study's use of class weighting for its tree models was not independently verified from the accessible article text and is not asserted as confirmed anywhere in this report.** The full verification record is `docs/references.md`.

**7.9 Scope of the endpoint.** Restating Section 1.2 because it bounds every number above: these are assay activity predictions, not determinations of human toxicity.

---

## 8. Conclusion

**The research question, answered as precisely as this evidence allows.**

**On detection.** Of the three balancing methods tested, only inverse-frequency class weighting avoided systematically reducing the number of active compounds found within a fixed 10% screening budget. Random undersampling and SMOTEENN reduced detection more often than they improved it under both split kinds and most clearly on the random split, where SMOTEENN's average-precision deltas were unfavourable in 16 of 18 comparisons with 10 intervals excluding zero unfavourably and none favourably. Class weighting was close to neutral on random splits (mean average-precision delta +0.0023) and modestly favourable on scaffold splits (+0.0201).

**On new scaffolds specifically.** The one combination with interval-supported gains was class weighting with the random forest on the scaffold split: a mean average-precision delta of +0.0253 across nine comparisons, with four intervals excluding zero favourably and none unfavourably. Those four come from two of the three assays — NR-AhR (seed means 0.469 → 0.510) and SR-MMP (0.477 → 0.516) — and are absent on NR-AR. This is the strongest balancing result in the study, and it is a two-of-three-assay, three-overlapping-seed, unadjusted-interval result on a scaffold test set that excludes the two largest scaffold groups by construction. It is a direction worth testing further; it is not an established effect.

**On reliable probabilities.** Balancing degrades the predicted probabilities badly and consistently. In the raw state, 84 of the 144 models produce probabilities strictly worse, by Brier score, than simply predicting the training prevalence for every compound — including every one of the 36 undersampling models — and all 84 are balanced models, none unbalanced. The reliability curves show the mechanism: raw balanced models are strongly over-confident toward the positive class, with mean signed bias between +0.18 and +0.30. Sigmoid calibration fitted on a separate, uncontaminated partition repairs most of this: all 144 models beat the constant-prevalence reference afterwards, the weighted reliability gap improves for all 48 groups, and the largest Brier delta between any balanced method and the unbalanced baseline falls from +0.2495 to 0.0221. Calibration is not costless — 8 models came out worse on at least one of the two losses, in one case by a visible margin, and all 8 are unbalanced models that were already close to well calibrated — but it is close to a prerequisite for using any balanced model's output as a probability.

**On what calibration does not do.** Calibration did not improve ranking, and could not have. Average precision and ROC AUC are bit-identical between the raw and sigmoid states for all 144 models, with maximum absolute difference exactly 0.0, because the fitted sigmoid is constrained to a nonnegative slope and no model's unconstrained fit wanted a negative one. Any gain reported in this study is a gain in probability quality, never in which compounds would be selected.

**On the harder split.** Testing on held-out scaffolds was uniformly harder than testing on a random split: average precision fell in all 24 assay × classifier × method cells (mean −0.0999), the prevalence-normalised ratio fell in all 24, and across-seed variability more than doubled (mean average-precision SD 0.0379 → 0.0921, a factor of 2.43). The similarity analysis shows why much of this happens: performance rises with nearest-neighbour Tanimoto similarity to the training set in all 48 series tested, and the scaffold split shifts test mass into the lowest-similarity bin.

**The practical reading.** For this benchmark, the defensible recommendation is to train without resampling, use inverse-frequency class weights if anything — they are the only option here that did not cost detection, and the only one with interval-supported gains on the harder split — and then calibrate on a separate partition before treating any output as a probability. Resampling the training data bought no reliable detection benefit and cost a great deal of probability quality that calibration then had to undo. Whether the forest-plus-class-weights gain on unfamiliar scaffolds is real cannot be settled by three assays and three overlapping seeds; establishing it would require the wider evaluation sketched in the proposal's research extension — all twelve assays, additional fingerprints, and an independent compound collection.

---

### Sources for every number in this report

| File | Contents |
| --- | --- |
| `data/audit/assay_counts.csv` | per-assay raw and cleaned counts, prevalences (Table 2.1) |
| `partitions/partition_summary.csv` | realised partition sizes, positives, fractions, deviations (Table 2.2) |
| `partitions/<assay>/scaffold/<seed>.csv` | scaffold group membership and sizes (Table 2.3) |
| `partitions/verification.json` | partition isolation and scaffold-overlap counts (Section 2.2) |
| `outputs/full/alignment_similarity_audit.csv` | nearest-neighbour Tanimoto per test partition (Table 2.4) |
| `outputs/full/summary.csv` | seed means and SDs, all metrics, all states (Tables 3.1–3.7, 4.1, 5.1) |
| `outputs/full/metrics.csv` | the 432 per-model metric rows (Sections 4.1, 4.2, 4.5) |
| `outputs/full/bootstrap_differences.csv` | 1,296 paired deltas with percentile intervals (Tables 3.8–3.12, 4.3, 4.4) |
| `outputs/full/similarity_metrics.csv` | fixed Tanimoto-bin metrics (Tables 6.1–6.5) |
| `outputs/full/evaluation/<assay>/<kind>/<seed>/<classifier>/reliability.csv` | reliability bins (Tables 4.5, 4.7) |
| `outputs/full/evaluation/<assay>/<kind>/<seed>/<classifier>/*.png` | precision–recall and reliability figures |
| `outputs/full/execution_status.json`, `config.json`, `environment.json` | run accounting, grids, package versions |
| `docs/data_verification.json`, `docs/results_verification.json` | the two independent audits |
| `docs/references.md` | per-reference verification status |
