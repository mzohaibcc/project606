# Implementation plan and fixed decisions

1. Audit the DeepChem-linked Tox21 download, standardize and deduplicate independently per assay, save structure/fingerprint provenance.
2. Freeze seeds 17, 42, 2026 and complete partition manifests before evaluating models. Scaffold allocation is label-independent; random allocation is stratified. Preserve empty scaffolds as one group.
3. Use the complete four-setting grids, three training-only CV folds, mean average precision, deterministic grid order for ties. Fit all balancing inside the current fold; cache only exact content-addressed training inputs. No learned feature transformation is required for binary Morgan bits.
4. Time a smoke run (one partition, both classifiers, all balancing choices) then run all 144 final models if feasible. Keep checkpoints, independent output roots and final predictions.
5. Evaluate raw and separate-partition sigmoid predictions; bootstrap all methods on identical samples; report descriptive seed means/SD without test-driven selection. Generate audit, plots, report, notebook and ZIP.

## Decisions fixed before results

- Float32 fingerprints enter resampling; SMOTE uses Euclidean neighbors and k=min(5, minority_count-1), with adjustments logged. ENN cleans all classes with three neighbors and unanimity selection. Missing either class, fewer than two minority cases for SMOTE, or single-class output is an explicit failed configuration; never substitute another sampler.
- Logistic regression uses liblinear (L2, max_iter=3000), no scaling; forests use sqrt feature subsampling, 200 trees and one estimator thread. Up to four independent partitions run in parallel; native BLAS/OpenMP limited to one thread per worker.
- Sigmoid calibration fits a positive-slope logistic mapping with Platt-smoothed targets on calibration scores only. Logistic regression uses decision_function; forest uses positive-class probability. A nonnegative slope preserves ordering; slope-zero degeneracy is logged. Probabilities clipped only for log loss (not ranking).
- Top 10% means ceil(0.10*n), score descending then structure_id ascending. Bootstrap occurrence order resolves repeated identical IDs.
- Bootstrap uses 1000 paired replicates, percentile 95% intervals, random seed derived from partition key. Undefined metrics are excluded per metric with valid/invalid replicate counts. Intervals condition on fitted models and current partition, not training uncertainty.
- Checkpoints bind code, package versions, configuration, fingerprints, partition contents and ordered training rows. Partial writes are atomic. Test data never informs model or sampler choices.
- Scientific date context: 25 November 2026; provenance additionally records actual UTC download timestamp from the execution host.

## Departures

RDKit 2026.9.2 failed basic scaffold extraction on this host (bad any cast); RDKit was pinned to 2025.9.3 before analysis. Sigmoid calibration uses a nonnegative slope constraint and Platt target smoothing, an explicit implementation choice that prevents rank reversal. Local CPU execution replaces the proposed Colab CPU execution; the notebook will support Colab reproduction. Exact engineering choices absent from the proposal are documented above and in data_methods.md.
