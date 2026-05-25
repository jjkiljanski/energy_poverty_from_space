# Model Tuning History

## Random Forest Broad Search

Date recorded: 2026-05-21

Purpose: first broad random-forest hyperparameter search for predicting the
five EPVI-related targets from satellite-derived indicators plus the basic
administrative indicators.

Code provenance:

- Repository state used for the recorded tuning run: `de84951`
  (`Map EPVI indicators with optional test boundary`).
- The random-forest progress-logging change used by that run was introduced in
  `2658028` (`Add RF training progress logs`).
- The executed notebook outputs from this run were committed in `32aef7a`
  (`Record initial RF tuning notebook outputs`).

The run used the committed random-forest training notebook with:

- 3,087 training rows after the EPVI/admin/satellite inner join;
- 46 non-constant/non-empty predictors;
- shuffled 5-fold CV;
- 50 random hyperparameter candidates per target;
- 250 search fits per target before out-of-fold prediction and full refit.

Search space:

```text
n_estimators:      500, 800, 1200
max_features:      sqrt, 0.35, 0.5, 0.75, 1.0
min_samples_leaf:  1, 2, 4, 8
min_samples_split: 2, 5, 10, 20
max_depth:         None, 8, 12, 18, 24
```

Observed completed results:

| Target | Search time | Best CV R2 | OOF R2 | OOF RMSE | OOF Spearman | Best parameters |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| EPG heating | 26m 41s | 0.4986 | 0.4996 | 1.2359 | 0.6725 | 800 trees, leaf 2, split 2, max features 0.5, unlimited depth |
| EPG cooling | 24m 49s | 0.6817 | 0.6791 | 0.6078 | 0.8422 | 800 trees, leaf 1, split 2, max features 0.75, depth 18 |
| AIAM | 28m 56s | 0.9023 | 0.9023 | 0.3160 | 0.9499 | 800 trees, leaf 1, split 2, max features 0.75, depth 18 |
| EPVI heating | 3h 23m 25s | 0.6356 | 0.6365 | 0.6779 | 0.8069 | 800 trees, leaf 2, split 2, max features 0.5, unlimited depth |

`EPVI cooling` was interrupted before results were produced.

Interpretation of the broad search:

- All completed targets selected `n_estimators=800` and
  `min_samples_split=2`.
- Completed winners used only `min_samples_leaf` 1 or 2,
  `max_features` 0.5 or 0.75, and `max_depth` 18 or unlimited.
- AIAM is much more predictable than the other completed targets. Heating
  generation is the weakest completed target; EPVI heating and EPG cooling are
  intermediate.
- In-sample metrics are much stronger than out-of-fold metrics, so the
  out-of-fold metrics and residuals should be used for model assessment.
- The EPVI-heating search time is not taken as representative. Its later OOF
  and full refit phases were fast, and the laptop was closed during this run.

## Focused Search Next

The next notebook version narrows the search to the parameter region selected
by the broad search. Search forests use fewer trees for speed; final selected
models use the larger evaluation forest when producing out-of-fold predictions
and fitted outputs.

## Spatial Split Correction

The focused tuning run completed much faster, but the resulting feature
importance inspection changed the modeling workflow. Unexpected predictors
appeared among the most important features for some targets, for example share
of superior education for `EPG heating` and `extreme_cold` for `EPG cooling`.
That does not prove spatial leakage on its own, but it is enough reason to
check whether the random forest is learning broad spatial relationships rather
than transportable associations between predictors and EPVI outcomes.

Further tuning should therefore use spatially driven folds at the NUTS3 level,
and model performance should be reported on a clear spatial test set. The
first NUTS3 workflow used `PT16E` and `PT16J` as that fixed holdout; the
parish-to-NUTS3 mapping is stored in `data/freguesias_to_NUTS3.csv`. Later EPG
diagnostics showed that this first holdout was undersized and atypical for
`EPG cooling`; the replacement rule is documented in
`pipeline/3_epvi_prediction/spatial_test_set_selection.ipynb`.

## Spatial Broad Search

Date recorded: 2026-05-21

Purpose: repeat a broad random-forest search after changing model selection to
NUTS3-held-out spatial validation.

Code provenance:

- Repository state used for the recorded spatial run: `6910c55`
  (`Use NUTS3 spatial splits for RF modeling`).
- The executed training notebook outputs from this run are committed together
  with this note.

The run used:

- 2,653 training rows outside the fixed test NUTS3 regions;
- 434 fixed test rows in `PT16E` and `PT16J`;
- 46 non-constant/non-empty predictors;
- five deterministic training-region NUTS3 folds;
- `PT200` and `PT300` placed in different validation folds;
- 40 random hyperparameter candidates per target;
- 200 spatial-CV search fits per target.

Later input validation found that the private EPVI table used an alternate
LAU2 code variant for eight 2013-era freguesias. Before the correction in
`pipeline/3_epvi_prediction/utils.py`, five of those rows were dropped by the
inner join and several overlapping code values could be joined to the wrong
current parish code. The row counts and scores below document this pre-fix run;
the next notebook run should regenerate corrected all-parish results.

Search space:

```text
n_estimators:      300, 500, 800
max_features:      sqrt, 0.35, 0.5, 0.75, 1.0
min_samples_leaf:  1, 2, 4, 8, 12
min_samples_split: 2, 5, 10, 20
max_depth:         None, 8, 12, 18, 24, 32
```

Observed results:

| Target | Best search spatial-CV R2 | Train spatial-CV R2 | Fixed test R2 | Fixed test RMSE | Fixed test Spearman | Best parameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| AIAM | 0.8491 | 0.8555 | 0.9092 | 0.3221 | 0.9527 | 500 trees, leaf 1, split 10, all features, unlimited depth |
| EPVI cooling | 0.3633 | 0.4784 | 0.3719 | 0.5057 | 0.7407 | 500 trees, leaf 1, split 10, all features, unlimited depth |
| EPVI heating | 0.3916 | 0.4500 | 0.3188 | 0.8977 | 0.6644 | 300 trees, leaf 2, split 5, max features 0.75, depth 8 |
| EPG heating | 0.0251 | 0.2188 | 0.0527 | 1.6908 | 0.2654 | 500 trees, leaf 2, split 20, `sqrt` features, depth 8 |
| EPG cooling | -0.0573 | 0.1319 | -0.6738 | 0.9320 | 0.2597 | 500 trees, leaf 2, split 20, `sqrt` features, depth 8 |

Interpretation:

- The spatial workflow changes the substantive conclusion. AIAM remains
  strongly predictable across held-out regions. EPVI heating and cooling retain
  moderate regional-transfer signal. EPG heating is weak, and EPG cooling does
  not transfer to the fixed test regions in this run.
- The poor EPG scores are not automatically a search failure. They may be the
  correct estimate once the forest cannot exploit parish-level spatial
  autocorrelation learned from randomly mixed neighboring regions.
- In-sample scores are still much higher than spatial-CV scores for every
  target, especially the weak EPG targets. Further tuning should not be judged
  from in-sample fit.
- The fixed test set is not used to choose the next parameters. It is recorded
  here to understand the broad run, but the next search should still select
  parameters from training-region spatial CV only.
- The broad winners split into two parameter regimes: the weak EPG targets
  prefer shallow, strongly regularized, `sqrt`-feature forests; AIAM and EPVI
  cooling prefer deep all-feature forests; EPVI heating sits between them.

## Corrected Spatial Second Search

Date recorded: 2026-05-22

Purpose: run a target-aware second spatial random search after aligning the
private EPVI LAU2 code variant to the CAOP/Eurostat-aligned parish IDs.

Code provenance:

- Repository state used for the recorded run: `2576712`
  (`Align EPVI parish IDs before modeling`).
- The executed training notebook outputs from this run are committed together
  with this note.

The corrected run used:

- 2,658 training rows outside fixed test NUTS3 regions;
- 434 fixed test rows in `PT16E` and `PT16J`;
- 46 predictors;
- five deterministic NUTS3-held-out training folds;
- target-aware random-search spaces derived from the spatial broad run;
- 36 sampled parameter candidates and 180 spatial-CV search fits per target.

Observed results:

| Target | Best search spatial-CV R2 | Train spatial-CV R2 | Fixed test R2 | Fixed test RMSE | Fixed test Spearman | Best parameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| AIAM | 0.8497 | 0.8556 | 0.9117 | 0.3175 | 0.9543 | 800 trees, leaf 2, split 10, max features 0.75, depth 24 |
| EPVI cooling | 0.3750 | 0.4903 | 0.3741 | 0.5048 | 0.7468 | 400 trees, leaf 2, split 20, all features, depth 24 |
| EPVI heating | 0.3979 | 0.4640 | 0.3060 | 0.9062 | 0.6614 | 300 trees, leaf 1, split 10, max features 0.75, depth 8 |
| EPG heating | 0.0398 | 0.2157 | 0.0539 | 1.6896 | 0.3144 | 600 trees, leaf 12, split 30, max features 0.2, depth 6 |
| EPG cooling | -0.0336 | 0.1325 | -0.8619 | 0.9829 | 0.2214 | 600 trees, leaf 8, split 30, max features 0.2, depth 4 |

Interpretation:

- Correcting the five dropped EPVI rows and the colliding EPVI ID variants did
  not change the substantive spatial-transfer conclusion.
- The target-aware second search makes only small improvements in spatial-CV
  selection score relative to the spatial broad run. It does not rescue the
  weak EPG targets.
- AIAM is ready for ordinary performance inspection under this RF design.
  EPVI heating and cooling have moderate transfer signal and should be
  inspected in the performance notebook before further RF search is justified.
- For EPG heating and especially EPG cooling, the next decision is likely more
  about model formulation, targets, predictors, spatial baseline comparison,
  and diagnostics than about another narrower random-forest parameter grid.

## Representative Test Split Broad Search

Date recorded: 2026-05-23

Purpose: rerun the broad spatial random-forest search after replacing the
undersized and `EPG cooling`-atypical `PT16E` + `PT16J` holdout with the
representative fixed test set selected in
`pipeline/3_epvi_prediction/spatial_test_set_selection.ipynb`.

Code provenance:

- Training notebook configuration used for this run: `f83baa4`
  (`Reset spatial RF notebook to broad tuning`).
- The representative test set had already been frozen in `bf832d3`
  (`Select representative spatial RF test set`).
- Output timestamp inspected: `20260522_192353`.

The run used:

- 2,474 training rows outside the fixed test NUTS3 regions;
- 618 fixed test rows in `PT112`, `PT16B`, `PT16I`, and `PT16J`;
- 46 predictors;
- five deterministic NUTS3-held-out training folds;
- `PT200` and `PT300` placed in different validation folds;
- one broad random-forest search space for all targets;
- 40 sampled parameter candidates and 200 spatial-CV search fits per target.

Search space:

```text
n_estimators:      300, 500, 800
max_features:      sqrt, 0.35, 0.5, 0.75, 1.0
min_samples_leaf:  1, 2, 4, 8, 12
min_samples_split: 2, 5, 10, 20
max_depth:         None, 8, 12, 18, 24, 32
```

Observed results:

| Target | Best search spatial-CV R2 | Train spatial-CV R2 | Fixed test R2 | Fixed test RMSE | Fixed test Spearman | Best parameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| AIAM | 0.8717 | 0.8754 | 0.8867 | 0.3108 | 0.9382 | 500 trees, leaf 1, split 10, all features, unlimited depth |
| EPVI cooling | 0.4296 | 0.4809 | 0.5891 | 0.4556 | 0.7824 | 500 trees, leaf 2, split 2, max features 0.5, depth 32 |
| EPG cooling | 0.0243 | 0.1802 | 0.3504 | 0.7213 | 0.6188 | 800 trees, leaf 12, split 5, `sqrt` features, depth 24 |
| EPVI heating | 0.4057 | 0.4452 | 0.2404 | 0.7886 | 0.5471 | 500 trees, leaf 4, split 10, max features 0.35, depth 32 |
| EPG heating | 0.0802 | 0.2273 | -0.0599 | 1.4825 | 0.0838 | 500 trees, leaf 8, split 10, `sqrt` features, depth 18 |

Interpretation:

- The representative holdout changes the fixed-test story, especially for
  `EPG cooling`: fixed-test R2 improves from -0.8619 under the earlier holdout
  to 0.3504. This supports the decision to replace the atypical fixed test
  regions before making final performance claims.
- The spatial-CV story is more cautious. `EPG cooling` fixed-test performance
  is now acceptable, but the training-region spatial-CV search score is still
  only 0.0243. This means the model may be fitting the selected test regions
  better than it transfers across arbitrary held-out NUTS3 folds.
- `EPG heating` remains weak. It has the best EPG spatial-CV score so far
  (0.0802), but the fixed-test R2 is slightly negative and Spearman correlation
  is very low.
- AIAM is consistently strong. EPVI cooling is materially better with the
  representative holdout. EPVI heating remains moderate in spatial CV but weak
  on the fixed test set.
- The broad-run winners again split into regimes: AIAM wants a deep all-feature
  forest; EPVI cooling prefers a deep moderately featured forest; EPVI heating
  prefers fewer features and moderate leaf regularization; EPG targets prefer
  `sqrt` features and stronger leaf regularization.

Next tuning step:

- Run a target-aware second pass derived from these broad-search winners.
- Keep parameter selection based on training-region spatial CV, not on fixed
  test performance.
- For EPG targets, search both nearby regularized forests and shallower options
  because spatial-CV transfer remains weak despite the better representative
  fixed-test result for `EPG cooling`.

## Representative Test Split Second Search

Date recorded: 2026-05-25

Purpose: run a target-aware second spatial random-forest search after the
representative-test broad search showed different parameter regimes by target.

Code provenance:

- Training notebook configuration used for this run: `d843f47`
  (`Prepare representative split RF second search`).
- Output timestamp inspected: `20260525_222602`.

The run used:

- 2,474 training rows outside the fixed test NUTS3 regions;
- 618 fixed test rows in `PT112`, `PT16B`, `PT16I`, and `PT16J`;
- 46 predictors;
- five deterministic NUTS3-held-out training folds;
- target-aware random-search spaces derived from the representative-test broad
  search;
- 36 sampled parameter candidates and 180 spatial-CV search fits per target.

Observed results:

| Target | Best search spatial-CV R2 | Train spatial-CV R2 | Fixed test R2 | Fixed test RMSE | Fixed test Spearman | Best parameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| AIAM | 0.8727 | 0.8762 | 0.8876 | 0.3095 | 0.9380 | 500 trees, leaf 2, split 5, all features, depth 32 |
| EPVI cooling | 0.4279 | 0.4789 | 0.5725 | 0.4648 | 0.7793 | 800 trees, leaf 4, split 2, max features 0.75, depth 24 |
| EPG cooling | 0.0276 | 0.1945 | 0.3785 | 0.7055 | 0.6431 | 800 trees, leaf 10, split 10, max features 0.2, depth 6 |
| EPVI heating | 0.4057 | 0.4452 | 0.2404 | 0.7886 | 0.5471 | 500 trees, leaf 4, split 10, max features 0.35, depth 32 |
| EPG heating | 0.0832 | 0.2179 | 0.0069 | 1.4350 | 0.1445 | 500 trees, leaf 10, split 5, `sqrt` features, depth 6 |

Interpretation:

- The second pass did not materially improve the EPG models. `EPG cooling`
  improved slightly on the fixed test set, but its mean fold search score is
  still near zero. `EPG heating` remains effectively unusable on the fixed test
  set.
- The chosen EPG forests became shallower and more regularized, which reduced
  in-sample fit but did not create strong spatial transfer. That is consistent
  with a target or predictor limitation rather than an overfitting-only problem.
- The EPG targets contain substantial regional structure. In the corrected
  modeling table, NUTS3 means explain about 41.7% of `EPG heating` variance and
  about 70.5% of `EPG cooling` variance. This makes region-held-out prediction
  intrinsically difficult if the regional offsets come from administrative or
  reconstructed target components that are not visible in the allowed
  predictors.
- `EPG cooling` is also ceiling-heavy: about 67.8% of parishes have values at
  least 19, and about 31.7% are exactly 20. The raw regression target may
  therefore be a poor formulation for this part of the index.
- AIAM remains stable and strong. EPVI cooling remains moderately strong.
  EPVI heating is moderate in spatial CV but weak on the fixed test set.

Decision:

- Stop ordinary random-forest hyperparameter tuning for EPG unless a new target
  formulation or predictor set is introduced.
- Next work for EPG should focus on target construction and diagnostic
  baselines: global mean, regional/fold means, possible NUTS3 or municipality
  fixed-effect upper bounds, and classification variants such as
  `EPG cooling >= 19` or `EPG cooling == 20`.
- The poor EPG transfer should be treated as a substantive result candidate,
  not merely a failed tuning run.
