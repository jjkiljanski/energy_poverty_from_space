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
