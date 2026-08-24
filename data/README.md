# `data/` schema

This directory holds the raw simulation output published alongside the
paper. It is organized by **outcome type** — `linear`, `continuous`,
`count`, `categorical` — each a self-contained tree with the same file
layout:

```
data/{outcome}/
  X/{bundle_id}.parquet             design matrix bundles
  Xk/{bundle_id}.parquet            knockoff-copy bundles
  y/{bundle_id}.parquet             response bundles ("trial" bundles)
  importances/{bundle_id}.parquet
  selections/{bundle_id}.parquet
  index.parquet                     bundle join table
  parameters_table.parquet          per-bundle generation parameters (long format)
  results.parquet                   per-bundle loss_value
  X_modes.parquet                   per-X-bundle mixture-mode metadata
```

For a polars-based read/transform API over this layout, see
`src/prismdata/` (`prismdata.loader` for raw access matching the schemas
below, `prismdata.transforms` for friendlier shapes).

## Bundle ids

Every parquet file under `X/`, `Xk/`, `y/`, `importances/`, `selections/`
is named `{bundle_id}.parquet`, where `bundle_id` is that bundle's own
`par_bundle_idn` (a 16-byte id) written as lowercase hex. `X`/`Xk` bundles
are *reused* across many `y`/`importances`/`selections` bundles — a
trial's filename is **not** the filename of the `X`/`Xk` it was generated
from. Use `index.parquet` (below) to join them; do not assume any
filename correspondence across kinds.

## Per-bundle files

### `X/{bundle_id}.parquet`, `Xk/{bundle_id}.parquet`

The design matrix and its knockoff copy. `Xk` has the same row/column
shape and dtype layout as its source `X` (see `index.parquet` for which
`X` a given `Xk` belongs to).

| column | type | description |
|---|---|---|
| `"0"`, `"1"`, ... `"p-1"` | `Float32` for the first `p_numeric` columns, `Categorical` for the remaining `p_categorical` columns | one column per feature, named by its stringified index. **Column count and the numeric/categorical split are not fixed** — they come from that bundle's own `p_numeric`/`p_categorical` generation parameters (see `parameters_table.parquet` below) and can differ bundle to bundle. Numeric columns are standardized continuous values; categorical columns hold category codes as strings (`"0"`, `"1"`, ...). |

Rows are observations (`n`, from the `n` generation parameter); row order
is shared with the bundle's `y`.

### `y/{bundle_id}.parquet`

The response. One row per observation, row-aligned with the `X`/`Xk`
bundle this trial was generated from (see `index.parquet`).

| column | type | description |
|---|---|---|
| `y` | `Float32` (`linear`, `continuous`), `UInt64` (`count`), `Enum` (`categorical`, 3 levels: `"0"`, `"1"`, `"2"`) | the response value; dtype depends on the outcome type. |

### `importances/{bundle_id}.parquet`

Per-feature importance scores for a single trial, from one **or more**
importance methods. Usually `p` rows (one method, one row per feature),
but sometimes `2*p` rows: `torch_grip2` and `torch_prism` are two
importance measures read off the *same* trained model (one grip2-style
group-norm score, one PRISM-style local-gradient score), so the pipeline
computes and writes both into one bundle file -- confirmed in ~20-40% of
trial bundles in the `continuous`/`count`/`categorical` outcomes. **Do
not assume one `importance_method` per file** or take its first row as
"the" method -- `feature_idx` repeats once per method present, so a naive
`{feature_idx: importance}` dict built without grouping by
`importance_method` first will silently keep only one method's values.
`prismdata.transforms.importances_to_dict` handles this correctly (keyed
by `importance_method`); use it rather than reading this table directly
unless you also group by `importance_method` yourself.

| column | type | description |
|---|---|---|
| `par_bundle_idn` | `Binary` | this bundle's own id — redundant with the filename. |
| `feature_idx` | `UInt32` | which `X`/`Xk` column (`0`..`p-1`) this row scores. |
| `importance` | `Float32` | importance score for the original feature. |
| `importance_k` | `Float32` | importance score for that feature's knockoff copy. |
| `importance_method` | `Utf8` | which method this row belongs to — group by this column, don't assume it's constant. Methods observed in this data: `LASSO`, `ranger_gini`, `torch_grip2`, `torch_prism`, `xgb_prism`, `xgb_score`. |

### `selections/{bundle_id}.parquet`

FDR-controlled feature selections derived from one `importances` file, at
one target FDR level. Same one-or-more-methods caveat as `importances/`
above applies here (a shared-fit bundle has `2*p` rows, one set per
method) — use `prismdata.transforms.selections_to_dict`, or group by
`importance_method` yourself.

| column | type | description |
|---|---|---|
| `par_bundle_idn` | `Binary` | this bundle's own id — redundant with the filename. |
| `fdr_target` | `Float32` | target false-discovery-rate level for this row's method (`0.2` throughout the current data). |
| `feature_idx` | `UInt32` | which `X`/`Xk` column (`0`..`p-1`) this row is about. |
| `selected` | `Boolean` | whether the feature was selected at `fdr_target`. |
| `importance_method` | `Utf8` | which method this row's selection was derived from — group by this column, don't assume it's constant. |

## Per-outcome tables

These cover every bundle for an outcome in a single file.

### `index.parquet`

The bundle join table. One row per `X`, `Xk`, or `trial` bundle
(`par_stage` distinguishes them). A `trial` row is a
`y`/`importances`/`selections` triple: all three share that row's own
`par_bundle_idn`, while `X_path`/`Xk_path` name the source `X`/`Xk`
bundle it was generated from.

| column | type | description |
|---|---|---|
| `par_bundle_idn` | `Binary` | this row's own bundle id. |
| `par_stage` | `Utf8` | `"X"`, `"Xk"`, or `"trial"`. |
| `trial_idx` | `Int64` | reserved; always null in the current data. |
| `X_path` | `Utf8` | absolute path (on the generating machine) to the source `X` bundle. Only the filename (basename) is meaningful here — use it to look up `X/{basename}`. Null except on `trial` rows. |
| `Xk_path` | `Utf8` | as `X_path`, for the source `Xk` bundle. |
| `y_path` | `Utf8` | path whose basename equals this row's own `par_bundle_idn` — i.e. `y/{bundle_id}.parquet`. Null except on `trial` rows. |
| `importances_path` | `Utf8` | as `y_path`, for `importances/{bundle_id}.parquet`. |
| `selections_path` | `Utf8` | as `y_path`, for `selections/{bundle_id}.parquet`. |
| `model_path` | `Utf8` | reserved; always null in the current data. |

### `parameters_table.parquet`

Long format: one row per `(bundle, parameter)` pair, covering every
bundle in the outcome (`X`, `Xk`, and `trial` stages alike). Each row's
literal value lives in exactly one of the typed `par_val_*` columns —
the rest are null.

| column | type | description |
|---|---|---|
| `par_group` | `Utf8` | which pipeline stage/component this parameter belongs to: `X`, `Xk`, `y`, `importances`, `selections`, or the pipeline-internal `_meta`/`_prereq` (bundle/config hashes, trial indices, file pointers — not research parameters; stripped by `prismdata.transforms.parameters_to_dict`). |
| `par_type` | `Utf8` | parameter kind; always `"fixed"` in the current data. |
| `par_idx` | `UInt32` | index of this parameter instance within its group, for parameters that can repeat; always `0` in the current data. |
| `par_name` | `Utf8` | parameter name, e.g. `n`, `p_numeric`, `p_categorical`, `covariance_type`, `modes`, `fdr`, `method`. |
| `par_feature` | `Utf8` | reserved; always the literal string `"value"` in the current data (not a feature index, despite the name). |
| `par_val_int64` | `Int64` | the value, if it's an int. |
| `par_val_float64` | `Float64` | the value, if it's a float. |
| `par_val_string` | `Utf8` | the value, if it's a string. |
| `par_val_boolean` | `Boolean` | the value, if it's a bool. |
| `par_val_list_float64` | `List[Float64]` | the value, if it's a list of floats. |
| `par_val_list_int64` | `List[Int64]` | the value, if it's a list of ints. |
| `par_val_list_string` | `List[Utf8]` | the value, if it's a list of strings. |
| `par_val_list_list_float64` | `List[List[Float64]]` | the value, if it's a nested list of floats. |
| `par_val_list_list_int64` | `List[List[Int64]]` | the value, if it's a nested list of ints. |
| `par_bundle_idn` | `Binary` | which bundle this parameter row belongs to. |

Notable `par_name`s seen in the `X`/`Xk` groups:
- `n`: row count
- `p_numeric`/`p_categorical`: feature split — see `X`/`Xk` above
- `categories_per_var`: category count per categorical feature
- `covariance_type`/`covariance_rho`: numeric-feature correlation structure 
- `modes`/`modes_magnitude`: mixture-mode count/spread, see `X_modes.parquet`
- `method`: knockoff construction method, `Xk` group
- `trials`: replicate count

In `y`: 
- `fn_str`: the response's generating function, as an expression string over `col(i)`
- `p_relevant_prob`
- `magnitude`

In `importances`: `method` (names the underlying model fit -- **not**
necessarily the same string as either row's `importance_method` column in
`importances.parquet`/`selections.parquet`; `torch_grip2` and
`torch_prism` both fit one shared model and both log
`method="torch_prism_grip2"` here, since it's genuinely one fit producing
two importance measures -- see `importances/{bundle_id}.parquet` above.
Always dispatch on the `importance_method` column, never on this `method`
parameter) plus method-specific hyperparameters (name varies by fit).

In `selections`: `fdr`/`offset` (same values as the `fdr_target` column
in `selections.parquet`, under prismdata/pipeline-internal names).

### `results.parquet`

| column | type | description |
|---|---|---|
| `par_bundle_idn` | `Binary` | which bundle this result belongs to. |
| `loss_value` | `Float32` | a fit/goodness-of-fit metric associated with the bundle. `0.0` for `X`/`Xk`-stage bundles (no model fit applies); populated for `trial`-stage bundles. |

### `X_modes.parquet`

Per-feature mixture-mode metadata for `X` bundles (numeric features are
generated as a mixture of Gaussian modes; see `modes`/`modes_magnitude`
in `parameters_table.parquet`).

| column | type | description |
|---|---|---|
| `par_bundle_idn` | `Binary` | which `X` bundle this row belongs to. |
| `modes_index` | `UInt32` | which mode (`0`..`modes-1`) this row describes. |
| `feature_idx` | `UInt32` | which `X` column (`0`..`p-1`) this row describes. Present for both numeric and categorical columns, but `value` is only meaningful for numeric ones. |
| `modes_probability` | `Float64` | mixture weight of this mode for this feature; weights across a feature's modes sum to `1.0`. |
| `value` | `Float32` | the mode's location (mean). `NaN` for categorical-coded columns, where the mixture-mode concept doesn't apply. |
