# PRISMdata

Data for simulation results from Heterogeneous Knockoff Importances, published
alongside the paper for transparency and reproducibility.

## Layout

Data is hosted on Zenodo ([10.5281/zenodo.22085306](https://doi.org/10.5281/zenodo.22085306),
`data.zip`), not in this repo -- `data/` and `data.zip` are gitignored.
Fetch it with:

```bash
uv run python scripts/fetch_data.py
```

This downloads `data.zip`, verifies its SHA-256 checksum
(`b9e632421711cf72f7526f53d304adacda5fbc448660758218e1b9818b0c14db`),
extracts it into `data/`, and deletes the zip. Re-running is safe --
an existing `data.zip` with a matching checksum is reused rather than
re-downloaded. See `scripts/fetch_data.py --help` for options
(`--keep-zip`, overriding the URL/checksum, extracting elsewhere).

Once downloaded, the data layout is

```
data/{outcome}/                  outcome in {linear, continuous, count, categorical}
  X/{bundle_id}.parquet          design matrix bundles
  Xk/{bundle_id}.parquet         knockoff-copy bundles
  y/{bundle_id}.parquet          response bundles ("trial" bundles)
  importances/{bundle_id}.parquet
  selections/{bundle_id}.parquet
  index.parquet                  join table: trial -> (X bundle, Xk bundle)
  parameters_table.parquet       per-bundle generation parameters (long format)
  results.parquet                per-bundle loss_value
  X_modes.parquet                per-X-bundle mixture-mode metadata
```

Other parts of the package are
```
src/prismdata/                   thin polars-based read API over data/
notebooks/                       marimo notebooks (.py) for exploration
tests/                           tests for the read API and data invariants
```

See [`data/README.md`](data/README.md) for the full column-by-column
schema of every file above.

`y`, `importances`, and `selections` files are each named by their own
`par_bundle_idn` (as lowercase hex). `X` and `Xk` files are similarly named
by their own bundle id but are *reused* across many trials — a trial's `y`
does not share a filename with the `X`/`Xk` it was generated from. The join
is `data/{outcome}/index.parquet`: its `trial` rows carry `X_path`/`Xk_path`
pointing at the source `X`/`Xk` bundle. Use `prismdata.trials()` /
`prismdata.index()` rather than joining artifacts by filename.

## Setup

```bash
uv sync                              # installs polars (the only runtime dependency)
uv sync --group notebooks            # + marimo, to run notebooks/
uv run python scripts/fetch_data.py  # download data/ from Zenodo (see Layout above)
```

## Usage

```python
import prismdata
from prismdata import transforms

prismdata.list_bundle_ids("linear", "y")   # bundle ids available for an artifact
prismdata.load_bundle("linear", "y", bundle_id)  # a single bundle
prismdata.load("linear", "importances")    # every bundle for an artifact, concatenated

prismdata.trials("linear")     # trial -> (x_bundle_id, xk_bundle_id) join table
prismdata.parameters("linear") # per-bundle generation parameters
prismdata.results("linear")    # per-bundle loss_value
prismdata.x_modes("linear")    # per-X-bundle mixture-mode metadata

# raw -> friendlier shapes (X/Xk are already a wide feature matrix on disk,
# no conversion needed there)
transforms.importances_to_dict(imp_df)   # {importance_method: {importance: {feature_idx: value}, importance_k: {...}}, ...}
transforms.selections_to_dict(sel_df)    # {importance_method: {fdr_target: value, selected: {feature_idx: bool}}, ...}
transforms.parameters_to_dict(params_df) # {group: {name: value}}, identifiers stripped

# X, Xk, y, and every relevant parameter group for one trial, correctly
# joined -- see "Reproducing with heteroknockoffpy" below
transforms.load_trial_bundle("linear", bundle_id)
```

`prismdata.OutcomeType` and `prismdata.Artifact` are `Literal` types (not
enumerable constants) -- use `typing.get_args()` if you need the value set,
e.g. for iterating in a notebook or test.

Run tests with `uv run pytest`. Run the notebook with
`uv run --group notebooks marimo edit notebooks/explore.py`.

## Using with heteroknockoffpy

The data was generated with `heteroknockoffpy`, and
every bundle's generation parameters (`prismdata.parameters()`) are real
`heteroknockoffpy` keyword arguments -- pulling them through
`transforms.load_trial_bundle()` (or `parameters_to_dict()` directly) and
splatting them into the matching `heteroknockoffpy` call creates
(structurally; a fresh knockoff draw uses fresh randomness, so exact
values won't match) that bundle's knockoffs/importances/selections.

```bash
uv sync --group reproduce   # + heteroknockoffpy, for the examples below
```

`load_trial_bundle` does the bundle-id bookkeeping for you: it fetches
`X`/`Xk`/`y` and every parameter group from the *correct* bundle (a
trial's own bundle only owns its `y`/`importances`/`selections` -- the
`"X"` and `"Xk"` parameter groups live on the `X`/`Xk` bundles it points
to), and it exposes `importances`/`selections` keyed by
`importance_method` -- **almost always one key, but sometimes two**:
`torch_grip2` and `torch_prism` are two importance measures read
off one shared trained model, so ~20-40% of `continuous`/`count`/
`categorical` trial bundles hold both under one bundle id. Always iterate
these dict keys; never assume exactly one method, and never dispatch on
`parameters["importances"]["method"]` (it names the shared *fit*, e.g.
`"torch_prism_grip2"` for both of the methods above -- not either
individual `importance_method`).

```python
import numpy as np
import prismdata
from prismdata import transforms
from heteroknockoffpy import knockoff, importance

outcome = "linear"
bundle_id = prismdata.trials(outcome)["bundle_id"][0]
tb = transforms.load_trial_bundle(outcome, bundle_id)
rng = np.random.default_rng(0)  # not stored in the data -- supply your own
```

**Knockoffs** (`X` → `Xk`):

```python
Xk = knockoff.get_knockoffs(tb["X"], rng=rng, **tb["parameters"]["Xk"])
# tb["parameters"]["Xk"] == {'method': 'second_order', 'categorical_method': 'ohe'}
```

**Importances** (`X`, `Xk`, `y` → per-feature scores), once per method
present on this trial. `tb["parameters"]["importances"]` isn't directly
forwardable as keyword arguments -- besides bookkeeping keys
(`method`/`standardize_y`) that aren't real parameters,
`torch_grip2`/`torch_prism` need a computed `layers` list instead of the
three ratio/count values that are actually stored, and `xgb_prism`/
`xgb_score` need their xgboost hyperparameters nested under a
`model_kwargs` dict rather than passed top-level (passed top-level, they
silently reach `XGBRegressor.fit()` instead of the constructor and raise
`TypeError`). `transforms.heteroknockoffpy_kwargs()` does all of this
reshaping -- see its docstring for the exact rules per method:

```python
DISPATCH = {
    "LASSO": importance.lassoImportances,
    "ranger_gini": importance.rangerGiniImportances,
    "torch_grip2": importance.grip2Importances,
    "torch_prism": importance.torchPrismImportances,
    "xgb_prism": importance.xgbPrismImportances,
    "xgb_score": importance.xgbImportances,
}

results = {}
for method in tb["importances"]:  # usually one method; two for a shared torch fit
    kwargs = transforms.heteroknockoffpy_kwargs(
        method, tb["parameters"]["importances"], p=tb["X"].width
    )
    arr = DISPATCH[method](tb["X"], Xk, tb["y"], rng=rng, **kwargs)
    results[method] = transforms.split_importance_array(arr)
```

**Selections** (importances + FDR target → boolean mask), same
per-method loop. Use `tb["selections"][method]["fdr_target"]` for `fdr`
(the authoritative, always-correct-per-method value; the raw
`tb["parameters"]["selections"]` dict has the same one-dict-per-multiple-
methods ambiguity as the `importances` group's `method` key above) --
`offset` doesn't vary per method in this data, so
`tb["parameters"]["selections"]["offset"]` is fine as-is. The boolean
mask itself isn't returned by `heteroknockoffpy` and must be computed
from its `W`/threshold:

```python
offset = tb["parameters"]["selections"]["offset"]

for method, (feature_importance, feature_importance_k) in results.items():
    arr_combined = transforms.combine_importance_arrays(feature_importance, feature_importance_k)
    W = importance.wFromImportances(arr_combined, W_method="difference")
    fdr = tb["selections"][method]["fdr_target"]
    T = importance.selection_threshold(W, fdr=fdr, offset=offset)
    selected = (W >= T) & (W > 0)  # matches tb["selections"][method]["selected"]
```

### heteroknockoffpy parameters stored in this data

The values that select *which* `heteroknockoffpy` code path ran, and
where to find them:

| parameter | lives in | meaning |
|---|---|---|
| `importance_method` | `importances`/`selections` table column, or `load_trial_bundle(...)["importances"]`/`["selections"]` keys | which `heteroknockoffpy.importance.*` function to call -- the *only* reliable dispatch key. A bundle can have more than one (see above); `parameters["importances"]["method"]` is **not** a substitute -- it names the shared underlying fit, and collapses `torch_grip2`/`torch_prism` onto the same value. |
| `method` (`Xk` group) | `parameters_table`, `par_group="Xk"`, on the **`Xk` bundle** | knockoff construction method, passed to `heteroknockoffpy.knockoff.get_knockoffs`. Always `"second_order"` in this data. |
| `categorical_method` (`Xk` group) | `parameters_table`, `par_group="Xk"` | how categorical columns are handled during knockoff construction; always `"ohe"` here. |
| `fdr` (`selections` group) | `parameters_table`, `par_group="selections"`, or `tb["selections"][method]["fdr_target"]` | target FDR, passed to `heteroknockoffpy.importance.selection_threshold`. |
| `offset` (`selections` group) | `parameters_table`, `par_group="selections"` | knockoff threshold offset (0 = modified FDR, 1 = exact FDR), passed to `selection_threshold`. |
| method-specific hyperparameters (`importances` group) | `parameters_table`, `par_group="importances"` | e.g. `n_splits`/`max_iter` (LASSO), `respect_unordered_factors` (ranger_gini), xgboost hyperparameters (xgb family -- see `heteroknockoffpy_kwargs`'s `model_kwargs` nesting above), `epochs`/`n_blocks`/`calibrate`/layer-shape inputs (grip2/torch_prism family) -- turned into real keyword arguments for the matching `heteroknockoffpy.importance.*` function by `transforms.heteroknockoffpy_kwargs()`.
