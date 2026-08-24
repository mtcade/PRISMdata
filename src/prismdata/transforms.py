"""Friendly conversions from :mod:`prismdata.loader`'s raw frames.

``loader`` hands back parquet files in whatever shape they're stored on
disk: typed-by-column, keyed by binary ids. This module turns those into
shapes that are easier to work with directly:

- :func:`importances_to_dict` / :func:`selections_to_dict` -- per-bundle
  ``importances``/``selections`` frames, as plain dicts keyed by
  ``importance_method`` then feature (typed as
  ``dict[str, ImportanceEntry]``/``dict[str, SelectionEntry]``). Usually
  one method per bundle, but not always -- see the note below.
- :func:`parameters_to_dict` -- a bundle's long-format parameter rows,
  collapsed to a plain ``{group: {name: value}}`` dict of literal Python
  values, with pipeline-internal identifiers (bundle/config hashes, trial
  indices, file pointers) stripped out. ``group``/``name`` are data-driven
  (not a fixed set), so this one isn't a ``TypedDict`` -- each value is
  ``Any`` since it's whichever ``par_val_*`` type held the value.
- :func:`load_trial_bundle` -- X, Xk, y, and every relevant parameter
  group for one trial, joined across the X/Xk/trial bundle ids a caller
  would otherwise have to track by hand (see its docstring for the
  ``importance_method`` pitfall it also resolves).
- :func:`split_importance_array` / :func:`combine_importance_arrays` --
  convert between the flat ``(2*p,)`` array every
  ``heteroknockoffpy.importance.*`` function consumes/returns and the
  ``importance``/``importance_k`` pair :func:`importances_to_dict` uses.
- :func:`heteroknockoffpy_kwargs` -- turn one bundle's raw ``importances``
  parameter dict into keyword arguments actually ready to splat into the
  matching ``heteroknockoffpy.importance.*`` call (handles the
  ``torch_grip2``/``torch_prism`` ``layers`` build and the
  ``xgb_prism``/``xgb_score`` ``model_kwargs`` nesting -- see its
  docstring for why both are needed, verified against
  ``heteroknockoffpy``'s actual implementations).

``X``/``Xk`` bundles are already a wide feature matrix on disk (one row
per observation, one column per feature) -- no conversion needed there.
Their schema isn't fixed enough to list in :data:`RAW_SCHEMAS`: columns
are named ``"0"``, ``"1"``, ... ``"p-1"``, the first ``p_numeric`` are
``pl.Float32`` and the remaining ``p_categorical`` are ``pl.Categorical``,
and ``p``/the numeric-categorical split are bundle-dependent generation
parameters (see :func:`parameters_to_dict`'s ``"X"`` group), not a fixed
schema.

:data:`RAW_SCHEMAS` documents the on-disk polars schema for every other
table under ``data/{outcome}/``: the remaining
:data:`prismdata.loader.Artifact` values (``y``, ``importances``,
``selections``) plus the outcome-level tables (``index``,
``parameters_table``, ``results``, ``X_modes``).

**A bundle can hold more than one ``importance_method``.** Most
``importances``/``selections`` files hold exactly one method (one row per
feature). But ``"torch_grip2"`` and ``"torch_prism"`` are two
importance measures read off the *same* trained model -- one grip2-style
group-norm score, one PRISM-style local-gradient score -- so the pipeline
computes them together and writes both into one bundle file (confirmed:
~20-40% of trial bundles in the ``continuous``/``count``/``categorical``
outcomes; verified across all of them). Never assume a bundle has exactly
one method or take ``df["importance_method"][0]`` as *the* method for the
whole frame -- that silently drops the other method's rows (same
``feature_idx`` values repeat, one set per method) and reports whichever
method happened to sort first. :func:`importances_to_dict` /
:func:`selections_to_dict` group by ``importance_method`` for exactly
this reason; a caller iterating their result already handles 1-method and
2-method bundles identically. Correspondingly,
``parameters["importances"]["method"]`` (from :func:`parameters_to_dict`)
names the shared underlying model fit (``"torch_prism_grip2"``), not
either individual method -- don't use it to dispatch a
``heteroknockoffpy.importance.*`` call; use the keys of
:func:`importances_to_dict`'s/:func:`selections_to_dict`'s result (or
:func:`load_trial_bundle`'s ``importances``/``selections`` fields)
instead.
"""

from typing import Any, TypedDict

import numpy as np
import polars as pl

from prismdata.loader import OutcomeType, load_bundle, parameters, trials

# On-disk schema for every table under data/{outcome}/, as it comes out of
# `loader` -- except X/Xk, which have no fixed schema; see the module
# docstring above.
RAW_SCHEMAS: dict[str, dict[str, pl.DataType] | dict[OutcomeType, pl.DataType]] = {
    "y": {
        "linear": pl.Float32,
        "continuous": pl.Float32,
        "count": pl.UInt64,
        "categorical": pl.Enum,
    },
    "importances": {
        "par_bundle_idn": pl.Binary,
        "feature_idx": pl.UInt32,
        "importance": pl.Float32,
        "importance_k": pl.Float32,
        "importance_method": pl.Utf8,
    },
    "selections": {
        "par_bundle_idn": pl.Binary,
        "fdr_target": pl.Float32,
        "feature_idx": pl.UInt32,
        "selected": pl.Boolean,
        "importance_method": pl.Utf8,
    },
    "index": {
        "par_bundle_idn": pl.Binary,
        "par_stage": pl.Utf8,
        "trial_idx": pl.Int64,
        "X_path": pl.Utf8,
        "Xk_path": pl.Utf8,
        "y_path": pl.Utf8,
        "importances_path": pl.Utf8,
        "selections_path": pl.Utf8,
        "model_path": pl.Utf8,
    },
    "parameters_table": {
        "par_group": pl.Utf8,
        "par_type": pl.Utf8,
        "par_idx": pl.UInt32,
        "par_name": pl.Utf8,
        "par_feature": pl.Utf8,
        "par_val_int64": pl.Int64,
        "par_val_list_float64": pl.List(pl.Float64),
        "par_val_list_int64": pl.List(pl.Int64),
        "par_val_string": pl.Utf8,
        "par_val_float64": pl.Float64,
        "par_val_boolean": pl.Boolean,
        "par_val_list_string": pl.List(pl.Utf8),
        "par_val_list_list_float64": pl.List(pl.List(pl.Float64)),
        "par_val_list_list_int64": pl.List(pl.List(pl.Int64)),
        "par_bundle_idn": pl.Binary,
    },
    "results": {
        "par_bundle_idn": pl.Binary,
        "loss_value": pl.Float32,
    },
    "X_modes": {
        "par_bundle_idn": pl.Binary,
        "modes_index": pl.UInt32,
        "feature_idx": pl.UInt32,
        "modes_probability": pl.Float64,
        "value": pl.Float32,
    },
}

_PARAMETER_VALUE_COLUMNS = (
    "par_val_int64",
    "par_val_float64",
    "par_val_string",
    "par_val_boolean",
    "par_val_list_float64",
    "par_val_list_int64",
    "par_val_list_string",
    "par_val_list_list_float64",
    "par_val_list_list_int64",
)

# par_group values holding pipeline bookkeeping (config hashes, trial
# indices, file pointers) rather than a research-meaningful parameter.
_IDENTIFIER_PARAM_GROUPS = ("_meta", "_prereq")


class ImportanceEntry(TypedDict):
    importance: dict[int, float]
    importance_k: dict[int, float]


class SelectionEntry(TypedDict):
    fdr_target: float
    selected: dict[int, bool]


def importances_to_dict(df: pl.DataFrame) -> dict[str, ImportanceEntry]:
    """A single bundle's ``importances`` frame as a plain dict, keyed by
    ``importance_method`` (see the module docstring: usually one key, but
    a bundle can hold more than one).

    ``par_bundle_idn`` is dropped -- the caller already knows which
    bundle this is.
    """
    result: dict[str, ImportanceEntry] = {}
    for (method,), sub in df.group_by("importance_method", maintain_order=True):
        result[method] = {
            "importance": dict(zip(sub["feature_idx"], sub["importance"])),
            "importance_k": dict(zip(sub["feature_idx"], sub["importance_k"])),
        }
    return result


def selections_to_dict(df: pl.DataFrame) -> dict[str, SelectionEntry]:
    """A single bundle's ``selections`` frame as a plain dict, keyed by
    ``importance_method`` (see the module docstring: usually one key, but
    a bundle can hold more than one).

    ``par_bundle_idn`` is dropped -- the caller already knows which
    bundle this is.
    """
    result: dict[str, SelectionEntry] = {}
    for (method,), sub in df.group_by("importance_method", maintain_order=True):
        result[method] = {
            "fdr_target": sub["fdr_target"][0],
            "selected": dict(zip(sub["feature_idx"], sub["selected"])),
        }
    return result


def _parameter_value(row: dict[str, Any]) -> Any:
    for column in _PARAMETER_VALUE_COLUMNS:
        value = row[column]
        if value is not None:
            return value
    return None


def parameters_to_dict(df: pl.DataFrame) -> dict[str, dict[str, Any]]:
    """A single bundle's ``parameters_table`` rows as a plain nested dict.

    ``{par_group: {par_name: value}}``, with each row's literal value
    pulled out of whichever ``par_val_*`` column holds it. Rows in the
    ``_meta``/``_prereq`` groups -- pipeline identifiers (bundle/config
    hashes, trial indices, file pointers), not research parameters -- are
    stripped. A ``(group, name)`` pair that appears more than once (a
    multi-component parameter, indexed by ``par_idx``) collapses to a
    list of values in ``par_idx`` order instead of a single value.
    """
    df = df.filter(~pl.col("par_group").is_in(_IDENTIFIER_PARAM_GROUPS)).sort("par_idx")

    result: dict[str, dict[str, Any]] = {}
    for (group, name), sub in df.group_by(("par_group", "par_name"), maintain_order=True):
        values = [_parameter_value(row) for row in sub.iter_rows(named=True)]
        result.setdefault(group, {})[name] = values[0] if len(values) == 1 else values
    return result


class TrialBundle(TypedDict):
    """Everything needed to reproduce one trial with heteroknockoffpy."""

    outcome: OutcomeType
    bundle_id: str
    x_bundle_id: str
    xk_bundle_id: str
    X: pl.DataFrame
    Xk: pl.DataFrame
    y: pl.DataFrame
    importances: dict[str, ImportanceEntry]
    selections: dict[str, SelectionEntry]
    parameters: dict[str, dict[str, Any]]


def load_trial_bundle(outcome: OutcomeType, bundle_id: str) -> TrialBundle:
    """Load and join everything for one trial: ``X``, ``Xk``, ``y``, and
    every relevant generation-parameter group, fetched by the *correct*
    bundle id for each.

    A trial's own ``bundle_id`` only owns its ``y``/``importances``/
    ``selections`` rows -- reconstructing knockoffs or importances needs
    the *X*-stage bundle's ``"X"`` parameter group and the *Xk*-stage
    bundle's ``"Xk"`` parameter group, which live under
    ``x_bundle_id``/``xk_bundle_id``, not ``bundle_id``. This joins all
    three so the caller doesn't have to track which bundle owns which
    group.

    ``importances``/``selections`` are keyed by ``importance_method``
    (via :func:`importances_to_dict`/:func:`selections_to_dict`) -- iterate
    their keys rather than assuming one method; see the module docstring
    for why a trial can have more than one, and why
    ``parameters["importances"]["method"]`` isn't a substitute for either
    key set.
    """
    trial = trials(outcome).filter(pl.col("bundle_id") == bundle_id).row(0, named=True)
    x_bundle_id = trial["x_bundle_id"]
    xk_bundle_id = trial["xk_bundle_id"]

    all_params = parameters(outcome)
    merged_params: dict[str, dict[str, Any]] = {}
    for bid in (x_bundle_id, xk_bundle_id, bundle_id):
        merged_params.update(parameters_to_dict(all_params.filter(pl.col("bundle_id") == bid)))

    return {
        "outcome": outcome,
        "bundle_id": bundle_id,
        "x_bundle_id": x_bundle_id,
        "xk_bundle_id": xk_bundle_id,
        "X": load_bundle(outcome, "X", x_bundle_id),
        "Xk": load_bundle(outcome, "Xk", xk_bundle_id),
        "y": load_bundle(outcome, "y", bundle_id),
        "importances": importances_to_dict(load_bundle(outcome, "importances", bundle_id)),
        "selections": selections_to_dict(load_bundle(outcome, "selections", bundle_id)),
        "parameters": merged_params,
    }


def split_importance_array(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a ``heteroknockoffpy`` importance array into ``(importance, importance_k)``.

    Every ``heteroknockoffpy.importance.*`` function returns one flat
    ``(2*p,)`` array: the first ``p`` entries score ``X``'s own columns,
    the last ``p`` score their knockoff copies. This splits that back into
    the two halves prismdata's ``importances`` table stores as separate
    ``importance``/``importance_k`` columns (see :func:`importances_to_dict`).
    Inverse of :func:`combine_importance_arrays`.
    """
    p = len(arr) // 2
    return arr[:p], arr[p:]


def combine_importance_arrays(importance: np.ndarray, importance_k: np.ndarray) -> np.ndarray:
    """Combine ``(importance, importance_k)`` into one flat array.

    Produces the same ``(2*p,)`` shape every ``heteroknockoffpy.importance.*``
    function returns and ``heteroknockoffpy.importance.wFromImportances``
    expects. Inverse of :func:`split_importance_array`.
    """
    return np.concatenate([importance, importance_k])


# heteroknockoffpy.importance functions whose xgboost model-constructor
# hyperparameters must arrive nested under a `model_kwargs` dict (they're
# otherwise silently forwarded to XGBRegressor/XGBClassifier.fit() instead
# of the constructor, and raise TypeError there -- verified against
# heteroknockoffpy's actual _fit_model implementation, not just its
# docstring).
_XGB_MODEL_KWARGS = frozenset({
    "max_depth", "learning_rate", "min_child_weight", "subsample",
    "colsample_bytree", "reg_alpha", "reg_lambda", "gamma", "n_estimators",
})

# heteroknockoffpy.importance functions that take no **kwargs and instead
# want a `layers: Sequence[int]` built from three ratio/count parameters.
_LAYERS_METHODS = frozenset({"torch_grip2", "torch_prism"})
_LAYERS_KEYS = ("first_layer_ratio", "layer_count", "layer_ratio")

# Keys present in every importances parameter group that are never real
# heteroknockoffpy keyword arguments (bookkeeping the pipeline logged
# alongside the actual hyperparameters).
_NON_KWARG_KEYS = frozenset({"method", "standardize_y", "local_grad_method"})


def heteroknockoffpy_kwargs(
    importance_method: str, raw_params: dict[str, Any], p: int
) -> dict[str, Any]:
    """Turn one bundle's raw ``importances`` parameter dict into keyword
    arguments ready to splat into the matching ``heteroknockoffpy.importance.*``
    call for ``importance_method``.

    ``raw_params`` -- typically
    ``load_trial_bundle(...)["parameters"]["importances"]`` or
    ``parameters_to_dict(...)["importances"]`` -- isn't directly forwardable
    as-is: besides carrying non-parameter bookkeeping (``method``,
    ``standardize_y``), two method families need real reshaping, both
    verified against heteroknockoffpy's actual implementations (not just
    their docstrings):

    - ``"torch_grip2"``/``"torch_prism"`` (``grip2Importances``/
      ``torchPrismImportances``) take no ``**kwargs`` at all -- they want a
      ``layers: Sequence[int]`` built from ``first_layer_ratio``/
      ``layer_count``/``layer_ratio`` and `p`, not those three directly.
      The formula matches the silverknockoff pipeline that generated this
      data exactly (``calculatorOps.py``'s ``CalculateImportances``):
      ``layers = [max(1, round(p * first_layer_ratio))]``, then
      ``layer_count - 1`` more entries each
      ``max(1, round(previous * layer_ratio))``.
    - ``"xgb_prism"``/``"xgb_score"`` (``xgbPrismImportances``/
      ``xgbImportances``) want xgboost model-constructor hyperparameters
      (``max_depth``, ``learning_rate``, ``min_child_weight``,
      ``subsample``, ``colsample_bytree``, ``reg_alpha``, ``reg_lambda``,
      ``gamma``, ``n_estimators``) nested under a ``model_kwargs`` dict --
      passed top-level, they're silently forwarded to
      ``XGBRegressor.fit()`` instead of the constructor and raise
      ``TypeError`` there.

    ``"LASSO"``/``"ranger_gini"`` need no reshaping -- their raw parameter
    names already match real keyword arguments (``LASSO``'s extras are
    silently accepted via ``kwargs.get(...)``; ``ranger_gini``'s are
    forwarded to ``ranger::ranger`` under the same names).

    :param p: the ``X`` bundle's own feature count (``X.width``) -- only
        consulted for the ``torch_grip2``/``torch_prism`` ``layers`` build;
        matches what the ``p`` in silverknockoff's own layer-width formula
        refers to (raw column count of ``X``, not ``Xk``, not an
        OHE-expanded or ``[X, Xk]``-combined width).
    """
    kwargs = {k: v for k, v in raw_params.items() if k not in _NON_KWARG_KEYS}

    if importance_method in _LAYERS_METHODS:
        first_layer_ratio, layer_count, layer_ratio = (kwargs.pop(k) for k in _LAYERS_KEYS)
        layers = [max(1, round(p * first_layer_ratio))]
        for _ in range(layer_count - 1):
            layers.append(max(1, round(layers[-1] * layer_ratio)))
        kwargs["layers"] = layers

    elif importance_method in ("xgb_prism", "xgb_score"):
        model_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in _XGB_MODEL_KWARGS}
        kwargs["model_kwargs"] = model_kwargs

    return kwargs
