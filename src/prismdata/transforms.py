"""Friendly conversions from :mod:`prismdata.loader`'s raw frames.

``loader`` hands back parquet files in whatever shape they're stored on
disk: typed-by-column, keyed by binary ids. This module turns those into
shapes that are easier to work with directly:

- :func:`importances_to_dict` / :func:`selections_to_dict` -- per-bundle
  ``importances``/``selections`` frames, as plain dicts keyed by feature.
- :func:`parameters_to_dict` -- a bundle's long-format parameter rows,
  collapsed to a plain ``{group: {name: value}}`` dict of literal Python
  values, with pipeline-internal identifiers (bundle/config hashes, trial
  indices, file pointers) stripped out.

``X``/``Xk`` bundles are already a wide feature matrix on disk (one row
per observation, one column per feature) -- no conversion needed there,
though their column count and dtype layout is bundle-dependent; see
:data:`RAW_SCHEMAS`.

:data:`RAW_SCHEMAS` documents the on-disk polars schema for every table
under ``data/{outcome}/``, keyed by the same names as
:data:`prismdata.loader.Artifact` plus the outcome-level tables
(``index``, ``parameters_table``, ``results``, ``X_modes``).
"""

from typing import Any

import polars as pl

from prismdata.loader import OutcomeType

# On-disk schema for every table under data/{outcome}/, as it comes out of
# `loader`. ``X``/``Xk`` are the one exception: every column is a feature,
# named by its stringified index ("0", "1", ... "p-1"), columns 0..p_numeric-1
# are pl.Float32 and the remaining p_categorical columns are pl.Categorical --
# but *how many* of each* a bundle has is determined at generation time (from
# that bundle's `p_numeric` + `p_categorical` parameters, see
# `parameters_to_dict`) and is not fixed across bundles or outcome types.
# Don't assume a column count or dtype layout.
RAW_SCHEMAS: dict[str, dict[str, pl.DataType] | dict[OutcomeType, pl.DataType] | str] = {
    "X": (
        "variable: columns '0'..'p-1'; the first p_numeric are pl.Float32, "
        "the remaining p_categorical are pl.Categorical; p is bundle-dependent"
    ),
    "Xk": (
        "variable: columns '0'..'p-1'; the first p_numeric are pl.Float32, "
        "the remaining p_categorical are pl.Categorical; p is bundle-dependent"
    ),
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


def importances_to_dict(df: pl.DataFrame) -> dict[str, Any]:
    """A single bundle's ``importances`` frame as a plain dict.

    ``{"importance_method": str, "importance": {feature_idx: float},
    "importance_k": {feature_idx: float}}``. ``par_bundle_idn`` is
    dropped -- the caller already knows which bundle this is.
    """
    return {
        "importance_method": df["importance_method"][0],
        "importance": dict(zip(df["feature_idx"], df["importance"])),
        "importance_k": dict(zip(df["feature_idx"], df["importance_k"])),
    }


def selections_to_dict(df: pl.DataFrame) -> dict[str, Any]:
    """A single bundle's ``selections`` frame as a plain dict.

    ``{"importance_method": str, "fdr_target": float,
    "selected": {feature_idx: bool}}``. ``par_bundle_idn`` is dropped --
    the caller already knows which bundle this is.
    """
    return {
        "importance_method": df["importance_method"][0],
        "fdr_target": df["fdr_target"][0],
        "selected": dict(zip(df["feature_idx"], df["selected"])),
    }


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
