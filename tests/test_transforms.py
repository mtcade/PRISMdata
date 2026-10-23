from typing import get_args

import numpy as np
import polars as pl
import pytest

import prismdata
from prismdata import transforms

OUTCOME_TYPES = get_args(prismdata.OutcomeType)


@pytest.mark.parametrize("outcome", OUTCOME_TYPES)
def test_importances_to_dict_covers_every_feature_per_method(outcome):
    trial = prismdata.trials(outcome).row(0, named=True)
    df = prismdata.load_bundle(outcome, "importances", trial["bundle_id"])
    d = transforms.importances_to_dict(df)

    assert set(d.keys()) == set(df["importance_method"].unique().to_list())
    for method, entry in d.items():
        expected_features = set(
            df.filter(pl.col("importance_method") == method)["feature_idx"].to_list()
        )
        assert set(entry.keys()) == {"importance", "importance_k"}
        assert set(entry["importance"].keys()) == expected_features
        assert set(entry["importance_k"].keys()) == expected_features


@pytest.mark.parametrize("outcome", OUTCOME_TYPES)
def test_selections_to_dict_covers_every_feature_per_method(outcome):
    trial = prismdata.trials(outcome).row(0, named=True)
    df = prismdata.load_bundle(outcome, "selections", trial["bundle_id"])
    d = transforms.selections_to_dict(df)

    assert set(d.keys()) == set(df["importance_method"].unique().to_list())
    for method, entry in d.items():
        sub = df.filter(pl.col("importance_method") == method)
        assert set(entry.keys()) == {"fdr_target", "selected"}
        assert isinstance(entry["fdr_target"], float)
        assert set(entry["selected"].keys()) == set(sub["feature_idx"].to_list())
        assert all(isinstance(v, bool) for v in entry["selected"].values())


def test_importances_to_dict_keeps_both_methods_when_a_bundle_shares_one_fit():
    # torch_grip2/torch_prism are two importance measures read off the
    # same trained model and share one bundle file (see module docstring).
    # A naive df["importance_method"][0] + zip(feature_idx, ...) approach
    # would silently drop one method's 30 rows since feature_idx repeats
    # across methods; importances_to_dict must keep both intact.
    outcome = "continuous"
    sel = prismdata.load(outcome, "selections")
    multi_method_bundles = (
        sel.group_by("bundle_id")
        .agg(pl.col("importance_method").n_unique().alias("n"))
        .filter(pl.col("n") > 1)["bundle_id"]
    )
    assert len(multi_method_bundles) > 0  # sanity: this case actually exists in the data
    bundle_id = multi_method_bundles[0]

    imp_df = prismdata.load_bundle(outcome, "importances", bundle_id)
    d = transforms.importances_to_dict(imp_df)
    assert set(d.keys()) == {"torch_grip2", "torch_prism"}
    for entry in d.values():
        assert len(entry["importance"]) == 30
        assert len(entry["importance_k"]) == 30

    sel_df = prismdata.load_bundle(outcome, "selections", bundle_id)
    sd = transforms.selections_to_dict(sel_df)
    assert set(sd.keys()) == {"torch_grip2", "torch_prism"}
    for entry in sd.values():
        assert len(entry["selected"]) == 30


@pytest.mark.parametrize("outcome", OUTCOME_TYPES)
def test_parameters_to_dict_strips_identifier_groups(outcome):
    trial = prismdata.trials(outcome).row(0, named=True)
    raw = prismdata.parameters(outcome).filter(pl.col("bundle_id") == trial["bundle_id"])
    d = transforms.parameters_to_dict(raw)

    assert "_meta" not in d
    assert "_prereq" not in d
    assert d  # a trial bundle has at least one non-identifier group

    for group_values in d.values():
        for value in group_values.values():
            assert not isinstance(value, dict)


@pytest.mark.parametrize("outcome", OUTCOME_TYPES)
def test_parameters_to_dict_covers_every_group_and_name(outcome):
    x_bundle_id = prismdata.list_bundle_ids(outcome, "X")[0]
    raw = prismdata.parameters(outcome).filter(pl.col("bundle_id") == x_bundle_id)
    d = transforms.parameters_to_dict(raw)

    non_identifier = raw.filter(~pl.col("par_group").is_in(["_meta", "_prereq"]))
    expected_pairs = set(non_identifier.select("par_group", "par_name").iter_rows())
    actual_pairs = {(group, name) for group, names in d.items() for name in names}
    assert actual_pairs == expected_pairs


@pytest.mark.parametrize("outcome", OUTCOME_TYPES)
def test_load_trial_bundle_joins_correct_bundles(outcome):
    trial = prismdata.trials(outcome).row(0, named=True)
    tb = transforms.load_trial_bundle(outcome, trial["bundle_id"])

    assert tb["outcome"] == outcome
    assert tb["bundle_id"] == trial["bundle_id"]
    assert tb["x_bundle_id"] == trial["x_bundle_id"]
    assert tb["xk_bundle_id"] == trial["xk_bundle_id"]

    # X/Xk/y come from the correct bundles and are row-aligned.
    expected_X = prismdata.load_bundle(outcome, "X", trial["x_bundle_id"])
    expected_Xk = prismdata.load_bundle(outcome, "Xk", trial["xk_bundle_id"])
    expected_y = prismdata.load_bundle(outcome, "y", trial["bundle_id"])
    assert tb["X"].equals(expected_X)
    assert tb["Xk"].equals(expected_Xk)
    assert tb["y"].equals(expected_y)

    # importances/selections match the trial's own frames, keyed by importance_method.
    imp = prismdata.load_bundle(outcome, "importances", trial["bundle_id"])
    sel = prismdata.load_bundle(outcome, "selections", trial["bundle_id"])
    assert tb["importances"] == transforms.importances_to_dict(imp)
    assert tb["selections"] == transforms.selections_to_dict(sel)

    # parameters is the union of the X-bundle's, Xk-bundle's, and trial's own groups.
    assert "X" in tb["parameters"]
    assert "Xk" in tb["parameters"]
    assert "importances" in tb["parameters"]
    assert "selections" in tb["parameters"]


def test_load_trial_bundle_keeps_both_methods_when_a_bundle_shares_one_fit():
    outcome = "continuous"
    sel = prismdata.load(outcome, "selections")
    multi_method_bundles = (
        sel.group_by("bundle_id")
        .agg(pl.col("importance_method").n_unique().alias("n"))
        .filter(pl.col("n") > 1)["bundle_id"]
    )
    bundle_id = multi_method_bundles[0]

    tb = transforms.load_trial_bundle(outcome, bundle_id)
    assert set(tb["importances"].keys()) == {"torch_grip2", "torch_prism"}
    assert set(tb["selections"].keys()) == {"torch_grip2", "torch_prism"}
    # the shared parameter group names the underlying fit, not either individual method.
    assert tb["parameters"]["importances"]["method"] == "torch_prism_grip2"


def test_split_and_combine_importance_arrays_round_trip():
    rng = np.random.default_rng(0)
    arr = rng.normal(size=60)

    importance, importance_k = transforms.split_importance_array(arr)
    assert len(importance) == len(importance_k) == 30
    recombined = transforms.combine_importance_arrays(importance, importance_k)
    assert np.array_equal(recombined, arr)

    importance = rng.normal(size=17)
    importance_k = rng.normal(size=17)
    combined = transforms.combine_importance_arrays(importance, importance_k)
    split_importance, split_importance_k = transforms.split_importance_array(combined)
    assert np.array_equal(split_importance, importance)
    assert np.array_equal(split_importance_k, importance_k)


def test_heteroknockoffpy_kwargs_strips_bookkeeping_keys():
    raw = {"method": "LASSO", "standardize_y": True, "n_splits": 5, "max_iter": 4000}
    kwargs = transforms.heteroknockoffpy_kwargs("LASSO", raw, p=30)
    assert kwargs == {"n_splits": 5, "max_iter": 4000}


def test_heteroknockoffpy_kwargs_ranger_gini_passes_through_unchanged():
    raw = {"method": "ranger_gini", "standardize_y": True, "respect_unordered_factors": "partition"}
    kwargs = transforms.heteroknockoffpy_kwargs("ranger_gini", raw, p=30)
    assert kwargs == {"respect_unordered_factors": "partition"}


@pytest.mark.parametrize("importance_method", ["torch_grip2", "torch_prism"])
def test_heteroknockoffpy_kwargs_builds_layers_matching_silverknockoff_formula(importance_method):
    raw = {
        "method": "torch_prism_grip2",
        "standardize_y": True,
        "local_grad_method": "auto_diff",
        "first_layer_ratio": 0.6934912420343522,
        "layer_count": 3,
        "layer_ratio": 0.2373494459019788,
        "epochs": 1043,
        "n_blocks": 34,
        "calibrate": True,
        "model_type": "mlp",
        "n_warmup": 13,
        "learning_rate": 0.061,
    }
    p = 30
    kwargs = transforms.heteroknockoffpy_kwargs(importance_method, raw, p=p)

    # matches silverknockoff's calculatorOps.py CalculateImportances formula exactly
    expected_layers = [max(1, round(p * raw["first_layer_ratio"]))]
    for _ in range(raw["layer_count"] - 1):
        expected_layers.append(max(1, round(expected_layers[-1] * raw["layer_ratio"])))
    assert kwargs["layers"] == expected_layers

    for stripped in ("method", "standardize_y", "local_grad_method", "first_layer_ratio", "layer_count", "layer_ratio"):
        assert stripped not in kwargs
    for passthrough in ("epochs", "n_blocks", "calibrate", "model_type", "n_warmup", "learning_rate"):
        assert kwargs[passthrough] == raw[passthrough]


@pytest.mark.parametrize("importance_method", ["xgb_prism", "xgb_score"])
def test_heteroknockoffpy_kwargs_nests_xgb_model_kwargs(importance_method):
    raw = {
        "method": importance_method,
        "standardize_y": True,
        "max_depth": 6,
        "learning_rate": 0.005,
        "subsample": 0.87,
        "colsample_bytree": 0.94,
        "reg_alpha": 0.19,
        "reg_lambda": 1.25e-5,
        "gamma": 1.88,
        "n_estimators": 728,
        "min_child_weight": 5.33,
        "importance_type": "gain",
    }
    kwargs = transforms.heteroknockoffpy_kwargs(importance_method, raw, p=30)

    assert "method" not in kwargs
    assert "standardize_y" not in kwargs
    assert kwargs["importance_type"] == "gain"  # a real top-level kwarg, not xgb model-only
    assert set(kwargs["model_kwargs"]) == {
        "max_depth", "learning_rate", "subsample", "colsample_bytree",
        "reg_alpha", "reg_lambda", "gamma", "n_estimators", "min_child_weight",
    }
    assert kwargs["model_kwargs"]["max_depth"] == 6


@pytest.mark.parametrize(
    "importance_method",
    ("LASSO", "ranger_gini", "torch_grip2", "torch_prism", "xgb_prism"),
)
def test_heteroknockoffpy_kwargs_actually_works_against_heteroknockoffpy(importance_method):
    """Integration check: every method's prepared kwargs must actually run
    against the real heteroknockoffpy functions without raising. Skipped
    unless the optional `reproduce` dependency group is installed."""
    heteroknockoffpy_importance = pytest.importorskip("heteroknockoffpy.importance")

    DISPATCH = {
        "LASSO": heteroknockoffpy_importance.lassoImportances,
        "ranger_gini": heteroknockoffpy_importance.rangerGiniImportances,
        "torch_grip2": heteroknockoffpy_importance.grip2Importances,
        "torch_prism": heteroknockoffpy_importance.torchPrismImportances,
        "xgb_prism": heteroknockoffpy_importance.xgbPrismImportances,
    }

    outcome = "linear" if importance_method == "LASSO" else "continuous"
    sel = prismdata.load(outcome, "selections")
    bundle_id = sel.filter(pl.col("importance_method") == importance_method)["bundle_id"][0]
    tb = transforms.load_trial_bundle(outcome, bundle_id)

    kwargs = transforms.heteroknockoffpy_kwargs(
        importance_method, tb["parameters"]["importances"], p=tb["X"].width
    )
    arr = DISPATCH[importance_method](tb["X"], tb["Xk"], tb["y"], **kwargs)
    assert arr.shape == (2 * tb["X"].width,)


def test_raw_schemas_cover_every_well_defined_table():
    # X/Xk have no fixed schema (see transforms module docstring) and are
    # deliberately absent from RAW_SCHEMAS.
    for artifact in ("y", "importances", "selections"):
        assert artifact in transforms.RAW_SCHEMAS
    assert "X" not in transforms.RAW_SCHEMAS
    assert "Xk" not in transforms.RAW_SCHEMAS
    for table in ("index", "parameters_table", "results", "X_modes"):
        assert table in transforms.RAW_SCHEMAS
