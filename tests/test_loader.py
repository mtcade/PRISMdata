from typing import get_args

import polars as pl
import pytest

import prismdata

OUTCOME_TYPES = get_args(prismdata.OutcomeType)
ARTIFACTS = get_args(prismdata.Artifact)


@pytest.mark.parametrize("outcome", OUTCOME_TYPES)
def test_artifact_dirs_have_bundles(outcome):
    for artifact in ARTIFACTS:
        ids = prismdata.list_bundle_ids(outcome, artifact)
        assert len(ids) > 0


@pytest.mark.parametrize("outcome", OUTCOME_TYPES)
def test_load_bundle_matches_list(outcome):
    bundle_id = prismdata.list_bundle_ids(outcome, "y")[0]
    df = prismdata.load_bundle(outcome, "y", bundle_id)
    assert df.height > 0
    assert "y" in df.columns


@pytest.mark.parametrize("outcome", OUTCOME_TYPES)
def test_load_concatenates_all_bundles(outcome):
    ids = prismdata.list_bundle_ids(outcome, "importances")
    df = prismdata.load(outcome, "importances")
    assert df["bundle_id"].n_unique() == len(ids)


@pytest.mark.parametrize("outcome", OUTCOME_TYPES)
def test_trials_join_to_existing_X_and_Xk_bundles(outcome):
    trials = prismdata.trials(outcome)
    y_ids = set(prismdata.list_bundle_ids(outcome, "y"))
    x_ids = set(prismdata.list_bundle_ids(outcome, "X"))
    xk_ids = set(prismdata.list_bundle_ids(outcome, "Xk"))

    assert set(trials["bundle_id"]) == y_ids
    assert set(trials["x_bundle_id"]).issubset(x_ids)
    assert set(trials["xk_bundle_id"]).issubset(xk_ids)


@pytest.mark.parametrize("outcome", OUTCOME_TYPES)
def test_trial_joins_to_correct_X_shape(outcome):
    trial = prismdata.trials(outcome).row(0, named=True)
    y = prismdata.load_bundle(outcome, "y", trial["bundle_id"])
    X = prismdata.load_bundle(outcome, "X", trial["x_bundle_id"])
    Xk = prismdata.load_bundle(outcome, "Xk", trial["xk_bundle_id"])
    assert y.height == X.height == Xk.height
    assert X.shape == Xk.shape


@pytest.mark.parametrize("outcome", OUTCOME_TYPES)
def test_parameters_and_results_key_on_known_bundle_ids(outcome):
    all_ids = set(prismdata.index(outcome)["bundle_id"])
    params = prismdata.parameters(outcome)
    res = prismdata.results(outcome)
    assert set(params["bundle_id"]).issubset(all_ids)
    assert set(res["bundle_id"]).issubset(all_ids)


@pytest.mark.parametrize("outcome", OUTCOME_TYPES)
def test_every_trial_bundle_is_complete(outcome):
    """Every trial's y/importances/selections/X/Xk files all exist on disk,
    and no artifact directory holds a file the index doesn't know about."""
    trials = prismdata.trials(outcome)
    y_ids = set(prismdata.list_bundle_ids(outcome, "y"))
    imp_ids = set(prismdata.list_bundle_ids(outcome, "importances"))
    sel_ids = set(prismdata.list_bundle_ids(outcome, "selections"))
    x_ids = set(prismdata.list_bundle_ids(outcome, "X"))
    xk_ids = set(prismdata.list_bundle_ids(outcome, "Xk"))

    trial_ids = set(trials["bundle_id"])
    assert trial_ids == y_ids == imp_ids == sel_ids
    assert set(trials["x_bundle_id"]).issubset(x_ids)
    assert set(trials["xk_bundle_id"]).issubset(xk_ids)

    idx = prismdata.index(outcome)
    assert set(idx.filter(pl.col("par_stage") == "X")["bundle_id"]) == x_ids
    assert set(idx.filter(pl.col("par_stage") == "Xk")["bundle_id"]) == xk_ids


def test_unknown_outcome_lists_no_bundles_and_fails_on_load():
    # No manual validation: an unknown outcome is just a directory that
    # doesn't exist. Listing sees nothing; loading fails when polars
    # actually tries to expand the (empty) glob.
    assert prismdata.list_bundle_ids("not_an_outcome", "X") == []
    with pytest.raises(pl.exceptions.ComputeError):
        prismdata.load("not_an_outcome", "X")


def test_unknown_artifact_lists_no_bundles_and_fails_on_load():
    assert prismdata.list_bundle_ids("linear", "not_an_artifact") == []
    with pytest.raises(pl.exceptions.ComputeError):
        prismdata.load("linear", "not_an_artifact")
