from typing import get_args

import polars as pl
import pytest

import prismdata
from prismdata import transforms

OUTCOME_TYPES = get_args(prismdata.OutcomeType)


@pytest.mark.parametrize("outcome", OUTCOME_TYPES)
def test_importances_to_dict_covers_every_feature(outcome):
    trial = prismdata.trials(outcome).row(0, named=True)
    df = prismdata.load_bundle(outcome, "importances", trial["bundle_id"])
    d = transforms.importances_to_dict(df)

    assert set(d.keys()) == {"importance_method", "importance", "importance_k"}
    assert isinstance(d["importance_method"], str)
    assert set(d["importance"].keys()) == set(df["feature_idx"].to_list())
    assert set(d["importance_k"].keys()) == set(df["feature_idx"].to_list())


@pytest.mark.parametrize("outcome", OUTCOME_TYPES)
def test_selections_to_dict_covers_every_feature(outcome):
    trial = prismdata.trials(outcome).row(0, named=True)
    df = prismdata.load_bundle(outcome, "selections", trial["bundle_id"])
    d = transforms.selections_to_dict(df)

    assert set(d.keys()) == {"importance_method", "fdr_target", "selected"}
    assert isinstance(d["fdr_target"], float)
    assert set(d["selected"].keys()) == set(df["feature_idx"].to_list())
    assert all(isinstance(v, bool) for v in d["selected"].values())


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


def test_raw_schemas_cover_every_artifact_and_outcome_table():
    from typing import get_args

    from prismdata.loader import Artifact

    for artifact in get_args(Artifact):
        assert artifact in transforms.RAW_SCHEMAS
    for table in ("index", "parameters_table", "results", "X_modes"):
        assert table in transforms.RAW_SCHEMAS
