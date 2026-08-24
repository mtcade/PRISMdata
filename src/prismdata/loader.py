"""Read access to the simulation data in ``data/``.

Per outcome type (see :data:`OutcomeType`), ``data/{outcome}/`` holds:

- ``{X,Xk,y,importances,selections}/{bundle_id}.parquet`` -- one file per
  bundle, named by that bundle's ``par_bundle_idn`` (as lowercase hex).
- ``index.parquet`` -- the join table. One row per ``X``/``Xk``/``trial``
  bundle; a ``trial`` row's ``X_path``/``Xk_path`` name the ``X``/``Xk``
  bundle it was generated from, and its ``y_path``/``importances_path``/
  ``selections_path`` all equal its own ``bundle_id``. Use :func:`index`
  or :func:`trials` rather than joining artifacts by filename directly.
- ``parameters_table.parquet`` -- per-bundle generation parameters.
- ``results.parquet`` -- per-bundle ``loss_value``.
- ``X_modes.parquet`` -- per-``X``-bundle categorical mode metadata.

This module returns frames in their raw on-disk shape. See
:mod:`prismdata.transforms` for friendlier, use-ready conversions.
"""

from pathlib import Path
from typing import Literal

import polars as pl

OutcomeType = Literal["linear", "continuous", "count", "categorical"]
Artifact = Literal["X", "Xk", "y", "importances", "selections"]
ParStage = Literal["X", "Xk", "trial"]


def data_dir() -> Path:
    """Root of the data tree (``<repo>/data``)."""
    return Path(__file__).resolve().parents[2] / "data"


def _outcome_dir(outcome: OutcomeType) -> Path:
    return data_dir() / outcome


def _artifact_dir(outcome: OutcomeType, artifact: Artifact) -> Path:
    return _outcome_dir(outcome) / artifact


def bundle_path(outcome: OutcomeType, artifact: Artifact, bundle_id: str) -> Path:
    """Path to a single bundle's parquet file."""
    return _artifact_dir(outcome, artifact) / f"{bundle_id}.parquet"


def list_bundle_ids(outcome: OutcomeType, artifact: Artifact) -> list[str]:
    """Bundle ids (parquet filename stems) available for ``outcome``/``artifact``."""
    return sorted(p.stem for p in _artifact_dir(outcome, artifact).glob("*.parquet"))


def load_bundle(outcome: OutcomeType, artifact: Artifact, bundle_id: str) -> pl.DataFrame:
    """Load a single bundle's parquet file."""
    return pl.read_parquet(bundle_path(outcome, artifact, bundle_id))


def scan(outcome: OutcomeType, artifact: Artifact) -> pl.LazyFrame:
    """Lazily scan every bundle for ``outcome``/``artifact``, tagged with ``bundle_id``."""
    lf = pl.scan_parquet(_artifact_dir(outcome, artifact) / "*.parquet", include_file_paths="_path")
    return lf.with_columns(
        pl.col("_path").str.extract(r"([^/]+)\.parquet$", 1).alias("bundle_id")
    ).drop("_path")


def load(outcome: OutcomeType, artifact: Artifact) -> pl.DataFrame:
    """Eagerly load and concatenate every bundle for ``outcome``/``artifact``."""
    return scan(outcome, artifact).collect()


def _with_bundle_id(df: pl.DataFrame, *, column: str = "par_bundle_idn") -> pl.DataFrame:
    """Replace a binary ``par_bundle_idn`` column with its hex ``bundle_id`` string."""
    return df.with_columns(pl.col(column).bin.encode("hex").alias("bundle_id")).drop(column)


def _path_to_bundle_id(col: str) -> pl.Expr:
    return pl.col(col).str.extract(r"([^/]+)\.parquet$", 1)


def index(outcome: OutcomeType) -> pl.DataFrame:
    """The full bundle index for ``outcome`` (``X``, ``Xk``, and ``trial`` rows).

    ``bundle_id`` is this row's own id; for ``trial`` rows, ``x_bundle_id``/
    ``xk_bundle_id`` name the ``X``/``Xk`` bundle used to generate it.
    """
    df = pl.read_parquet(_outcome_dir(outcome) / "index.parquet")
    df = _with_bundle_id(df)
    return df.with_columns(
        _path_to_bundle_id("X_path").alias("x_bundle_id"),
        _path_to_bundle_id("Xk_path").alias("xk_bundle_id"),
    ).select(
        "bundle_id", "par_stage", "trial_idx", "x_bundle_id", "xk_bundle_id"
    )


def trials(outcome: OutcomeType) -> pl.DataFrame:
    """Bundle index filtered to ``trial`` rows: one per ``y``/``importances``/``selections`` bundle."""
    return index(outcome).filter(pl.col("par_stage") == "trial").drop("par_stage")


def parameters(outcome: OutcomeType) -> pl.DataFrame:
    """Long-format generation parameters, one bundle per ``bundle_id``."""
    return _with_bundle_id(pl.read_parquet(_outcome_dir(outcome) / "parameters_table.parquet"))


def results(outcome: OutcomeType) -> pl.DataFrame:
    """Per-bundle ``loss_value``."""
    return _with_bundle_id(pl.read_parquet(_outcome_dir(outcome) / "results.parquet"))


def x_modes(outcome: OutcomeType) -> pl.DataFrame:
    """Per-``X``-bundle categorical mode metadata."""
    return _with_bundle_id(pl.read_parquet(_outcome_dir(outcome) / "X_modes.parquet"))
