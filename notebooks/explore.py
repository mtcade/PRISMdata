import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    from typing import get_args

    import marimo as mo
    import prismdata
    from prismdata import transforms

    return get_args, mo, prismdata, transforms


@app.cell
def _(get_args, mo, prismdata):
    outcome = mo.ui.dropdown(
        options=get_args(prismdata.OutcomeType), value="linear", label="outcome type"
    )
    outcome
    return (outcome,)


@app.cell
def _(mo, outcome, prismdata):
    trials = prismdata.trials(outcome.value)
    mo.md(f"**{trials.height}** trial bundles for `{outcome.value}`")
    return (trials,)


@app.cell
def _(trials):
    trials.head(10)
    return


@app.cell
def _(mo, outcome, prismdata, trials):
    trial = trials.row(0, named=True)
    X = prismdata.load_bundle(outcome.value, "X", trial["x_bundle_id"])
    y = prismdata.load_bundle(outcome.value, "y", trial["bundle_id"])
    mo.md(f"first trial `{trial['bundle_id']}`: X {X.shape}, y {y.shape}")
    return X, trial


@app.cell
def _(X):
    X.head(5)
    return


@app.cell
def _(mo, outcome, prismdata, trial, transforms):
    imp = prismdata.load_bundle(outcome.value, "importances", trial["bundle_id"])
    sel = prismdata.load_bundle(outcome.value, "selections", trial["bundle_id"])
    mo.md(f"""
    importances: `{transforms.importances_to_dict(imp)}`

    selections: `{transforms.selections_to_dict(sel)}`
    """)
    return


@app.cell
def _(outcome, prismdata):
    importances = prismdata.load(outcome.value, "importances")
    importances.group_by("importance_method").len()
    return


if __name__ == "__main__":
    app.run()
