import marimo

__generated_with = "0.22.4"
app = marimo.App()


@app.cell
def _():
    return


@app.cell
def _():
    try:
        pd
    except NameError:
        import pandas as pd

    try:
        alt
    except NameError:
        import altair as alt

    try:
        Path
    except NameError:
        from pathlib import Path

    alt.data_transformers.disable_max_rows()

    conf_wder_scatter_path = Path("../data/test/conf_wder.csv")
    conf_wder_scatter_df = pd.read_csv(conf_wder_scatter_path)

    conf_wder_scatter_columns = list(conf_wder_scatter_df.columns)

    confidence_scatter_col = (
        "confidence"
        if "confidence" in conf_wder_scatter_columns
        else next((c for c in conf_wder_scatter_columns if "conf" in c.lower()), None)
    )

    success_rate_scatter_col = (
        "success_rate"
        if "success_rate" in conf_wder_scatter_columns
        else next(
            (
                c
                for c in conf_wder_scatter_columns
                if "success" in c.lower() and "rate" in c.lower()
            ),
            None,
        )
    )

    if confidence_scatter_col is None or success_rate_scatter_col is None:
        raise ValueError(
            f"Could not find confidence/success_rate columns in {conf_wder_scatter_columns}"
        )

    conf_wder_scatter_plot_df = conf_wder_scatter_df[
        [confidence_scatter_col, success_rate_scatter_col]
    ].copy()

    conf_wder_scatter_plot_df[confidence_scatter_col] = pd.to_numeric(
        conf_wder_scatter_plot_df[confidence_scatter_col], errors="coerce"
    )
    conf_wder_scatter_plot_df[success_rate_scatter_col] = pd.to_numeric(
        conf_wder_scatter_plot_df[success_rate_scatter_col], errors="coerce"
    )

    conf_wder_scatter_plot_df = conf_wder_scatter_plot_df.dropna()
    return (
        Path,
        alt,
        conf_wder_scatter_plot_df,
        confidence_scatter_col,
        pd,
        success_rate_scatter_col,
    )


@app.cell
def _(
    alt,
    conf_wder_scatter_plot_df,
    confidence_scatter_col,
    success_rate_scatter_col,
):
    alt.Chart(conf_wder_scatter_plot_df).mark_circle(size=60, opacity=0.6).encode(
        x=alt.X(confidence_scatter_col, title="Confidence"),
        y=alt.Y(success_rate_scatter_col, title="Success rate"),
        tooltip=[
            alt.Tooltip(confidence_scatter_col, title="Confidence", format=".4f"),
            alt.Tooltip(success_rate_scatter_col, title="Success rate", format=".4f"),
        ],
        color=alt.value("#4c78a8"),
    ).properties(
        title="Success Rate vs Confidence"
    ).interactive()
    return


if __name__ == "__main__":
    app.run()
