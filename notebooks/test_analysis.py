import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd

    return Path, json, pd, plt


@app.cell
def _(Path, json):
    print(Path.cwd())
    base_diarization_path = Path("data/test/base_diarization.json")
    turn_end_path = Path("data/test/turn_end_updated.json")
    first_stitched_path = Path("data/test/first_stitched.json")

    base_diarization_text = base_diarization_path.read_text()
    turn_end_text = turn_end_path.read_text()
    first_stitched_text = first_stitched_path.read_text()

    try:
        base_diarization_data = json.loads(base_diarization_text)
    except json.JSONDecodeError:
        base_diarization_data = [json.loads(line) for line in base_diarization_text.splitlines() if line.strip()]

    try:
        turn_end_data = json.loads(turn_end_text)
    except:
        turn_end_data = [json.loads(line) for line in turn_end_text.splitlines() if line.strip()]

    try:
        first_stitched_data = json.loads(first_stitched_text)
    except:
        first_stitched_data = [json.loads(line) for line in first_stitched_text.splitlines() if line.strip()]
    return (
        base_diarization_data,
        base_diarization_path,
        first_stitched_data,
        first_stitched_path,
        turn_end_data,
        turn_end_path,
    )


@app.cell
def _(base_diarization_data, pd, plt):
    def collect_time_segments(obj):
        segments = []
        if isinstance(obj, dict):
            if "start_time" in obj and "end_time" in obj:
                segments.append(
                    {
                        "start_time": obj.get("start_time"),
                        "end_time": obj.get("end_time"),
                    }
                )
            for value in obj.values():
                segments.extend(collect_time_segments(value))
        elif isinstance(obj, list):
            for item in obj:
                segments.extend(collect_time_segments(item))
        return segments

    diarization_segments = collect_time_segments(base_diarization_data)

    diarization_df = pd.DataFrame(diarization_segments)
    diarization_df["start_time"] = pd.to_numeric(diarization_df["start_time"], errors="coerce")
    diarization_df["end_time"] = pd.to_numeric(diarization_df["end_time"], errors="coerce")
    diarization_df = (
        diarization_df.dropna(subset=["start_time", "end_time"])
        .query("end_time >= start_time")
        .sort_values("start_time")
        .reset_index(drop=True)
    )

    diarization_plot_df = diarization_df.assign(
        duration=diarization_df["end_time"] - diarization_df["start_time"],
        gap=diarization_df["start_time"] - diarization_df["end_time"].shift(1),
    ).dropna(subset=["gap"])

    plt.figure(figsize=(9, 6))
    plt.scatter(
        diarization_plot_df["duration"],
        diarization_plot_df["gap"],
        alpha=0.7,
        color="#4C78A8",
        edgecolors="white",
        linewidths=0.5,
    )
    plt.title("Diarization Segment Duration vs Gap")
    plt.xlabel("Duration (end_time - start_time)")
    plt.ylabel("Gap from Prior Segment (start_time - previous end_time)")
    plt.xlim(right=12, left=0)
    plt.ylim(top=2, bottom=0)
    plt.grid(True, alpha=0.3)
    plt.gca()
    return


@app.cell
def _(pd, plt, turn_end_data):
    def collect_field_values(obj, field_name):
        collected = []
        if isinstance(obj, dict):
            if field_name in obj:
                collected.append(obj[field_name])
            for value in obj.values():
                collected.extend(collect_field_values(value, field_name))
        elif isinstance(obj, list):
            for item in obj:
                collected.extend(collect_field_values(item, field_name))
        return collected

    end_turn_probability_values = collect_field_values(turn_end_data, "end_of_turn_probability")
    end_turn_probability_series = pd.to_numeric(pd.Series(end_turn_probability_values), errors="coerce").dropna()
    end_turn_probability_df = pd.DataFrame({"end_of_turn_probability": end_turn_probability_series})

    end_turn_probability_df.head()

    plt.figure(figsize=(8, 5))
    plt.hist(end_turn_probability_df["end_of_turn_probability"], bins=30, color="#4C78A8", edgecolor="white")
    plt.title("Histogram of end_turn_probability")
    plt.xlabel("end_turn_probability")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.3)
    plt.gca()
    return


@app.cell
def _(
    base_diarization_data,
    base_diarization_path,
    first_stitched_data,
    first_stitched_path,
    pd,
    turn_end_data,
    turn_end_path,
):
    record_count_df = pd.DataFrame(
        {
            "file": [
                str(base_diarization_path),
                str(turn_end_path),
                str(first_stitched_path),
            ],
            "record_count": [
                len(base_diarization_data) if isinstance(base_diarization_data, list) else 1,
                len(turn_end_data) if isinstance(turn_end_data, list) else 1,
                len(first_stitched_data) if isinstance(first_stitched_data, list) else 1,
            ],
        }
    )
    return (record_count_df,)


@app.cell
def _(plt, record_count_df):
    plt.figure(figsize=(8, 5))
    plt.bar(
        record_count_df["file"],
        record_count_df["record_count"],
        color=["#4C78A8", "#F58518"],
        edgecolor="white",
    )
    plt.title("Record Counts by JSON File")
    plt.xlabel("File")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
