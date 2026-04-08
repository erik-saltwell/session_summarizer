import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import json
    import matplotlib.pyplot as plt

    _path = Path("../data/test/identity_stitched.json")
    with _path.open("r", encoding="utf-8") as _f:
        _data = json.load(_f)

    def _first_present(_d, _keys):
        for _k in _keys:
            if _k in _d:
                return _d[_k]
        return None

    def _to_float(_value):
        if isinstance(_value, (int, float)):
            return float(_value)
        if isinstance(_value, str):
            try:
                return float(_value)
            except ValueError:
                return None
        return None

    _word_points = []

    def _walk(_obj):
        if isinstance(_obj, dict):
            _text = _first_present(_obj, ["word", "text", "punctuated_word", "token", "value"])
            _start = _first_present(_obj, ["start", "start_time", "begin", "offset"])
            _confidence = _first_present(_obj, ["confidence", "score", "conf"])
            _start_num = _to_float(_start)
            _confidence_num = _to_float(_confidence)

            if _start_num is not None and _confidence_num is not None and isinstance(_text, str):
                _word_points.append((_start_num, _confidence_num, _text))

            for _v in _obj.values():
                _walk(_v)
        elif isinstance(_obj, list):
            for _item in _obj:
                _walk(_item)

    _walk(_data)

    _seen = set()
    _word_points = [
        _row for _row in _word_points
        if not ((_row[0], _row[1], _row[2]) in _seen or _seen.add((_row[0], _row[1], _row[2])))
    ]

    if not _word_points:
        raise ValueError("No word-level entries with both start time and confidence were found in ../data/test/identity_stiched.json")

    _x = [_row[0] for _row in _word_points]
    _y = [_row[1] for _row in _word_points]
    _labels = [_row[2] for _row in _word_points]

    plt.figure(figsize=(10, 6))
    _scatter = plt.scatter(
        _x,
        _y,
        c=_y,
        cmap="viridis",
        alpha=0.8,
        s=36,
        edgecolors="white",
        linewidths=0.4,
    )
    plt.xlabel("Start time (s)")
    plt.ylabel("Confidence score")
    plt.title("Word Confidence Score vs Start Time")
    plt.grid(True, alpha=0.3)

    _colorbar = plt.colorbar(_scatter)
    _colorbar.set_label("Confidence score")

    if len(_word_points) <= 20:
        for _sx, _sy, _label in _word_points:
            plt.annotate(_label, (_sx, _sy), xytext=(4, 4), textcoords="offset points", fontsize=8, alpha=0.8)
    else:
        _lowest = sorted(_word_points, key=lambda _r: _r[1])[:10]
        for _sx, _sy, _label in _lowest:
            plt.annotate(_label, (_sx, _sy), xytext=(4, 4), textcoords="offset points", fontsize=8, alpha=0.8)

    plt.gca()
    return


if __name__ == "__main__":
    app.run()
