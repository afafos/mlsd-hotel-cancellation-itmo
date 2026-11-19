"""Rule-based baselines B0 and B1 on validation windows; the test period is never scored.

B0 ranks candidates by the nearest arrival or at random. B1 ranks by the smoothed train rate
of a segment. Snapshots are built only up to the end of the calibration window, so test
labels cannot enter any table here. Ties are resolved in expectation, never by row order.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hotel_cancellation.audit import DERIVED_COLUMNS, MAIN_HORIZON, SPLITS, TEST_START
from hotel_cancellation.data import DATA_SHA256, RAW_PATH, download, load_bookings
from hotel_cancellation.metrics import average_precision, daily_top_k, roc_auc, summarize_top_k
from hotel_cancellation.report import COLORS, HOTELS, prepare_output, save_figure, save_table
from hotel_cancellation.snapshots import build_snapshots

OUTPUT_DIR = Path("reports/baselines")
K = 20
KS = (10, 20, 30, 50)
ALPHAS = (10, 30, 100)
MIN_SUPPORT = 30
# A window starting on day d is fitted on snapshots up to d - 15 days, i.e. with a 14-day gap,
# and only on labels known before d. Its own labels are used if known 15 days after its end.
GAP = pd.Timedelta(days=MAIN_HORIZON + 1)
TUNE = (pd.Timestamp("2017-01-01"), pd.Timestamp("2017-02-14"))
CALIBRATION = (pd.Timestamp("2017-03-01"), pd.Timestamp("2017-04-16"))
BACKTEST_STARTS = ("2016-03-01", "2016-05-01", "2016-07-01", "2016-09-01", "2016-11-01")
WINDOW_DAYS = 45
AGE_BINS = (0, 7, 30, 90, np.inf)
AGE_LABELS = ("1-7", "8-30", "31-90", ">90")
ARRIVAL_BINS = (0, 3, 7, MAIN_HORIZON)
ARRIVAL_LABELS = ("1-3", "4-7", "8-14")
B1_KEYS = ("hotel", "market_segment", "deposit_type", "customer_type", "age_bin")
B1_VARIANTS = {
    "B1": B1_KEYS,
    "B1 без депозита": tuple(key for key in B1_KEYS if key != "deposit_type"),
    "B1 + дни до заезда": (*B1_KEYS, "arrival_bin"),
}
B0_RANDOM = "B0: случайный порядок"
B0_NEAREST = "B0: ближайший заезд"
CONSTANT = "Доля отмен отеля в train"


def candidates(df: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    """Daily candidates up to ``end`` with the booking attributes the baselines use."""
    if end >= TEST_START:
        msg = f"Baselines must not build snapshots in the test period: {end:%Y-%m-%d}"
        raise ValueError(msg)
    snaps = build_snapshots(df, MAIN_HORIZON, end=end)
    rows = df.loc[snaps["booking_idx"]]
    duplicate = df.drop(columns=DERIVED_COLUMNS).duplicated()
    return snaps.assign(
        market_segment=rows["market_segment"].to_numpy(),
        deposit_type=rows["deposit_type"].to_numpy(),
        customer_type=rows["customer_type"].to_numpy(),
        age_bin=pd.cut(snaps["days_since_booking"], AGE_BINS, labels=AGE_LABELS).astype(str),
        arrival_bin=pd.cut(snaps["days_to_arrival"], ARRIVAL_BINS, labels=ARRIVAL_LABELS).astype(
            str
        ),
        status_date=pd.to_datetime(rows["reservation_status_date"]).to_numpy(),
        duplicate=duplicate.loc[snaps["booking_idx"]].to_numpy(),
    )


def history(frame: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """Training snapshots for a window starting on ``start``: gap and mature labels only."""
    in_time = frame["snapshot_date"].le(start - GAP)
    return frame[in_time & frame["status_date"].lt(start)]


def window(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Evaluation snapshots whose outcome is known 15 days after the window end."""
    in_time = frame["snapshot_date"].between(start, end)
    return frame[in_time & frame["status_date"].lt(end + GAP)]


def fit_rates(
    train: pd.DataFrame, keys: Sequence[str], alpha: float
) -> tuple[pd.DataFrame, pd.Series]:
    """Smoothed segment rates ``(positives + alpha * hotel rate) / (snapshots + alpha)``.

    Segments with fewer than ``MIN_SUPPORT`` unique bookings fall back to the hotel rate.
    """
    hotel_rate = train.groupby("hotel")["label"].mean()
    segments = (
        train.groupby(list(keys), observed=True)
        .agg(
            snapshots=("label", "size"),
            positives=("label", "sum"),
            bookings=("booking_idx", "nunique"),
        )
        .reset_index()
    )
    prior = segments["hotel"].map(hotel_rate)
    segments["rate"] = (segments["positives"] + alpha * prior) / (segments["snapshots"] + alpha)
    segments["supported"] = segments["bookings"].ge(MIN_SUPPORT)
    return segments, hotel_rate


def score_rates(
    frame: pd.DataFrame, keys: Sequence[str], segments: pd.DataFrame, hotel_rate: pd.Series
) -> pd.Series:
    supported = segments[segments["supported"]]
    rate = frame[list(keys)].merge(supported, on=list(keys), how="left")["rate"].to_numpy()
    fallback = frame["hotel"].map(hotel_rate).to_numpy()
    return pd.Series(np.where(np.isnan(rate), fallback, rate), index=frame.index)


def scored(train: pd.DataFrame, evaluation: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Scores of every baseline for the evaluation snapshots; B1 is fitted on ``train``."""
    result = evaluation.copy()
    result[B0_RANDOM] = 0.0
    result[B0_NEAREST] = -evaluation["days_to_arrival"].astype(float)
    hotel_rate = train.groupby("hotel")["label"].mean()
    result[CONSTANT] = result["hotel"].map(hotel_rate)
    for name, keys in B1_VARIANTS.items():
        segments, rate = fit_rates(train, keys, alpha)
        result[name] = score_rates(result, keys, segments, rate)
    return result


def evaluate(frame: pd.DataFrame, models: Sequence[str]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for hotel in HOTELS:
        part = frame[frame["hotel"].eq(hotel)]
        labels = part["label"].to_numpy(dtype=np.float64)
        for model in models:
            values = part[model].to_numpy(dtype=np.float64)
            row: dict[str, float | str] = {"model": model, "hotel": hotel}
            row |= summarize_top_k(daily_top_k(part, model, KS), KS)
            row["snapshots"] = len(part)
            row["bookings"] = part["booking_idx"].nunique()
            row["average_precision"] = average_precision(values, labels)
            row["roc_auc"] = roc_auc(pd.Series(values), pd.Series(labels))
            probabilistic = model == CONSTANT or model in B1_VARIANTS
            row["brier"] = float(np.mean((values - labels) ** 2)) if probabilistic else np.nan
            row["mean_prediction"] = float(values.mean()) if probabilistic else np.nan
            rows.append(row)
    return rows


def select_alpha(train: pd.DataFrame, tune: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Choose alpha on the tune window by mean Precision@20 of both hotels, then by AP."""
    rows = []
    for alpha in ALPHAS:
        for row in evaluate(scored(train, tune, alpha), ["B1"]):
            rows.append({"alpha": alpha, **row})
    grid = pd.DataFrame(rows)
    means = grid.groupby("alpha")[[f"precision_at_{K}", "average_precision"]].mean()
    best = means.sort_values(
        [f"precision_at_{K}", "average_precision"], ascending=False, kind="mergesort"
    ).index[0]
    return float(best), grid


def label_exclusions(
    frame: pd.DataFrame, windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]]
) -> pd.DataFrame:
    """Snapshots dropped from each window because their outcome was not yet known."""
    rows = []
    for name, (start, end) in windows.items():
        in_time = frame[frame["snapshot_date"].between(start, end)]
        kept = window(frame, start, end)
        rows.append(
            {
                "window": name,
                "first_day": start.date(),
                "last_day": end.date(),
                "labels_known_before": (end + GAP).date(),
                "snapshots": len(in_time),
                "excluded_immature_snapshots": len(in_time) - len(kept),
                "excluded_immature_bookings": in_time["booking_idx"].nunique()
                - kept["booking_idx"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def protocol_tables(frame: pd.DataFrame, output: Path) -> tuple[float, pd.DataFrame]:
    """Baselines fitted once on the train split and scored on the tune and calibration windows."""
    train = history(frame, TUNE[0])
    tune = window(frame, *TUNE)
    alpha, grid = select_alpha(train, tune)
    save_table(grid, output, "b1_alpha_grid")
    models = [B0_RANDOM, B0_NEAREST, *B1_VARIANTS, CONSTANT]
    rows = []
    for name, (start, end) in (("tune", TUNE), ("calibration", CALIBRATION)):
        result = scored(train, window(frame, start, end), alpha)
        rows += [{"window": name, **row} for row in evaluate(result, models)]
    metrics = pd.DataFrame(rows)
    save_table(metrics, output, "baseline_metrics")
    save_table(
        label_exclusions(
            frame, {"train": SPLITS["train"], "tune": TUNE, "calibration": CALIBRATION}
        ),
        output,
        "label_maturity",
    )
    segments, _ = fit_rates(train, B1_KEYS, alpha)
    save_table(
        segments.sort_values(["hotel", "rate"], ascending=[True, False]), output, "b1_segments"
    )
    deduplicated = frame[~frame["duplicate"]]
    train_unique = history(deduplicated, TUNE[0])
    rows = []
    for name, (start, end) in (("tune", TUNE), ("calibration", CALIBRATION)):
        for policy, fit, evaluation in (
            ("keep_all", train, window(frame, start, end)),
            ("drop_exact_duplicates", train_unique, window(deduplicated, start, end)),
        ):
            result = scored(fit, evaluation, alpha)
            rows += [
                {"window": name, "policy": policy, **row}
                for row in evaluate(result, [B0_RANDOM, B0_NEAREST, "B1", "B1 + дни до заезда"])
            ]
    save_table(pd.DataFrame(rows), output, "duplicate_sensitivity")
    save_table(tie_break_leak(frame), output, "tie_break_leak")
    return alpha, metrics


def tie_break_leak(frame: pd.DataFrame) -> pd.DataFrame:
    """Why ties are never broken by row index: the CSV order is related to the label."""
    rows = []
    for name, (start, end) in (("tune", TUNE), ("calibration", CALIBRATION)):
        part = window(frame, start, end)
        nearest = -part["days_to_arrival"].astype(float)
        index = part["booking_idx"].astype(float)
        part = part.assign(
            **{
                "ties in expectation": nearest,
                "ties by row index, ascending": nearest * 1e6 - index,
                "ties by row index, descending": nearest * 1e6 + index,
            }
        )
        for hotel in HOTELS:
            daily_frame = part[part["hotel"].eq(hotel)]
            for policy in (
                "ties in expectation",
                "ties by row index, ascending",
                "ties by row index, descending",
            ):
                summary = summarize_top_k(daily_top_k(daily_frame, policy, (K,)), (K,))
                rows.append(
                    {
                        "window": name,
                        "hotel": hotel,
                        "model": B0_NEAREST,
                        "tie_policy": policy,
                        f"precision_at_{K}": summary[f"precision_at_{K}"],
                        f"lift_at_{K}": summary[f"lift_at_{K}"],
                    }
                )
    return pd.DataFrame(rows)


def backtest(frame: pd.DataFrame, alpha: float, output: Path) -> pd.DataFrame:
    """Rolling origin inside 2016: each window is fitted on all earlier mature snapshots."""
    rows = []
    for first in BACKTEST_STARTS:
        start = pd.Timestamp(first)
        end = start + pd.Timedelta(days=WINDOW_DAYS - 1)
        result = scored(history(frame, start), window(frame, start, end), alpha)
        for row in evaluate(result, [B0_NEAREST, *B1_VARIANTS]):
            rows.append({"window": f"{start:%Y-%m-%d}", "first_day": start.date(), **row})
    table = pd.DataFrame(rows)
    save_table(table, output, "backtest")
    return table


def plot_lift(backtest_table: pd.DataFrame, metrics: pd.DataFrame, output: Path) -> None:
    """Lift@20 of the rules in every window; points sit at the middle of each window."""
    protocol = metrics[metrics["model"].isin([B0_NEAREST, *B1_VARIANTS])].copy()
    bounds = {"tune": TUNE, "calibration": CALIBRATION}
    protocol["middle"] = protocol["window"].map(
        {name: start + (end - start) / 2 for name, (start, end) in bounds.items()}
    )
    rolling = backtest_table.assign(
        middle=pd.to_datetime(backtest_table["first_day"]) + pd.Timedelta(days=WINDOW_DAYS // 2)
    )
    combined = pd.concat([rolling, protocol], ignore_index=True)
    styles = {
        B0_NEAREST: ("#8a94a3", "s", "--"),
        "B1": (COLORS[0], "o", "-"),
        "B1 без депозита": ("#54a47b", "^", ":"),
        "B1 + дни до заезда": (COLORS[1], "D", "-"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, layout="constrained")
    for ax, hotel in zip(axes, HOTELS, strict=True):
        part = combined[combined["hotel"].eq(hotel)]
        for model, (color, marker, line) in styles.items():
            series = part[part["model"].eq(model)].sort_values("middle")
            ax.plot(
                series["middle"],
                series[f"lift_at_{K}"],
                color=color,
                marker=marker,
                linestyle=line,
                label=model,
            )
        ax.axhline(1, color="#4b5563", linewidth=1, label="Случайный порядок: Lift = 1")
        ax.axhline(3, color="#aa394b", linewidth=1, linestyle="--", label="Критерий MVP: Lift = 3")
        ax.axvspan(TUNE[0], CALIBRATION[1], color="#a1acba", alpha=0.18)
        ax.set(
            title=hotel, xlabel="Середина 45-дневного окна; серым — valid 2017", ylabel="Lift@20"
        )
        ax.tick_params(axis="x", rotation=30)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    save_figure(fig, output, "01_lift_by_window")


def plot_precision_curve(frame: pd.DataFrame, alpha: float, output: Path) -> None:
    """Precision@K over K on both validation windows: the budget trade-off."""
    train = history(frame, TUNE[0])
    evaluation = pd.concat([window(frame, *TUNE), window(frame, *CALIBRATION)])
    result = scored(train, evaluation, alpha)
    ks = (5, 10, 15, 20, 30, 40, 50, 75, 100)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True, layout="constrained")
    for ax, hotel in zip(axes, HOTELS, strict=True):
        part = result[result["hotel"].eq(hotel)]
        for model, color in (
            (B0_NEAREST, "#8a94a3"),
            ("B1", COLORS[0]),
            ("B1 + дни до заезда", COLORS[1]),
        ):
            summary = summarize_top_k(daily_top_k(part, model, ks), ks)
            ax.plot(
                ks, [summary[f"precision_at_{k}"] for k in ks], marker="o", color=color, label=model
            )
        base = summarize_top_k(daily_top_k(part, B0_RANDOM, ks), ks)["mean_daily_rate"]
        ax.axhline(base, color="#4b5563", linewidth=1, label="Случайный порядок")
        ax.axhline(3 * base, color="#aa394b", linewidth=1, linestyle="--", label="Lift = 3")
        ax.axvline(K, color="#4b5563", linewidth=0.8, linestyle=":")
        ax.set(title=hotel, xlabel="Бюджет K, броней в день", ylabel="Precision@K")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    axes[0].legend(fontsize=8)
    save_figure(fig, output, "02_precision_at_k")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    output = args.output_dir
    prepare_output(output)
    df = load_bookings(download(args.path))
    frame = candidates(df, CALIBRATION[1])
    alpha, metrics = protocol_tables(frame, output)
    backtest_table = backtest(frame, alpha, output)
    plot_lift(backtest_table, metrics, output)
    plot_precision_curve(frame, alpha, output)
    manifest = {
        "source_sha256": DATA_SHA256,
        "horizon_days": MAIN_HORIZON,
        "k": K,
        "selected_alpha": alpha,
        "min_support_bookings": MIN_SUPPORT,
        "last_snapshot_day": f"{frame['snapshot_date'].max():%Y-%m-%d}",
        "test_period_scored": False,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    last_day = manifest["last_snapshot_day"]
    print(f"Baseline artifacts ready: {output} (alpha = {alpha:g}, last snapshot day {last_day})")


if __name__ == "__main__":
    main()
