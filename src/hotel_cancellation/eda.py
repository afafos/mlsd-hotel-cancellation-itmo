import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from hotel_cancellation.audit import DERIVED_COLUMNS, MAIN_HORIZON, SPLITS, roc_auc
from hotel_cancellation.data import DATA_SHA256, DATA_URL, RAW_PATH, download, load_bookings
from hotel_cancellation.snapshots import build_snapshots

OUTPUT_DIR = Path("reports/eda")
HOTELS = ("City Hotel", "Resort Hotel")
COLORS = ("#2563a6", "#d17b21")
LEAD_BINS = [-1, 0, 7, 14, 30, 60, 90, 180, 365, 1000]
LEAD_LABELS = ["0", "1-7", "8-14", "15-30", "31-60", "61-90", "91-180", "181-365", ">365"]


def save_table(table: pd.DataFrame, output: Path, name: str) -> None:
    """Export deterministic aggregates, never booking-level records."""
    table.to_csv(output / "tables" / f"{name}.csv", index=False, float_format="%.8f")


def save_figure(fig: Figure, output: Path, name: str) -> None:
    fig.savefig(output / "figures" / f"{name}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def profiles(df: pd.DataFrame, output: Path) -> pd.DataFrame:
    raw = df.drop(columns=DERIVED_COLUMNS)
    fields = pd.DataFrame(
        {
            "field": raw.columns,
            "dtype": raw.dtypes.astype(str).to_numpy(),
            "non_null": raw.count().to_numpy(),
            "missing": raw.isna().sum().to_numpy(),
            "missing_pct": raw.isna().mean().to_numpy() * 100,
            "distinct_non_null": raw.nunique().to_numpy(),
        }
    )
    save_table(fields, output, "field_profile")
    numeric = raw.select_dtypes(include="number").describe(
        percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]
    )
    save_table(numeric.T.rename_axis("field").reset_index(), output, "numeric_profile")
    return fields


def quality_checks(df: pd.DataFrame, output: Path) -> None:
    nights = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    guests = df["adults"] + df["children"].fillna(0) + df["babies"]
    lost = df["is_canceled"].eq(1)
    status_date = pd.to_datetime(df["reservation_status_date"])
    checks = {
        "target_outside_0_1": ~df["is_canceled"].isin([0, 1]),
        "target_status_mismatch": lost.ne(df["reservation_status"].isin(["Canceled", "No-Show"])),
        "unknown_status": ~df["reservation_status"].isin(["Canceled", "No-Show", "Check-Out"]),
        "negative_lead_time": df["lead_time"].lt(0),
        "lost_before_booking": lost & df["cancel_date"].lt(df["booking_date"]),
        "canceled_after_arrival": df["reservation_status"].eq("Canceled")
        & df["cancel_date"].gt(df["arrival_date"]),
        "no_show_date_not_arrival": df["reservation_status"].eq("No-Show")
        & df["cancel_date"].ne(df["arrival_date"]),
        "checkout_before_arrival": df["reservation_status"].eq("Check-Out")
        & status_date.lt(df["arrival_date"]),
        "adr_negative": df["adr"].lt(0),
        "adr_zero": df["adr"].eq(0),
        "adr_above_1000": df["adr"].gt(1000),
        "zero_nights": nights.eq(0),
        "negative_nights": nights.lt(0),
        "zero_guests_children_missing_as_zero": guests.eq(0),
        "negative_guest_count": df[["adults", "children", "babies"]].lt(0).any(axis=1),
        "children_missing": df["children"].isna(),
        "fully_duplicated_after_first": df.drop(columns=DERIVED_COLUMNS).duplicated(),
    }
    for field in ("meal", "market_segment", "distribution_channel"):
        checks[f"undefined_{field}"] = df[field].eq("Undefined")
    result = pd.DataFrame(
        [
            {"check": name, "bookings": int(mask.sum()), "pct": mask.mean() * 100}
            for name, mask in checks.items()
        ]
    )
    save_table(result, output, "quality_checks")
    statuses = pd.crosstab(df["hotel"], df["reservation_status"])
    statuses["bookings"] = statuses.sum(axis=1)
    statuses["lost_pct"] = (statuses["Canceled"] + statuses["No-Show"]) / statuses["bookings"] * 100
    save_table(statuses.reset_index(), output, "outcomes")


def split_tables(df: pd.DataFrame, snaps: pd.DataFrame, output: Path) -> None:
    rows = []
    readiness = []
    status_dates = pd.to_datetime(df["reservation_status_date"])
    for name, (start, end) in SPLITS.items():
        part = snaps[snaps["snapshot_date"].between(start, end)]
        for hotel in ("All", *HOTELS):
            group = part if hotel == "All" else part[part["hotel"].eq(hotel)]
            daily = group.groupby(["hotel", "snapshot_date"])["label"].agg(["size", "sum"])
            rows.append(
                {
                    "split": name,
                    "hotel": hotel,
                    "first_day": start.date(),
                    "last_day": end.date(),
                    "snapshot_rows": len(group),
                    "source_rows": group["booking_idx"].nunique(),
                    "positive_rows": int(group["label"].sum()),
                    "positive_pct": group["label"].mean() * 100,
                    "mean_candidates_per_hotel_day": daily["size"].mean(),
                    "mean_positive_per_hotel_day": daily["sum"].mean(),
                    "hotel_days_without_positives": int(daily["sum"].eq(0).sum()),
                    "hotel_days_with_fewer_than_20": int(daily["size"].lt(20).sum()),
                }
            )
        if name in ("train", "valid"):
            cutoff = SPLITS["valid" if name == "train" else "test"][0]
            unavailable = status_dates.loc[part["booking_idx"]].ge(cutoff).to_numpy()
            immature = part.loc[unavailable]
            readiness.append(
                {
                    "split": name,
                    "fit_cutoff_exclusive": cutoff.date(),
                    "immature_snapshot_rows": len(immature),
                    "immature_source_rows": immature["booking_idx"].nunique(),
                    "mature_snapshot_rows": len(part) - len(immature),
                    "mature_source_rows": part.loc[~unavailable, "booking_idx"].nunique(),
                }
            )
    save_table(pd.DataFrame(rows), output, "split_summary")
    save_table(pd.DataFrame(readiness), output, "label_readiness")
    overlaps = []
    names = list(SPLITS)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            a = set(snaps.loc[snaps["snapshot_date"].between(*SPLITS[left]), "booking_idx"])
            b = set(snaps.loc[snaps["snapshot_date"].between(*SPLITS[right]), "booking_idx"])
            overlaps.append({"left": left, "right": right, "shared_source_rows": len(a & b)})
    save_table(pd.DataFrame(overlaps), output, "split_overlap")


def sensitivity(df: pd.DataFrame, snaps: pd.DataFrame, output: Path) -> None:
    unique = df.loc[~df.drop(columns=DERIVED_COLUMNS).duplicated()]
    unique_snaps = snaps[snaps["booking_idx"].isin(unique.index)]
    rows = []
    for name, bookings, part in (
        ("keep_all", df, snaps),
        ("drop_exact_duplicates", unique, unique_snaps),
    ):
        for hotel in HOTELS:
            raw = bookings[bookings["hotel"].eq(hotel)]
            daily = part[part["hotel"].eq(hotel)]
            test = daily[daily["snapshot_date"].between(*SPLITS["test"])]
            rows.append(
                {
                    "policy": name,
                    "hotel": hotel,
                    "bookings": len(raw),
                    "raw_positive_pct": raw["is_canceled"].mean() * 100,
                    "snapshot_rows": len(daily),
                    "snapshot_positive_pct": daily["label"].mean() * 100,
                    "test_rows": len(test),
                    "test_positive_pct": test["label"].mean() * 100,
                }
            )
    save_table(pd.DataFrame(rows), output, "duplicate_sensitivity")
    event_day = df.loc[snaps["booking_idx"], "cancel_date"].reset_index(drop=True)
    same_day = snaps["snapshot_date"].eq(event_day)
    rows = []
    for name, part in (
        ("include_cancellation_day", snaps),
        ("exclude_cancellation_day", snaps[~same_day]),
    ):
        for hotel in HOTELS:
            test = part[part["hotel"].eq(hotel) & part["snapshot_date"].between(*SPLITS["test"])]
            rows.append(
                {
                    "convention": name,
                    "hotel": hotel,
                    "test_rows": len(test),
                    "positive_pct": test["label"].mean() * 100,
                }
            )
    save_table(pd.DataFrame(rows), output, "cancellation_day_sensitivity")
    rows = []
    for horizon in (7, 14, 30):
        part = build_snapshots(df, horizon, start=SPLITS["test"][0], end=pd.Timestamp("2017-08-01"))
        for hotel in HOTELS:
            group = part[part["hotel"].eq(hotel)]
            daily = group.groupby("snapshot_date")["label"].agg(["size", "sum"])
            rows.append(
                {
                    "horizon": horizon,
                    "hotel": hotel,
                    "first_day": "2017-05-01",
                    "last_day": "2017-08-01",
                    "snapshot_rows": len(group),
                    "positive_pct": group["label"].mean() * 100,
                    "mean_candidates": daily["size"].mean(),
                    "mean_positives": daily["sum"].mean(),
                }
            )
    save_table(pd.DataFrame(rows), output, "horizon_sensitivity")


def feature_tables(
    df: pd.DataFrame, snaps: pd.DataFrame, output: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = df.assign(month=df["arrival_date"].dt.to_period("M").astype(str))
    monthly = (
        raw.groupby(["hotel", "month"])["is_canceled"]
        .agg(bookings="size", lost="sum", lost_rate="mean")
        .reset_index()
    )
    save_table(monthly, output, "monthly_bookings")
    daily = snaps.assign(month=snaps["snapshot_date"].dt.to_period("M").astype(str))
    monthly_snaps = (
        daily.groupby(["hotel", "month"])["label"]
        .agg(snapshot_rows="size", positive_rows="sum", positive_rate="mean")
        .reset_index()
    )
    save_table(monthly_snaps, output, "monthly_snapshots")
    categories = []
    for field in ("market_segment", "distribution_channel", "deposit_type", "customer_type"):
        for population, values, target in (
            ("bookings", df, "is_canceled"),
            ("snapshots", snaps, "label"),
        ):
            part = values[["hotel", target]].copy()
            part["category"] = (
                df[field].to_numpy()
                if population == "bookings"
                else df.loc[snaps["booking_idx"], field].to_numpy()
            )
            grouped = (
                part.groupby(["hotel", "category"])[target]
                .agg(rows="size", positives="sum", positive_rate="mean")
                .reset_index()
            )
            grouped.insert(0, "population", population)
            grouped.insert(0, "feature", field)
            categories.append(grouped)
    cat_table = pd.concat(categories, ignore_index=True)
    save_table(cat_table, output, "categorical_rates")
    leads = []
    for population, values, target in (
        ("bookings", df, "is_canceled"),
        ("snapshots", snaps, "label"),
    ):
        part = values[["hotel", target]].copy()
        lead = (
            df["lead_time"].to_numpy()
            if population == "bookings"
            else df.loc[snaps["booking_idx"], "lead_time"].to_numpy()
        )
        part["lead_time_bin"] = pd.cut(lead, LEAD_BINS, labels=LEAD_LABELS)
        grouped = (
            part.groupby(["hotel", "lead_time_bin"], observed=False)[target]
            .agg(rows="size", positives="sum", positive_rate="mean")
            .reset_index()
        )
        grouped.insert(0, "population", population)
        leads.append(grouped)
    lead_table = pd.concat(leads, ignore_index=True)
    save_table(lead_table, output, "lead_time_rates")
    return monthly, cat_table, lead_table


def plot_overview(
    df: pd.DataFrame, fields: pd.DataFrame, monthly: pd.DataFrame, output: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    missing = fields[fields["missing"].gt(0)].sort_values("missing_pct")
    axes[0].barh(missing["field"], missing["missing_pct"], color=COLORS[0])
    axes[0].set(xlabel="Пропуски, % строк", title="Пропуски: 4 из 32 полей", xlim=(0, 100))
    duplicate = df.drop(columns=DERIVED_COLUMNS).duplicated()
    axes[1].bar(
        ["Первая/уникальная строка", "Повтор после первой"],
        [int((~duplicate).sum()), int(duplicate.sum())],
        color=list(COLORS),
    )
    axes[1].set(ylabel="Строк", title="Точное совпадение всех 32 полей")
    axes[1].tick_params(axis="x", labelsize=8)
    save_figure(fig, output, "01_quality")

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    counts = pd.crosstab(df["hotel"], df["reservation_status"]).reindex(HOTELS)
    bottom = np.zeros(len(HOTELS))
    for status, color in zip(
        ("Check-Out", "Canceled", "No-Show"), ("#54a47b", "#d17b21", "#aa394b"), strict=True
    ):
        values = (counts[status] / counts.sum(axis=1)).to_numpy()
        ax.bar(HOTELS, values, bottom=bottom, label=status, color=color)
        bottom += values
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(ylabel="Доля исходов", title="Исходы всех броней: отели отличаются")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
    save_figure(fig, output, "02_outcomes")

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
    for hotel, color in zip(HOTELS, COLORS, strict=True):
        part = monthly[monthly["hotel"].eq(hotel)]
        dates = pd.to_datetime(part["month"])
        axes[0].plot(dates, part["bookings"], label=hotel, color=color, marker=".")
        axes[1].plot(dates, part["lost_rate"], label=hotel, color=color, marker=".")
    axes[0].set(ylabel="Броней", title="Объём и доля потерь по месяцу заезда")
    axes[0].legend()
    axes[1].set(ylabel="Отмена или no-show", xlabel="Месяц заезда (2015 и 2017 — неполные годы)")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    save_figure(fig, output, "03_monthly")


def plot_numeric(df: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    numeric = {
        "lead_time": df["lead_time"],
        "Число ночей": df["stays_in_weekend_nights"] + df["stays_in_week_nights"],
        "ADR": df["adr"],
    }
    bounds = []
    for ax, (name, values) in zip(axes, numeric.items(), strict=True):
        upper = float(values.quantile(0.99))
        shown = values[values.between(0, upper)]
        ax.hist(shown, bins=35, color=COLORS[0])
        ax.set(title=f"{name}: 0 … q99={upper:g}", ylabel="Строк", xlabel="Значение")
        bounds.append(
            {
                "feature": name,
                "plot_upper_q99": upper,
                "not_shown_rows": len(values) - len(shown),
                "actual_min": values.min(),
                "actual_max": values.max(),
            }
        )
    save_table(pd.DataFrame(bounds), output, "histogram_bounds")
    save_figure(fig, output, "04_numeric")

    canceled = df[df["reservation_status"].eq("Canceled")]
    days = (canceled["arrival_date"] - canceled["cancel_date"]).dt.days
    x = np.arange(int(days.max()) + 1)
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    ax.plot(x, [days.le(day).mean() for day in x], color=COLORS[0])
    rows = []
    for horizon in (7, 14, 30):
        rate = float(days.le(horizon).mean())
        ax.scatter([horizon], [rate], color=COLORS[1])
        ax.annotate(
            f"≤{horizon} дней: {rate:.1%}", (horizon, rate), xytext=(horizon + 20, rate - 0.025)
        )
        rows.append(
            {
                "days_before_arrival": horizon,
                "canceled_bookings": int(days.le(horizon).sum()),
                "pct_of_canceled": rate * 100,
            }
        )
    ax.set(
        xlabel="Дней от отмены до заезда",
        ylabel="Накопленная доля отмен",
        title="Большинство отмен происходит задолго до заезда; no-show отдельно",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    save_table(pd.DataFrame(rows), output, "cancellation_timing")
    save_figure(fig, output, "05_cancellation_timing")


def plot_features(categories: pd.DataFrame, lead: pd.DataFrame, output: Path) -> None:
    populations = (
        ("bookings", "Исходные брони", COLORS[0]),
        ("snapshots", "Дневные снимки H=14", COLORS[1]),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    for ax, feature in zip(axes, ("market_segment", "deposit_type"), strict=True):
        selected = categories[categories["feature"].eq(feature)]
        sums = selected.groupby(["population", "category"])[["rows", "positives"]].sum()
        sums["rate"] = sums["positives"] / sums["rows"]
        rates = sums.reset_index().pivot_table(
            index="category", columns="population", values="rate"
        )
        positions = np.arange(len(rates))
        for offset, (population, label, color) in zip((-0.18, 0.18), populations, strict=True):
            ax.barh(positions + offset, rates[population], height=0.34, label=label, color=color)
        ax.set(
            yticks=positions, yticklabels=rates.index, xlabel="Доля отмен / no-show", title=feature
        )
        ax.xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=8)
    save_figure(fig, output, "06_segments")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, layout="constrained")
    for ax, hotel in zip(axes, HOTELS, strict=True):
        for population, label, color in populations:
            part = lead[lead["hotel"].eq(hotel) & lead["population"].eq(population)]
            ax.plot(
                part["lead_time_bin"].astype(str),
                part["positive_rate"],
                marker="o",
                label=label,
                color=color,
            )
        ax.set(title=hotel, xlabel="lead_time, дней", ylabel="Доля отмен / no-show")
        ax.tick_params(axis="x", rotation=45)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(fontsize=8)
    save_figure(fig, output, "08_lead_time")


def plot_snapshots(snaps: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, layout="constrained")
    daily = (
        snaps.groupby(["hotel", "snapshot_date"])["label"]
        .agg(candidates="size", positive="sum")
        .reset_index()
    )
    daily["positive_rate"] = daily["positive"] / daily["candidates"]
    save_table(daily, output, "daily_lists")
    for hotel, color in zip(HOTELS, COLORS, strict=True):
        part = daily[daily["hotel"].eq(hotel)]
        axes[0].plot(
            part["snapshot_date"], part["candidates"], color=color, label=hotel, linewidth=1
        )
        axes[1].plot(
            part["snapshot_date"],
            part["positive_rate"].rolling(7, min_periods=1).mean(),
            color=color,
            label=hotel,
        )
    for name, (start, end) in SPLITS.items():
        for ax in axes:
            ax.axvspan(start, end, color="#a1acba", alpha=0.10 if name != "valid" else 0.22)
        axes[0].text(
            start + (end - start) / 2,
            0.96,
            name,
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="top",
        )
    axes[0].set(
        ylabel="Кандидатов в день",
        title="Рабочая популяция: активные брони, заезд через 1-14 дней",
    )
    axes[0].legend(loc="upper left")
    axes[1].set(
        ylabel="Доля будущих потерь",
        xlabel="Дата снимка; доля сглажена средним за последние 7 дней",
    )
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    save_figure(fig, output, "07_daily_lists")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    output = args.output_dir
    (output / "figures").mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )
    df = load_bookings(download(args.path))
    snaps = build_snapshots(df, MAIN_HORIZON)
    fields = profiles(df, output)
    quality_checks(df, output)
    split_tables(df, snaps, output)
    sensitivity(df, snaps, output)
    monthly, categories, lead = feature_tables(df, snaps, output)
    plot_overview(df, fields, monthly, output)
    plot_numeric(df, output)
    plot_features(categories, lead, output)
    plot_snapshots(snaps, output)
    auc_raw = roc_auc(df["lead_time"], df["is_canceled"])
    auc_snap = roc_auc(
        pd.Series(df.loc[snaps["booking_idx"], "lead_time"].to_numpy()), snaps["label"]
    )
    save_table(
        pd.DataFrame(
            {"population": ["bookings", "snapshots"], "lead_time_roc_auc": [auc_raw, auc_snap]}
        ),
        output,
        "single_feature_auc",
    )
    manifest = {
        "source_url": DATA_URL,
        "source_sha256": DATA_SHA256,
        "raw_rows": len(df),
        "raw_columns": len(df.columns) - len(DERIVED_COLUMNS),
        "snapshot_rows": len(snaps),
        "snapshot_source_rows": snaps["booking_idx"].nunique(),
        "horizon_days": MAIN_HORIZON,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"EDA artifacts ready: {output} (8 figures and aggregate CSV tables)")


if __name__ == "__main__":
    main()
