"""Feature-level EDA on daily snapshots: signal, correlations and the effect of duplicates.

Only the development period is used: bookings arriving before the test period and the
train/valid snapshot splits. Encodings are fitted on train and evaluated on valid inside
each hotel, because the product ranks candidates within a hotel and day.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from hotel_cancellation.audit import DERIVED_COLUMNS, SPLITS, TEST_START
from hotel_cancellation.metrics import roc_auc
from hotel_cancellation.report import COLORS, HOTELS, save_figure, save_table

GROUPS = {
    "Условно доступные": (
        "market_segment",
        "distribution_channel",
        "customer_type",
        "reserved_room_type",
        "meal",
        "agent",
        "company",
        "adults",
        "children",
        "babies",
        "nights",
        "stays_in_weekend_nights",
        "is_repeated_guest",
        "previous_bookings_not_canceled",
    ),
    "Календарь и возраст": (
        "lead_time",
        "days_to_arrival",
        "days_since_booking",
        "arrival_month",
        "arrival_weekday",
    ),
    "Спорные": ("adr", "deposit_type", "previous_cancellations"),
    "Исключены в MVP": (
        "booking_changes",
        "total_of_special_requests",
        "required_car_parking_spaces",
        "days_in_waiting_list",
        "assigned_room_type",
        "country",
    ),
}
GROUP_COLORS = {
    "Условно доступные": COLORS[0],
    "Календарь и возраст": "#54a47b",
    "Спорные": COLORS[1],
    "Исключены в MVP": "#a1acba",
}
SNAPSHOT_ONLY = ("days_to_arrival", "days_since_booking")
CATEGORICAL = (
    "market_segment",
    "distribution_channel",
    "customer_type",
    "reserved_room_type",
    "meal",
    "agent",
    "company",
    "deposit_type",
    "assigned_room_type",
    "country",
)
CORRELATION_FEATURES = (
    "lead_time",
    "days_to_arrival",
    "days_since_booking",
    "adr",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "total_of_special_requests",
    "required_car_parking_spaces",
    "days_in_waiting_list",
)
VIF_SETS: dict[str, tuple[str, ...]] = {
    "включая lead_time": (
        "lead_time",
        "days_to_arrival",
        "days_since_booking",
        "adr",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "adults",
        "children",
        "babies",
        "is_repeated_guest",
        "previous_cancellations",
        "previous_bookings_not_canceled",
    ),
}
VIF_SETS["без lead_time"] = tuple(
    name for name in VIF_SETS["включая lead_time"] if name != "lead_time"
)
ASSOCIATION_FEATURES = (
    "hotel",
    "market_segment",
    "distribution_channel",
    "customer_type",
    "deposit_type",
    "meal",
    "reserved_room_type",
    "agent",
    "company",
)
SMOOTHING = 30.0
MAX_EXACT_VALUES = 20
AGE_BINS = (0, 7, 30, 90, 180, 365, np.inf)
AGE_LABELS = ("1-7", "8-30", "31-90", "91-180", "181-365", ">365")
DUPLICATE_FIELDS = ("market_segment", "deposit_type", "customer_type", "distribution_channel")
SHIFT_FEATURES = (
    "lead_time",
    "adr",
    "nights",
    "adults",
    "children",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "is_repeated_guest",
    "total_of_special_requests",
    "booking_changes",
)


def booking_features(df: pd.DataFrame) -> pd.DataFrame:
    """Booking attributes with derived fields used across the feature analysis."""
    return df.assign(
        nights=df["stays_in_weekend_nights"] + df["stays_in_week_nights"],
        arrival_month=df["arrival_date"].dt.month,
        arrival_weekday=df["arrival_date"].dt.dayofweek,
        agent=df["agent"].astype("string").fillna("нет"),
        company=df["company"].astype("string").fillna("нет"),
        country=df["country"].fillna("нет"),
        status_date=pd.to_datetime(df["reservation_status_date"]),
        duplicate=df.drop(columns=DERIVED_COLUMNS).duplicated(),
    )


def populations(
    bookings: pd.DataFrame, snaps: pd.DataFrame
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Train and valid parts of both populations with labels mature at each cutoff."""
    valid_start, test_start = SPLITS["valid"][0], SPLITS["test"][0]
    raw = bookings.rename(columns={"is_canceled": "label"})
    raw_train = raw[raw["arrival_date"].lt(valid_start) & raw["status_date"].lt(valid_start)]
    raw_valid = raw[
        raw["arrival_date"].between(valid_start, TEST_START - pd.Timedelta(days=1))
        & raw["status_date"].lt(test_start)
    ]
    frames = {}
    for name, (start, end) in (("train", SPLITS["train"]), ("valid", SPLITS["valid"])):
        if end >= TEST_START:
            msg = "Feature analysis must not use test-period snapshots"
            raise ValueError(msg)
        part = snaps[snaps["snapshot_date"].between(start, end)]
        attributes = bookings.loc[part["booking_idx"]].drop(columns=["hotel"])
        attributes.index = part.index
        joined = pd.concat([part, attributes], axis=1)
        cutoff = valid_start if name == "train" else test_start
        frames[name] = joined[joined["status_date"].lt(cutoff)]
    return {
        "bookings": (raw_train, raw_valid),
        "snapshots": (frames["train"], frames["valid"]),
        "snapshots_unique": (
            frames["train"][~frames["train"]["duplicate"]],
            frames["valid"][~frames["valid"]["duplicate"]],
        ),
    }


def encode(train: pd.Series, label: pd.Series, values: pd.Series) -> pd.Series:
    """One-feature lookup fitted on train: smoothed positive rate of a category or decile."""
    prior = float(label.mean())
    if train.name in CATEGORICAL or train.nunique() <= MAX_EXACT_VALUES:
        train_keys, keys = train.astype(str), values.astype(str)
    else:
        edges = np.unique(np.quantile(train, np.linspace(0.1, 0.9, 9)))
        train_keys = pd.Series(np.searchsorted(edges, train, side="right"), index=train.index)
        keys = pd.Series(np.searchsorted(edges, values, side="right"), index=values.index)
    stats = label.groupby(train_keys).agg(["sum", "size"])
    rate = (stats["sum"] + SMOOTHING * prior) / (stats["size"] + SMOOTHING)
    return keys.map(rate).astype(float).fillna(prior)


def within_hotel_auc(train: pd.DataFrame, valid: pd.DataFrame, feature: str) -> float:
    """Mean over hotels of the valid ROC-AUC of a train-fitted one-feature lookup."""
    scores = []
    for hotel in HOTELS:
        fit = train[train["hotel"].eq(hotel)]
        test = valid[valid["hotel"].eq(hotel)]
        encoded = encode(fit[feature], fit["label"], test[feature])
        scores.append(roc_auc(encoded.reset_index(drop=True), test["label"].reset_index(drop=True)))
    return float(np.mean(scores))


def raw_direction(train: pd.DataFrame, feature: str) -> float:
    """Mean within-hotel ROC-AUC of the raw numeric value: above 0.5 means higher is riskier."""
    scores = []
    for hotel in HOTELS:
        part = train[train["hotel"].eq(hotel)]
        values = part[feature].astype(float).fillna(part[feature].median())
        scores.append(roc_auc(values.reset_index(drop=True), part["label"].reset_index(drop=True)))
    return float(np.mean(scores))


def signal_table(parts: dict[str, tuple[pd.DataFrame, pd.DataFrame]], output: Path) -> pd.DataFrame:
    rows = []
    for group, features in GROUPS.items():
        for feature in features:
            row: dict[str, str | float] = {"feature": feature, "group": group}
            for population, (train, valid) in parts.items():
                if population == "bookings" and feature in SNAPSHOT_ONLY:
                    row[f"{population}_auc"] = np.nan
                    continue
                row[f"{population}_auc"] = within_hotel_auc(train, valid, feature)
            if feature not in CATEGORICAL:
                for population in ("bookings", "snapshots"):
                    train = parts[population][0]
                    row[f"{population}_raw_auc"] = (
                        np.nan
                        if population == "bookings" and feature in SNAPSHOT_ONLY
                        else raw_direction(train, feature)
                    )
            rows.append(row)
    table = pd.DataFrame(rows).sort_values("snapshots_auc", ascending=False)
    save_table(table, output, "feature_signal")
    return table


def vif(frame: pd.DataFrame, features: Sequence[str]) -> list[float]:
    """Variance inflation factors via least squares; an exact linear identity gives inf."""
    data = frame[list(features)].astype(float)
    data = (data - data.mean()) / data.std()
    matrix = data.fillna(0).to_numpy()
    result = []
    for i in range(matrix.shape[1]):
        target = matrix[:, i]
        others = np.delete(matrix, i, axis=1)
        coef, *_ = np.linalg.lstsq(others, target, rcond=None)
        residual = float(np.sum((target - others @ coef) ** 2))
        r_squared = 1 - residual / float(np.sum(target**2))
        result.append(np.inf if r_squared > 1 - 1e-9 else 1 / (1 - r_squared))
    return result


def cramers_v(left: pd.Series, right: pd.Series) -> float:
    """Bias-corrected Cramér's V (Bergsma, 2013) for two categorical series."""
    table = pd.crosstab(left, right).to_numpy(dtype=float)
    n = table.sum()
    expected = table.sum(axis=1, keepdims=True) @ table.sum(axis=0, keepdims=True) / n
    chi2 = float(np.sum((table - expected) ** 2 / expected))
    rows, cols = table.shape
    phi2 = max(0.0, chi2 / n - (rows - 1) * (cols - 1) / (n - 1))
    rows_corr = rows - (rows - 1) ** 2 / (n - 1)
    cols_corr = cols - (cols - 1) ** 2 / (n - 1)
    denominator = min(rows_corr - 1, cols_corr - 1)
    return float(np.sqrt(phi2 / denominator)) if denominator > 0 else 0.0


def dependence_tables(train: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = train[[*CORRELATION_FEATURES, "label"]].astype(float)
    spearman = numeric.corr(method="spearman")
    save_table(spearman.rename_axis("feature").reset_index(), output, "feature_spearman")
    rows = []
    for name, features in VIF_SETS.items():
        for feature, value in zip(features, vif(train, features), strict=True):
            rows.append({"feature_set": name, "feature": feature, "vif": value})
    save_table(pd.DataFrame(rows), output, "feature_vif")
    names = list(ASSOCIATION_FEATURES)
    association = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            value = cramers_v(train[left].astype(str), train[right].astype(str))
            association.loc[left, right] = association.loc[right, left] = value
    save_table(association.rename_axis("feature").reset_index(), output, "feature_cramers_v")
    return spearman, association


def snapshot_rates(train: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Train-snapshot loss rate by value of the features reviewers asked about."""
    binned = train.assign(
        days_since_booking=pd.cut(train["days_since_booking"], AGE_BINS, labels=AGE_LABELS),
        previous_cancellations=train["previous_cancellations"]
        .clip(upper=2)
        .map({0: "0", 1: "1", 2: "2+"}),
        adr=train.groupby("hotel")["adr"].transform(
            lambda values: pd.qcut(values, 10, labels=[f"D{i}" for i in range(1, 11)])
        ),
    )
    rows = []
    for feature in (
        "days_to_arrival",
        "days_since_booking",
        "adr",
        "deposit_type",
        "previous_cancellations",
        "customer_type",
    ):
        grouped = (
            binned.groupby(["hotel", feature], observed=True)["label"]
            .agg(snapshots="size", positives="sum", positive_rate="mean")
            .reset_index()
            .rename(columns={feature: "value"})
        )
        grouped["value"] = grouped["value"].astype(str)
        grouped["share_of_hotel_snapshots"] = grouped["snapshots"] / grouped.groupby("hotel")[
            "snapshots"
        ].transform("sum")
        grouped.insert(0, "feature", feature)
        rows.append(grouped)
    table = pd.concat(rows, ignore_index=True)
    save_table(table, output, "snapshot_feature_rates")
    return table


def duplicate_tables(
    bookings: pd.DataFrame, train: pd.DataFrame, output: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """How dropping exact repeats changes composition, segment rates and numeric features."""
    development = bookings[bookings["arrival_date"].lt(TEST_START)]
    unique = development[~development["duplicate"]]
    train_unique = train[~train["duplicate"]]
    rows = []
    for field in ("hotel", *DUPLICATE_FIELDS):
        for value, part in development.groupby(field):
            kept = unique[unique[field].eq(value)]
            snaps = train[train[field].eq(value)]
            snaps_kept = train_unique[train_unique[field].eq(value)]
            rows.append(
                {
                    "field": field,
                    "value": value,
                    "bookings": len(part),
                    "repeat_share": part["duplicate"].mean(),
                    "share_all": len(part) / len(development),
                    "share_unique": len(kept) / len(unique),
                    "loss_rate_all": part["is_canceled"].mean(),
                    "loss_rate_unique": kept["is_canceled"].mean() if len(kept) else np.nan,
                    "train_snapshots_all": len(snaps),
                    "train_snapshots_unique": len(snaps_kept),
                    "snapshot_rate_all": snaps["label"].mean() if len(snaps) else np.nan,
                    "snapshot_rate_unique": snaps_kept["label"].mean()
                    if len(snaps_kept)
                    else np.nan,
                }
            )
    composition = pd.DataFrame(rows)
    save_table(composition, output, "duplicate_composition")
    repeats = development[development["duplicate"]]
    rows = []
    for feature in SHIFT_FEATURES:
        pooled = np.sqrt((repeats[feature].var() + unique[feature].var()) / 2)
        rows.append(
            {
                "feature": feature,
                "mean_all": development[feature].mean(),
                "mean_unique": unique[feature].mean(),
                "median_all": development[feature].median(),
                "median_unique": unique[feature].median(),
                "mean_removed_repeats": repeats[feature].mean(),
                "smd_repeats_vs_unique": (repeats[feature].mean() - unique[feature].mean())
                / pooled,
            }
        )
    shift = pd.DataFrame(rows)
    save_table(shift, output, "duplicate_numeric_shift")
    return composition, shift


def row_order_table(bookings: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Loss rate by the position of a row inside its hotel and arrival day in the CSV.

    The row index is only a technical key: if the order carries the label, it must never
    break ties, sort candidates or enter a model.
    """
    development = bookings[bookings["arrival_date"].lt(TEST_START)]
    groups = development.groupby(["hotel", "arrival_date"])
    size = groups["hotel"].transform("size")
    position = groups.cumcount() / (size - 1).clip(lower=1)
    bins = pd.cut(position, [0, 0.2, 0.4, 0.6, 0.8, 1], include_lowest=True)
    table = (
        development.groupby(bins, observed=True)["is_canceled"]
        .agg(bookings="size", loss_rate="mean")
        .rename_axis("relative_position_in_day")
        .reset_index()
    )
    table["relative_position_in_day"] = table["relative_position_in_day"].astype(str)
    auc = roc_auc(
        position.reset_index(drop=True), development["is_canceled"].reset_index(drop=True)
    )
    table["position_roc_auc_oriented"] = max(auc, 1 - auc)
    save_table(table, output, "row_order_label")
    return table


def plot_signal(table: pd.DataFrame, output: Path) -> None:
    ordered = table.sort_values("snapshots_auc")
    positions = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(10, 8.5), layout="constrained")
    ax.barh(
        positions,
        ordered["snapshots_auc"] - 0.5,
        left=0.5,
        color=[GROUP_COLORS[group] for group in ordered["group"]],
        height=0.65,
    )
    ax.scatter(ordered["bookings_auc"], positions, color="#1f2937", s=18, zorder=3)
    ax.scatter(
        ordered["snapshots_unique_auc"],
        positions,
        facecolors="none",
        edgecolors="#aa394b",
        marker="D",
        s=26,
        zorder=3,
    )
    ax.axvline(0.5, color="#4b5563", linewidth=1)
    ax.set(
        yticks=positions,
        yticklabels=ordered["feature"],
        xlabel="ROC-AUC на valid внутри отеля; кодировка признака обучена на train",
        title="Сигнал одного признака: снимки H=14 (столбцы) против исходных броней (точки)",
        xlim=(0.4, 0.9),
    )
    handles: list[Artist] = [Patch(color=color) for color in GROUP_COLORS.values()]
    handles += [
        Line2D([], [], color="#1f2937", marker="o", linestyle=""),
        Line2D([], [], color="#aa394b", marker="D", markerfacecolor="none", linestyle=""),
    ]
    labels = [*GROUP_COLORS, "Исходные брони", "Снимки без полных повторов"]
    ax.legend(handles, labels, loc="lower right", fontsize=8)
    save_figure(fig, output, "09_feature_signal")


def plot_dependence(spearman: pd.DataFrame, association: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6), layout="constrained")
    for ax, matrix, title, cmap, limits in (
        (axes[0], spearman, "Спирмен: числовые признаки и метка", "RdBu_r", (-1, 1)),
        (axes[1], association, "Cramér's V: категориальные признаки", "Blues", (0, 1)),
    ):
        image = ax.imshow(matrix.to_numpy(), cmap=cmap, vmin=limits[0], vmax=limits[1])
        ax.set_xticks(range(len(matrix)), matrix.columns, rotation=90, fontsize=8)
        ax.set_yticks(range(len(matrix)), matrix.index, fontsize=8)
        ax.grid(False)
        ax.set_title(title)
        for (i, j), value in np.ndenumerate(matrix.to_numpy()):
            if i != j and abs(value) >= 0.3:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5)
        fig.colorbar(image, ax=ax, shrink=0.75)
    save_figure(fig, output, "10_feature_dependence")


def plot_snapshot_rates(table: pd.DataFrame, output: Path) -> None:
    titles = {
        "days_to_arrival": "Дней до заезда",
        "days_since_booking": "Возраст брони, дней",
        "adr": "ADR: децили внутри отеля",
        "deposit_type": "deposit_type (доля снимков)",
        "previous_cancellations": "previous_cancellations (доля снимков)",
        "customer_type": "customer_type (доля снимков)",
    }
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), layout="constrained")
    for ax, (feature, title) in zip(axes.flat, titles.items(), strict=True):
        part = table[table["feature"].eq(feature)]
        order = list(dict.fromkeys(part["value"]))
        if feature == "days_to_arrival":
            order = sorted(order, key=int)
        positions = np.arange(len(order))
        for offset, hotel, color in zip((-0.2, 0.2), HOTELS, COLORS, strict=True):
            values = part[part["hotel"].eq(hotel)].set_index("value").reindex(order)
            ax.bar(positions + offset, values["positive_rate"], width=0.4, color=color, label=hotel)
            if feature in ("deposit_type", "previous_cancellations", "customer_type"):
                for x, (rate, share) in zip(
                    positions + offset,
                    values[["positive_rate", "share_of_hotel_snapshots"]].to_numpy(),
                    strict=True,
                ):
                    if not np.isnan(rate):
                        ax.text(x, rate, f"{share:.1%}", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(positions, order, rotation=45 if len(order) > 6 else 0, fontsize=8)
        ax.set(title=title, ylabel="Доля будущих потерь")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0, 0].legend(fontsize=8)
    save_figure(fig, output, "11_snapshot_features")


def plot_duplicates(composition: pd.DataFrame, output: Path) -> None:
    selected = composition[
        composition["field"].isin(["market_segment", "deposit_type"])
        & composition["share_all"].ge(0.005)
    ].copy()
    selected["name"] = selected["field"].str.replace("_", " ") + ": " + selected["value"]
    selected = selected.iloc[::-1]
    positions = np.arange(len(selected))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, layout="constrained")
    axes[0].barh(positions, selected["repeat_share"], color="#aa394b")
    axes[0].set(
        yticks=positions,
        yticklabels=selected["name"],
        title="Доля полных повторов в категории",
        xlim=(0, 1),
    )
    for ax, (left, right), title in (
        (axes[1], ("share_all", "share_unique"), "Доля категории среди броней"),
        (axes[2], ("snapshot_rate_all", "snapshot_rate_unique"), "Доля потерь в train-снимках"),
    ):
        ax.barh(
            positions + 0.2, selected[left], height=0.4, color=COLORS[0], label="Исходные строки"
        )
        ax.barh(positions - 0.2, selected[right], height=0.4, color=COLORS[1], label="Без повторов")
        ax.set_title(title)
    for ax in axes:
        ax.xaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(fontsize=8, loc="lower right")
    save_figure(fig, output, "12_duplicates")


def feature_report(df: pd.DataFrame, snaps: pd.DataFrame, output: Path) -> None:
    """Write feature-level tables and figures 09-12 into the EDA report folder."""
    bookings = booking_features(df)
    parts = populations(bookings, snaps)
    train = parts["snapshots"][0]
    signal = signal_table(parts, output)
    spearman, association = dependence_tables(train, output)
    rates = snapshot_rates(train, output)
    composition, _ = duplicate_tables(bookings, train, output)
    row_order_table(bookings, output)
    plot_signal(signal, output)
    plot_dependence(spearman, association, output)
    plot_snapshot_rates(rates, output)
    plot_duplicates(composition, output)
