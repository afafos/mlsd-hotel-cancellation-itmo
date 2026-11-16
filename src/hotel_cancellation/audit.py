import argparse
from pathlib import Path

import pandas as pd

from hotel_cancellation.data import RAW_PATH, download, load_bookings
from hotel_cancellation.snapshots import build_snapshots

HORIZONS = (7, 14, 30)
MAIN_HORIZON = 14
SPLITS = {
    "train": (pd.Timestamp("2015-07-01"), pd.Timestamp("2016-12-17")),
    "valid": (pd.Timestamp("2017-01-01"), pd.Timestamp("2017-04-16")),
    "test": (pd.Timestamp("2017-05-01"), pd.Timestamp("2017-08-17")),
}
TEST_START = SPLITS["test"][0]
DERIVED_COLUMNS = ["arrival_date", "booking_date", "cancel_date"]


def section(title: str) -> None:
    print(f"\n## {title}\n")


def roc_auc(score: pd.Series, label: pd.Series) -> float:
    """ROC-AUC of a single score via the Mann-Whitney U statistic (ties get average ranks)."""
    ranks = score.rank()
    n_pos = int(label.sum())
    n_neg = len(label) - n_pos
    return float((ranks[label == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def overview(df: pd.DataFrame) -> None:
    section("Overview")
    print(f"bookings: {len(df)}")
    print(f"arrivals: {df['arrival_date'].min():%Y-%m-%d} .. {df['arrival_date'].max():%Y-%m-%d}")
    status = pd.crosstab(df["hotel"], df["reservation_status"], margins=True, margins_name="Total")
    status["cancel_rate"] = (status["Canceled"] + status["No-Show"]) / status["Total"]
    print(status.round(3).to_string())


def data_quality(df: pd.DataFrame) -> None:
    section("Data quality")
    missing = df[["children", "country", "agent", "company"]].isna().sum()
    print("missing values:", missing.to_dict())
    print(f"distinct agents: {df['agent'].nunique()}, companies: {df['company'].nunique()}")
    undefined = {
        col: int((df[col] == "Undefined").sum())
        for col in ("meal", "market_segment", "distribution_channel")
    }
    print("'Undefined' categories:", undefined)
    nights = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    guests = df["adults"] + df["children"].fillna(0) + df["babies"]
    print(
        f"adr < 0: {(df['adr'] < 0).sum()}, adr == 0: {(df['adr'] == 0).sum()}, "
        f"adr > 1000: {(df['adr'] > 1000).sum()}, zero nights: {(nights == 0).sum()}, "
        f"zero guests: {(guests == 0).sum()}"
    )
    duplicated = df.drop(columns=DERIVED_COLUMNS).duplicated()
    print(f"fully duplicated rows: {duplicated.sum()} ({duplicated.mean():.1%})")


def cancellation_timing(df: pd.DataFrame) -> None:
    section("When bookings are lost")
    cancelled = df[df["reservation_status"] == "Canceled"]
    days_before = (cancelled["arrival_date"] - cancelled["cancel_date"]).dt.days
    print(f"days from cancellation to arrival: median {days_before.median():.0f}")
    for days in (7, 14, 30):
        print(f"  cancelled <= {days} days before arrival: {(days_before <= days).mean():.1%}")
    no_show = df[df["reservation_status"] == "No-Show"]
    print(
        f"sanity: cancelled after arrival {(days_before < 0).sum()}, "
        f"before booking {(cancelled['cancel_date'] < cancelled['booking_date']).sum()}; "
        f"no-shows dated on arrival day {(no_show['cancel_date'] == no_show['arrival_date']).sum()}"
        f" of {len(no_show)}"
    )

    nights = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    revenue = df["adr"].clip(lower=0) * nights
    lost = df["is_canceled"] == 1
    late = (df["reservation_status"] == "No-Show") | (
        lost & ((df["arrival_date"] - df["cancel_date"]).dt.days <= 7)
    )
    months = df["arrival_date"].dt.to_period("M").nunique()
    print(
        f"booked revenue (adr * nights): {revenue.sum() / 1e6:.1f}M, "
        f"lost to cancellations and no-shows: {revenue[lost].sum() / 1e6:.1f}M "
        f"({revenue[lost].sum() / revenue.sum():.1%})"
    )
    print(
        f"late losses (no-show or cancelled <= 7 days before arrival): {late.sum()} bookings "
        f"({late.sum() / lost.sum():.1%} of losses), {revenue[late].sum() / 1e6:.2f}M, "
        f"{revenue[late].sum() / months / 1e3:.0f}K per month over {months} months"
    )
    print(f"mean booking value of late losses: {revenue[late].mean():.0f}")


def leakage_evidence(df: pd.DataFrame) -> None:
    section("Leakage evidence: cancellation rate when the condition holds / does not hold")
    offline = df["market_segment"] == "Offline TA/TO"
    conditions = {
        "required_car_parking_spaces > 0": df["required_car_parking_spaces"] > 0,
        "assigned_room_type != reserved_room_type": (
            df["assigned_room_type"] != df["reserved_room_type"]
        ),
        "booking_changes > 0": df["booking_changes"] > 0,
        "total_of_special_requests > 0": df["total_of_special_requests"] > 0,
        "country == PRT": df["country"] == "PRT",
        "country == PRT (Offline TA/TO only)": (df["country"] == "PRT")[offline],
        "deposit_type == Non Refund": df["deposit_type"] == "Non Refund",
        "previous_cancellations > 0": df["previous_cancellations"] > 0,
    }
    rows = []
    for name, condition in conditions.items():
        target = df.loc[condition.index, "is_canceled"]
        rows.append(
            {
                "condition": name,
                "n_true": int(condition.sum()),
                "rate_true": target[condition].mean(),
                "rate_false": target[~condition].mean(),
            }
        )
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    non_refund = df[df["deposit_type"] == "Non Refund"]
    segments = ["Groups", "Offline TA/TO"]
    shares = {
        "country PRT": non_refund["country"].eq("PRT").mean(),
        "Groups or Offline TA/TO": non_refund["market_segment"].isin(segments).mean(),
        "duplicated rows": non_refund.drop(columns=DERIVED_COLUMNS).duplicated().mean(),
    }
    details = ", ".join(f"{name} {share:.1%}" for name, share in shares.items())
    print(f"\nNon Refund bookings: {len(non_refund)}; {details}")
    lost = non_refund[non_refund["is_canceled"] == 1]
    per_agent_day = lost.groupby(["cancel_date", "agent"]).size().rename("bookings").reset_index()
    top = per_agent_day.nlargest(1, "bookings").iloc[0]
    print(
        f"largest one-day cancellation by one agent: {top['bookings']} bookings "
        f"on {top['cancel_date']:%Y-%m-%d}"
    )


def snapshot_lists(df: pd.DataFrame) -> None:
    section("Daily lists: active bookings arriving in the next H days (mean per hotel per day)")
    rows = []
    for horizon in HORIZONS:
        snaps = build_snapshots(df, horizon)
        days = pd.date_range(snaps["snapshot_date"].min(), snaps["snapshot_date"].max())
        daily = (
            snaps.groupby(["hotel", "snapshot_date"])["label"]
            .agg(["size", "sum"])
            .reindex(pd.MultiIndex.from_product([sorted(df["hotel"].unique()), days]))
            .fillna(0)
        )
        for period, mask in (
            ("all", pd.Series(True, index=daily.index)),
            ("test", daily.index.get_level_values(1) >= TEST_START),
        ):
            part = daily[mask].groupby(level=0)
            stats = part.mean().rename(columns={"size": "list_size", "sum": "cancellations"})
            stats["cancel_rate"] = part["sum"].sum() / part["size"].sum()
            for hotel, values in stats.iterrows():
                rows.append({"H": horizon, "period": period, "hotel": hotel, **values.to_dict()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"test period: snapshots from {TEST_START:%Y-%m-%d}")


def split_sizes(df: pd.DataFrame) -> None:
    section(f"Out-of-time split of daily lists (H = {MAIN_HORIZON})")
    snaps = build_snapshots(df, MAIN_HORIZON)
    rows = []
    for name, (start, end) in SPLITS.items():
        part = snaps[snaps["snapshot_date"].between(start, end)]
        rows.append(
            {
                "split": name,
                "snapshot_days": f"{start:%Y-%m-%d} .. {end:%Y-%m-%d}",
                "rows": len(part),
                "bookings": part["booking_idx"].nunique(),
                "positive_rate": part["label"].mean(),
            }
        )
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def single_feature_auc(df: pd.DataFrame) -> None:
    section(f"Single-feature ROC-AUC: all bookings vs daily lists (H = {MAIN_HORIZON})")
    snaps = build_snapshots(df, MAIN_HORIZON)
    snaps["lead_time"] = df.loc[snaps["booking_idx"], "lead_time"].to_numpy()
    scores = {
        "lead_time, all bookings": roc_auc(df["lead_time"], df["is_canceled"]),
        "lead_time, daily lists": roc_auc(snaps["lead_time"], snaps["label"]),
        "days_since_booking, daily lists": roc_auc(snaps["days_since_booking"], snaps["label"]),
    }
    for name, auc in scores.items():
        print(f"{name:<32} {auc:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=RAW_PATH)
    args = parser.parse_args()
    df = load_bookings(download(args.path))
    overview(df)
    data_quality(df)
    cancellation_timing(df)
    leakage_evidence(df)
    snapshot_lists(df)
    split_sizes(df)
    single_feature_auc(df)


if __name__ == "__main__":
    main()
