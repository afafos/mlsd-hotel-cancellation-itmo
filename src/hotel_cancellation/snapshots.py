import numpy as np
import numpy.typing as npt
import pandas as pd

_EPOCH = pd.Timestamp("1970-01-01")


def _day_numbers(dates: pd.Series) -> npt.NDArray[np.int64]:
    """Convert a datetime series without NaT to day numbers since the Unix epoch."""
    return dates.to_numpy(dtype="datetime64[D]").astype(np.int64)


def build_snapshots(
    bookings: pd.DataFrame,
    horizon_days: int,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Expand bookings into (booking, snapshot day) rows.

    Args:
        bookings: output of ``load_bookings``.
        horizon_days: list horizon H in days.
        start: first snapshot day; defaults to the first arrival date in the data.
        end: last snapshot day; defaults to the last day whose whole horizon is covered by data.

    Returns:
        One row per candidate with columns: booking_idx (index label in ``bookings``),
        hotel, snapshot_date, days_to_arrival, days_since_booking, label.
    """
    arrival = bookings["arrival_date"]
    if start is None:
        start = pd.Timestamp(arrival.min())
    if end is None:
        end = pd.Timestamp(arrival.max()) - pd.Timedelta(days=horizon_days)

    booked = _day_numbers(bookings["booking_date"])
    arrive = _day_numbers(arrival)
    active_until = _day_numbers(bookings["cancel_date"].fillna(arrival))

    first = np.maximum(np.maximum(booked + 1, arrive - horizon_days), (start - _EPOCH).days)
    last = np.minimum(np.minimum(arrive - 1, active_until), (end - _EPOCH).days)
    counts = np.clip(last - first + 1, 0, None)

    rows = np.repeat(np.arange(len(bookings)), counts)
    offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    snapshot_day = np.repeat(first, counts) + offsets

    return pd.DataFrame(
        {
            "booking_idx": bookings.index.to_numpy()[rows],
            "hotel": bookings["hotel"].to_numpy()[rows],
            "snapshot_date": pd.to_datetime(snapshot_day, unit="D"),
            "days_to_arrival": arrive[rows] - snapshot_day,
            "days_since_booking": snapshot_day - booked[rows],
            "label": bookings["is_canceled"].to_numpy()[rows],
        }
    )
