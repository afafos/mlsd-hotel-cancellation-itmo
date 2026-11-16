import argparse
import hashlib
import shutil
import ssl
import urllib.request
from pathlib import Path

import certifi
import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/"
    "d75aaa0d31596ad6487ae0db20138067d301e031/data/2020/2020-02-11/hotels.csv"
)
DATA_SHA256 = "7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06"
RAW_PATH = Path("data/raw/hotel_bookings.csv")

MONTHS = {
    name: number
    for number, name in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        start=1,
    )
}


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def download(path: Path = RAW_PATH, url: str = DATA_URL) -> Path:
    """Download the raw CSV if it is missing and verify its checksum."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=context) as response, partial.open("wb") as f:
            shutil.copyfileobj(response, f)
        partial.replace(path)
    digest = file_sha256(path)
    if digest != DATA_SHA256:
        msg = f"Checksum mismatch for {path}: got {digest}, expected {DATA_SHA256}"
        raise ValueError(msg)
    return path


def load_bookings(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read raw bookings and add derived date columns.

    Added columns:
        arrival_date: assembled from the arrival_date_* columns.
        booking_date: arrival_date - lead_time, the day the booking was entered into the PMS.
        cancel_date: the day the booking was lost (reservation_status_date): the cancellation
            day for "Canceled", the arrival day for "No-Show", NaT for "Check-Out".
            It is used only to build labels and snapshots, never as a feature.
    """
    df = pd.read_csv(path, na_values=["NULL"])
    df["arrival_date"] = pd.to_datetime(
        pd.DataFrame(
            {
                "year": df["arrival_date_year"],
                "month": df["arrival_date_month"].map(MONTHS),
                "day": df["arrival_date_day_of_month"],
            }
        )
    )
    df["booking_date"] = df["arrival_date"] - pd.to_timedelta(df["lead_time"], unit="D")
    status_date = pd.to_datetime(df["reservation_status_date"])
    df["cancel_date"] = status_date.where(df["is_canceled"] == 1)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=RAW_PATH)
    args = parser.parse_args()
    path = download(args.path)
    print(f"Dataset is ready: {path} (sha256 verified)")


if __name__ == "__main__":
    main()
