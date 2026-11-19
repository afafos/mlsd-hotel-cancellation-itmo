"""Business boundaries of the retrospective morning candidate lists."""

import unittest

import pandas as pd

from hotel_cancellation.audit import MAIN_HORIZON, SPLITS
from hotel_cancellation.snapshots import build_snapshots


def bookings(rows: list[tuple[int, str, str, str | None, int]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows, columns=["id", "booking_date", "arrival_date", "cancel_date", "is_canceled"]
    ).set_index("id")
    for column in ("booking_date", "arrival_date", "cancel_date"):
        frame[column] = pd.to_datetime(frame[column])
    frame["hotel"] = "City Hotel"
    return frame


class SnapshotTests(unittest.TestCase):
    def test_morning_creation_cancellation_and_arrival_boundaries(self) -> None:
        frame = bookings(
            [
                (101, "2017-01-01", "2017-01-10", None, 0),
                (205, "2017-01-07", "2017-01-10", None, 0),
                (309, "2017-01-01", "2017-01-10", "2017-01-07", 1),
                (412, "2017-01-01", "2017-01-10", "2017-01-06", 1),
                (516, "2017-01-01", "2017-01-07", None, 0),
                (620, "2017-01-10", "2017-01-10", None, 0),
                (724, "2017-01-07", "2017-01-10", "2017-01-07", 1),
                (828, "2017-01-01", "2017-01-10", "2017-01-10", 1),
            ]
        )
        result = build_snapshots(frame, 3, pd.Timestamp("2017-01-07"), pd.Timestamp("2017-01-10"))
        actual = set(
            zip(
                result["booking_idx"].tolist(),
                result["snapshot_date"].dt.strftime("%Y-%m-%d").tolist(),
                result["label"].tolist(),
                strict=True,
            )
        )
        expected = {
            (101, "2017-01-07", 0),
            (101, "2017-01-08", 0),
            (101, "2017-01-09", 0),
            (205, "2017-01-08", 0),
            (205, "2017-01-09", 0),
            (309, "2017-01-07", 1),
            (828, "2017-01-07", 1),
            (828, "2017-01-08", 1),
            (828, "2017-01-09", 1),
        }
        self.assertEqual(actual, expected)
        self.assertEqual(len(result), len(expected))
        self.assertTrue(result["days_to_arrival"].between(1, 3).all())
        self.assertTrue((result["days_since_booking"] >= 1).all())
        self.assertEqual(result["hotel"].unique().tolist(), ["City Hotel"])

    def test_default_end_excludes_incomplete_horizon(self) -> None:
        frame = bookings(
            [
                (11, "2017-01-01", "2017-01-10", None, 0),
                (22, "2017-01-01", "2017-01-20", None, 0),
            ]
        )
        result = build_snapshots(frame, 3)
        self.assertEqual(result["snapshot_date"].tolist(), [pd.Timestamp("2017-01-17")])
        self.assertEqual(result["booking_idx"].tolist(), [22])

    def test_no_candidates_preserves_output_schema(self) -> None:
        frame = bookings([(1, "2017-01-10", "2017-01-10", None, 0)])
        result = build_snapshots(frame, 14, pd.Timestamp("2017-01-01"), pd.Timestamp("2017-01-31"))
        self.assertTrue(result.empty)
        self.assertEqual(
            list(result.columns),
            [
                "booking_idx",
                "hotel",
                "snapshot_date",
                "days_to_arrival",
                "days_since_booking",
                "label",
            ],
        )

    def test_split_gaps_separate_bookings_but_not_terminal_label_dates(self) -> None:
        # A guest arriving before the next split may check out after that split starts.
        # Booking isolation must not be mistaken for label readiness at model fitting.
        frame = bookings(
            [
                (1, "2016-11-01", "2016-12-20", None, 0),
                (2, "2016-11-01", "2017-01-05", None, 0),
                (3, "2017-03-01", "2017-04-20", None, 0),
                (4, "2017-03-01", "2017-05-05", None, 0),
            ]
        )
        frame["reservation_status_date"] = pd.to_datetime(
            ["2017-01-02", "2017-01-06", "2017-05-02", "2017-05-06"]
        )
        result = build_snapshots(
            frame, MAIN_HORIZON, start=SPLITS["train"][0], end=SPLITS["test"][1]
        )
        split_ids = {
            name: set(result.loc[result["snapshot_date"].between(start, end), "booking_idx"])
            for name, (start, end) in SPLITS.items()
        }
        self.assertEqual(split_ids, {"train": {1}, "valid": {2, 3}, "test": {4}})
        self.assertFalse(split_ids["train"] & split_ids["valid"])
        self.assertFalse(split_ids["valid"] & split_ids["test"])
        self.assertFalse(split_ids["train"] & split_ids["test"])
        train = frame.loc[list(split_ids["train"])]
        valid_start = SPLITS["valid"][0]
        self.assertTrue((train["arrival_date"] < valid_start).all())
        self.assertFalse((train["reservation_status_date"] < valid_start).all())


if __name__ == "__main__":
    unittest.main()
