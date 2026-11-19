"""Ranking metrics, rule baselines and feature diagnostics on synthetic data."""

import unittest

import numpy as np
import pandas as pd

from hotel_cancellation.audit import TEST_START
from hotel_cancellation.baselines import (
    MIN_SUPPORT,
    candidates,
    fit_rates,
    history,
    score_rates,
    window,
)
from hotel_cancellation.feature_eda import cramers_v, encode, vif
from hotel_cancellation.metrics import (
    FloatArray,
    average_precision,
    daily_top_k,
    expected_top_k_hits,
    summarize_top_k,
)


def array(*values: float) -> FloatArray:
    return np.array(values, dtype=np.float64)


class MetricTests(unittest.TestCase):
    def test_ties_are_resolved_in_expectation(self) -> None:
        score = array(3, 2, 2, 2, 1)
        label = array(0, 1, 0, 0, 1)
        self.assertAlmostEqual(expected_top_k_hits(score, label, 2), 1 / 3)
        self.assertAlmostEqual(expected_top_k_hits(score, label, 4), 1.0)
        self.assertEqual(expected_top_k_hits(score, label, 10), 2.0)

    def test_row_order_cannot_change_a_tied_ranking(self) -> None:
        # The source CSV lists cancelled bookings first inside a day: order must not leak.
        score = np.zeros(6)
        label = array(1, 1, 0, 0, 0, 0)
        reversed_hits = expected_top_k_hits(score[::-1], label[::-1], 2)
        self.assertAlmostEqual(expected_top_k_hits(score, label, 2), 2 / 3)
        self.assertAlmostEqual(reversed_hits, 2 / 3)

    def test_average_precision_counts_a_tie_as_one_threshold(self) -> None:
        score = array(0.9, 0.8, 0.8, 0.1)
        label = array(1, 0, 1, 0)
        self.assertAlmostEqual(average_precision(score, label), 0.5 * 1 + 0.5 * 2 / 3)

    def test_macro_metrics_follow_the_design_doc(self) -> None:
        frame = pd.DataFrame(
            {
                "hotel": "City Hotel",
                "snapshot_date": pd.to_datetime(["2017-01-01"] * 4 + ["2017-01-02"] * 2),
                "score": [4.0, 3.0, 2.0, 1.0, 2.0, 1.0],
                "label": [1, 0, 1, 0, 0, 0],
            }
        )
        daily = daily_top_k(frame, "score", (2,))
        self.assertEqual(daily["hits_2"].tolist(), [1.0, 0.0])
        summary = summarize_top_k(daily, (2,))
        self.assertAlmostEqual(summary["precision_at_2"], (1 / 2 + 0 / 2) / 2)
        self.assertAlmostEqual(summary["recall_at_2"], 1 / 2)
        self.assertAlmostEqual(summary["mean_daily_rate"], (2 / 4 + 0) / 2)
        self.assertAlmostEqual(summary["lift_at_2"], 0.25 / 0.25)
        self.assertAlmostEqual(summary["ceiling_precision_at_2"], (2 / 2 + 0) / 2)
        self.assertEqual(summary["days_without_positives"], 1.0)

    def test_short_list_uses_available_budget(self) -> None:
        frame = pd.DataFrame(
            {
                "hotel": "Resort Hotel",
                "snapshot_date": pd.Timestamp("2017-01-01"),
                "score": [1.0, 0.0],
                "label": [1, 0],
            }
        )
        summary = summarize_top_k(daily_top_k(frame, "score", (20,)), (20,))
        self.assertAlmostEqual(summary["precision_at_20"], 1 / 2)
        self.assertAlmostEqual(summary["recall_at_20"], 1.0)


def segment_frame() -> pd.DataFrame:
    rows = []
    for i in range(MIN_SUPPORT):
        rows.append(("City Hotel", "Online TA", i, int(i < 6)))
    for i in range(MIN_SUPPORT - 1):
        rows.append(("City Hotel", "Groups", 100 + i, 1))
    rows.append(("City Hotel", "Groups", 100, 1))
    frame = pd.DataFrame(rows, columns=["hotel", "market_segment", "booking_idx", "label"])
    return frame


class SegmentRateTests(unittest.TestCase):
    def test_smoothing_and_support_in_unique_bookings(self) -> None:
        train = segment_frame()
        segments, hotel_rate = fit_rates(train, ("hotel", "market_segment"), alpha=10)
        prior = float(train["label"].mean())
        self.assertAlmostEqual(hotel_rate["City Hotel"], prior)
        online = segments.set_index("market_segment").loc["Online TA"]
        self.assertTrue(online["supported"])
        self.assertAlmostEqual(online["rate"], (6 + 10 * prior) / (MIN_SUPPORT + 10))
        groups = segments.set_index("market_segment").loc["Groups"]
        self.assertEqual(groups["snapshots"], MIN_SUPPORT)
        self.assertFalse(groups["supported"])

    def test_rare_and_unseen_segments_fall_back_to_hotel_rate(self) -> None:
        train = segment_frame()
        keys = ("hotel", "market_segment")
        segments, hotel_rate = fit_rates(train, keys, alpha=10)
        evaluation = pd.DataFrame(
            {"hotel": "City Hotel", "market_segment": ["Groups", "Direct", "Online TA"]},
            index=[7, 8, 9],
        )
        scores = score_rates(evaluation, keys, segments, hotel_rate)
        self.assertEqual(scores.index.tolist(), [7, 8, 9])
        self.assertAlmostEqual(scores[7], hotel_rate["City Hotel"])
        self.assertAlmostEqual(scores[8], hotel_rate["City Hotel"])
        self.assertNotAlmostEqual(scores[9], hotel_rate["City Hotel"])


class WindowTests(unittest.TestCase):
    def test_history_keeps_the_gap_and_mature_labels(self) -> None:
        start = pd.Timestamp("2017-01-01")
        frame = pd.DataFrame(
            {
                "snapshot_date": pd.to_datetime(["2016-12-17", "2016-12-18", "2016-12-10"]),
                "status_date": pd.to_datetime(["2016-12-20", "2016-12-20", "2017-01-01"]),
            }
        )
        kept = history(frame, start)
        self.assertEqual(kept["snapshot_date"].tolist(), [pd.Timestamp("2016-12-17")])

    def test_window_drops_outcomes_unknown_at_decision_time(self) -> None:
        frame = pd.DataFrame(
            {
                "snapshot_date": pd.to_datetime(["2017-01-05", "2017-02-14", "2017-02-15"]),
                "status_date": pd.to_datetime(["2017-02-28", "2017-03-01", "2017-02-20"]),
            }
        )
        kept = window(frame, pd.Timestamp("2017-01-01"), pd.Timestamp("2017-02-14"))
        self.assertEqual(kept["snapshot_date"].tolist(), [pd.Timestamp("2017-01-05")])

    def test_baselines_refuse_to_build_test_period_snapshots(self) -> None:
        with self.assertRaises(ValueError):
            candidates(pd.DataFrame(), TEST_START)


class FeatureDiagnosticsTests(unittest.TestCase):
    def test_exact_linear_identity_gives_infinite_vif(self) -> None:
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(
            {"to_arrival": rng.integers(1, 15, 500), "since_booking": rng.integers(1, 300, 500)}
        )
        frame["lead_time"] = frame["to_arrival"] + frame["since_booking"]
        frame["noise"] = rng.normal(size=500)
        values = vif(frame, ["lead_time", "to_arrival", "since_booking", "noise"])
        self.assertTrue(np.isinf(values[:3]).all())
        self.assertLess(vif(frame, ["to_arrival", "since_booking", "noise"])[2], 1.1)

    def test_cramers_v_bounds(self) -> None:
        left = pd.Series(["a", "b", "c"] * 200)
        self.assertAlmostEqual(cramers_v(left, left), 1.0, places=2)
        rng = np.random.default_rng(42)
        noise = pd.Series(rng.choice(["x", "y"], size=600))
        self.assertLess(cramers_v(left, noise), 0.1)

    def test_encoding_is_fitted_on_train_and_unseen_values_get_prior(self) -> None:
        train = pd.Series(["Non Refund"] * 10 + ["No Deposit"] * 90, name="deposit_type")
        label = pd.Series([1] * 10 + [0] * 90)
        encoded = encode(train, label, pd.Series(["Non Refund", "Refundable"]))
        prior = 0.1
        self.assertAlmostEqual(encoded[0], (10 + 30 * prior) / (10 + 30))
        self.assertAlmostEqual(encoded[1], prior)


if __name__ == "__main__":
    unittest.main()
