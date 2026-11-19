"""Shared helpers for aggregate report tables and figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

HOTELS = ("City Hotel", "Resort Hotel")
COLORS = ("#2563a6", "#d17b21")
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
}


def prepare_output(output: Path) -> None:
    """Create the report folders and apply the shared plot style."""
    (output / "figures").mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)


def save_table(table: pd.DataFrame, output: Path, name: str) -> None:
    """Export deterministic aggregates, never booking-level records."""
    table.to_csv(output / "tables" / f"{name}.csv", index=False, float_format="%.8f")


def save_figure(fig: Figure, output: Path, name: str) -> None:
    fig.savefig(output / "figures" / f"{name}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
