"""Training dashboard — a reproducible visual read on marathon readiness.

Reads the Apple Health gold facts and renders a multi-panel PNG to
docs/charts/dashboard.png. Regenerate any time with:
    uv run python -m analysis.charts
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from analysis.marathon_training import (
    GOAL_PACE_MIN_PER_MI,
    GOAL_TIME,
    LONG_RUN_MI,
    MP_TARGET_MIN_PER_MI,
    RACE_DATE,
    load,
    quality_miles,
    weekly_summary,
)

CHART_DIR = Path(__file__).resolve().parent.parent / "docs" / "charts"

INK = "#1d2433"
MUTED = "#8a94a6"
GRID = "#e7ebf0"
ACCENT = "#2a9d8f"   # primary (volume, distance)
MP_C = "#2a9d8f"     # marathon-pace miles
TEMPO_C = "#e76f51"  # tempo / faster miles
WARN = "#e76f51"

RACE_WEIGHT_TARGET = 168  # realistic-by-September target from the handoff


def _style(ax, title):
    ax.set_title(title, color=INK, fontsize=12, fontweight="bold", loc="left", pad=8)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    return ax


def _months(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


def panel_weekly_mileage(ax, weekly):
    _style(ax, "Weekly mileage")
    ax.bar(weekly["date"], weekly["total_mi"], width=5.5, color=ACCENT, zorder=3)
    peak = weekly.loc[weekly["total_mi"].idxmax()]
    ax.annotate(
        f"peak {peak['total_mi']:.0f}",
        (peak["date"], peak["total_mi"]),
        textcoords="offset points", xytext=(0, 4), ha="center",
        fontsize=8, color=INK, fontweight="bold",
    )
    _months(ax)
    ax.set_ylabel("miles", color=MUTED, fontsize=9)


def panel_long_runs(ax, runs):
    _style(ax, "Long-run progression (≥ 10 mi)")
    lr = runs[runs["distance_mi"] >= LONG_RUN_MI]
    ax.plot(lr["date"], lr["distance_mi"], "-", color=GRID, linewidth=2, zorder=2)
    ax.scatter(lr["date"], lr["distance_mi"], s=40, color=ACCENT, zorder=3)
    top = lr.loc[lr["distance_mi"].idxmax()]
    ax.annotate(
        f"{top['distance_mi']:.0f} mi",
        (top["date"], top["distance_mi"]),
        textcoords="offset points", xytext=(0, 6), ha="center",
        fontsize=8, color=INK, fontweight="bold",
    )
    ax.axhline(21, color=MUTED, linestyle=":", linewidth=1)
    ax.annotate("planned peak 21", (lr["date"].min(), 21), textcoords="offset points",
                xytext=(0, 3), fontsize=8, color=MUTED)
    _months(ax)
    ax.set_ylabel("miles", color=MUTED, fontsize=9)


def panel_quality(ax, weekly, labeled):
    _style(ax, "Quality miles per week (MP vs. tempo+)")
    wk = (
        labeled.set_index("date").groupby([pd.Grouper(freq="W-SUN"), "label"])
        .size().unstack(fill_value=0)
    )
    idx = wk.index
    mp = wk.get("MP", pd.Series(0, index=idx))
    tempo = wk.get("tempo", pd.Series(0, index=idx))
    ax.bar(idx, mp, width=5.5, color=MP_C, zorder=3, label="marathon pace")
    ax.bar(idx, tempo, width=5.5, bottom=mp, color=TEMPO_C, zorder=3, label="tempo / faster")
    _months(ax)
    ax.set_ylabel("miles", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")


def panel_weight(ax, daily):
    _style(ax, "Body weight")
    w = daily.dropna(subset=["body_mass_lb"]).sort_values("date")
    w = w[w["date"].dt.year == 2026]
    ax.plot(w["date"], w["body_mass_lb"], color=GRID, linewidth=1, zorder=2)
    roll = w.set_index("date")["body_mass_lb"].rolling("14D").mean()
    ax.plot(roll.index, roll.values, color=ACCENT, linewidth=2.5, zorder=3)
    latest = w.iloc[-1]
    ax.annotate(f"{latest['body_mass_lb']:.0f} lb", (latest["date"], latest["body_mass_lb"]),
                textcoords="offset points", xytext=(6, 0), fontsize=8, color=INK, fontweight="bold")
    ax.axhline(RACE_WEIGHT_TARGET, color=MUTED, linestyle=":", linewidth=1)
    ax.annotate(f"target ~{RACE_WEIGHT_TARGET}", (w["date"].min(), RACE_WEIGHT_TARGET),
                textcoords="offset points", xytext=(0, 3), fontsize=8, color=MUTED)
    _months(ax)
    ax.set_ylabel("lb", color=MUTED, fontsize=9)


def panel_trend(ax, daily, col, title, unit, good_down, target_line=None):
    _style(ax, title)
    d = daily.dropna(subset=[col]).sort_values("date")
    d = d[d["date"].dt.year == 2026]
    roll = d.set_index("date")[col].rolling("14D").mean()
    ax.plot(d["date"], d[col], color=GRID, linewidth=1, zorder=2)
    ax.plot(roll.index, roll.values, color=ACCENT, linewidth=2.5, zorder=3)
    if len(roll.dropna()):
        latest = roll.dropna().iloc[-1]
        ax.annotate(f"{latest:.0f} {unit}", (roll.dropna().index[-1], latest),
                    textcoords="offset points", xytext=(6, 0), fontsize=8,
                    color=INK, fontweight="bold")
    _months(ax)
    ax.set_ylabel(unit, color=MUTED, fontsize=9)


def build_dashboard() -> Path:
    d = load()
    runs = d["runs_2026"]
    weekly = weekly_summary(runs)
    labeled = quality_miles(d["splits"], d["runs"])
    labeled = labeled[labeled["date"].dt.year == 2026]
    daily = d["daily"]

    plt.rcParams["font.family"] = "sans-serif"
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    fig.patch.set_facecolor("white")

    weeks_out = (pd.Timestamp(RACE_DATE).date() - runs["date"].max().date()).days / 7
    fig.suptitle(
        f"Marathon readiness — Marquette {RACE_DATE}  ·  ~{weeks_out:.0f} weeks out  ·  goal sub-{GOAL_TIME}",
        color=INK, fontsize=15, fontweight="bold", x=0.02, ha="left", y=0.995,
    )

    panel_weekly_mileage(axes[0, 0], weekly)
    panel_long_runs(axes[0, 1], runs)
    panel_quality(axes[1, 0], weekly, labeled)
    panel_weight(axes[1, 1], daily)
    panel_trend(axes[2, 0], daily, "resting_hr", "Resting HR (14-day avg)", "bpm", good_down=True)
    panel_trend(axes[2, 1], daily, "hrv_ms", "HRV (14-day avg)", "ms", good_down=False)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    out = CHART_DIR / "dashboard.png"
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(f"Wrote {build_dashboard()}")
