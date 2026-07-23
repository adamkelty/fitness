"""Training dashboard — a reproducible visual read on marathon readiness.

Reads the Apple Health gold facts and renders a multi-panel PNG to
docs/charts/dashboard.png. Regenerate any time with:
    uv run python -m analysis.charts
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.marathon_training import (
    GOAL_PACE_MIN_PER_MI,
    GOAL_TIME,
    LONG_RUN_MI,
    RACE_DATE,
    format_pace,
    load,
    quality_miles,
    weekly_summary,
)

CHART_DIR = Path(__file__).resolve().parent.parent / "docs" / "charts"

INK = "#1d2433"
MUTED = "#8a94a6"
GRID = "#e7ebf0"
ACCENT = "#2a9d8f"   # primary (volume, distance)
PLAN_C = "#c9b98f"   # planned (future) — distinct, muted gold
MP_C = "#2a9d8f"
TEMPO_C = "#e76f51"
WARN = "#e76f51"

RACE_WEIGHT_TARGET = 168

# Remaining weekly plan (week-ending Sunday -> target miles), from the training
# plan. Actuals fill in as weeks complete; these are the forward targets.
PLAN_WEEKLY = {
    "2026-08-02": 48,
    "2026-08-09": 52,
    "2026-08-16": 56,  # peak
    "2026-08-23": 44,  # taper begins
    "2026-08-30": 32,
    "2026-09-06": 16,  # race week (race Sat Sep 5)
}


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


def panel_plan_vs_actual(ax, weekly, today):
    _style(ax, "Weekly mileage — plan vs. actual")
    ax.bar(weekly["date"], weekly["total_mi"], width=5.5, color=ACCENT, zorder=3, label="actual")
    plan_dates = [pd.Timestamp(d) for d in PLAN_WEEKLY]
    plan_vals = list(PLAN_WEEKLY.values())
    ax.bar(plan_dates, plan_vals, width=5.5, facecolor="none", edgecolor=PLAN_C,
           linewidth=2, hatch="///", zorder=3, label="planned")
    ax.axvline(pd.Timestamp(today), color=MUTED, linestyle=":", linewidth=1)
    ax.annotate("now", (pd.Timestamp(today), ax.get_ylim()[1]), textcoords="offset points",
                xytext=(3, -10), fontsize=8, color=MUTED)
    _months(ax)
    ax.set_ylabel("miles", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")


def panel_long_runs(ax, runs):
    _style(ax, "Long-run progression (≥ 10 mi)")
    lr = runs[runs["distance_mi"] >= LONG_RUN_MI]
    ax.plot(lr["date"], lr["distance_mi"], "-", color=GRID, linewidth=2, zorder=2)
    ax.scatter(lr["date"], lr["distance_mi"], s=40, color=ACCENT, zorder=3)
    top = lr.loc[lr["distance_mi"].idxmax()]
    ax.annotate(f"{top['distance_mi']:.0f} mi", (top["date"], top["distance_mi"]),
                textcoords="offset points", xytext=(0, 6), ha="center",
                fontsize=8, color=INK, fontweight="bold")
    ax.axhline(21, color=MUTED, linestyle=":", linewidth=1)
    ax.annotate("planned peak 21", (lr["date"].min(), 21), textcoords="offset points",
                xytext=(0, 3), fontsize=8, color=MUTED)
    _months(ax)
    ax.set_ylabel("miles", color=MUTED, fontsize=9)


def panel_quality(ax, labeled):
    _style(ax, "Quality miles per week (MP vs. tempo+)")
    wk = (labeled.set_index("date").groupby([pd.Grouper(freq="W-SUN"), "label"])
          .size().unstack(fill_value=0))
    idx = wk.index
    mp = wk.get("MP", pd.Series(0, index=idx))
    tempo = wk.get("tempo", pd.Series(0, index=idx))
    ax.bar(idx, mp, width=5.5, color=MP_C, zorder=3, label="marathon pace")
    ax.bar(idx, tempo, width=5.5, bottom=mp, color=TEMPO_C, zorder=3, label="tempo / faster")
    _months(ax)
    ax.set_ylabel("miles", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")


def compute_decoupling(labeled, min_easy=4):
    """Aerobic decoupling per long run: does pace:HR efficiency drift over the
    run? Computed on easy miles only (excludes intentional MP/tempo surges), as
    first-half vs second-half efficiency. <5% = good aerobic durability."""
    rows = []
    for wid, g in labeled.groupby("workout_id"):
        if g["distance_mi"].iloc[0] < LONG_RUN_MI:
            continue
        easy = g[(g["label"] == "easy") & g["avg_heartrate"].notna()].sort_values("mile")
        if len(easy) < min_easy:
            continue
        half = len(easy) // 2
        first, second = easy.iloc[:half], easy.iloc[half:]
        ef = lambda df: ((60 / df["pace_min_per_mi"]) / df["avg_heartrate"]).mean()
        ef1, ef2 = ef(first), ef(second)
        rows.append({"date": g["date"].iloc[0], "decoupling_pct": (ef1 - ef2) / ef1 * 100})
    return pd.DataFrame(rows).sort_values("date")


def panel_decoupling(ax, labeled):
    _style(ax, "Long-run aerobic decoupling (easy-mile pace:HR drift)")
    dec = compute_decoupling(labeled)
    colors = [WARN if v > 5 else ACCENT for v in dec["decoupling_pct"]]
    ax.bar(dec["date"], dec["decoupling_pct"], width=5.5, color=colors, zorder=3)
    ax.axhline(5, color=MUTED, linestyle=":", linewidth=1)
    ax.annotate("5% — good durability", (dec["date"].min(), 5), textcoords="offset points",
                xytext=(0, 3), fontsize=8, color=MUTED)
    ax.axhline(0, color=GRID, linewidth=1)
    _months(ax)
    ax.set_ylabel("%", color=MUTED, fontsize=9)


def panel_pace_hr(ax, runs):
    _style(ax, "Pace vs. heart rate (outdoor steady runs)")
    r = runs[(~runs["indoor"]) & runs["avg_heartrate"].notna()
             & (runs["distance_mi"] >= 3) & (runs["pace_min_per_mi"] < 11)].copy()
    r["month"] = r["date"].dt.month
    sc = ax.scatter(r["pace_min_per_mi"], r["avg_heartrate"], c=r["month"],
                    cmap="viridis", s=32, zorder=3, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("pace (min/mi)", color=MUTED, fontsize=9)
    ax.set_ylabel("avg HR", color=MUTED, fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: format_pace(x)))
    cb = plt.colorbar(sc, ax=ax, ticks=range(1, 13), pad=0.02)
    cb.ax.set_yticklabels([pd.Timestamp(2026, m, 1).strftime("%b") for m in range(1, 13)],
                          fontsize=7, color=MUTED)
    cb.outline.set_visible(False)
    ax.annotate("down-left over time = fitter", (0.5, 0.03), xycoords="axes fraction",
                fontsize=8, color=MUTED, style="italic")


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


def panel_trend(ax, daily, col, title, unit):
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
    today = runs["date"].max().date()

    plt.rcParams["font.family"] = "sans-serif"
    fig, axes = plt.subplots(4, 2, figsize=(13, 17))
    fig.patch.set_facecolor("white")

    weeks_out = (pd.Timestamp(RACE_DATE).date() - today).days / 7
    fig.suptitle(
        f"Marathon readiness — Marquette {RACE_DATE}  ·  ~{weeks_out:.0f} weeks out  ·  goal sub-{GOAL_TIME}",
        color=INK, fontsize=15, fontweight="bold", x=0.02, ha="left", y=0.997,
    )

    panel_plan_vs_actual(axes[0, 0], weekly, today)
    panel_long_runs(axes[0, 1], runs)
    panel_quality(axes[1, 0], labeled)
    panel_decoupling(axes[1, 1], labeled)
    panel_pace_hr(axes[2, 0], runs)
    panel_weight(axes[2, 1], daily)
    panel_trend(axes[3, 0], daily, "resting_hr", "Resting HR (14-day avg)", "bpm")
    panel_trend(axes[3, 1], daily, "hrv_ms", "HRV (14-day avg)", "ms")

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    out = CHART_DIR / "dashboard.png"
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(f"Wrote {build_dashboard()}")
