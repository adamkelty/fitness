"""Marathon training analysis for the 2026 Marquette race (goal: sub-3:30).

Reads the gold layer, derives imperial + pace metrics (pace in min/mi, the
standard running convention), and renders a living markdown report to
docs/marathon-training.md that refreshes every time this runs.
"""

from datetime import date
from pathlib import Path

import pandas as pd

from pipeline.config import GOLD_DIR, SILVER_DIR

MILES_PER_KM = 0.621371
FEET_PER_METER = 3.28084
MARATHON_MI = 26.2188
LONG_RUN_MI = 10  # threshold for what counts as a long run

# Goal: finish under 3:30. Target race pace ~7:40 min/mi.
GOAL_PACE_MIN_PER_MI = 7 + 40 / 60
SUB_330_PACE_MIN_PER_MI = (3 * 60 + 30) / MARATHON_MI

# Marathon-pace work inside long runs: user currently targets ~7:50/mi.
# Count a split as an MP mile if it lands within ±15s of that target.
MP_TARGET_MIN_PER_MI = 7 + 50 / 60
MP_BAND = 15 / 60
MP_LOW, MP_HIGH = MP_TARGET_MIN_PER_MI - MP_BAND, MP_TARGET_MIN_PER_MI + MP_BAND

REPORT_PATH = Path(__file__).resolve().parent.parent / "docs" / "marathon-training.md"


def format_pace(min_per_mi: float) -> str:
    """Turn a float min/mi into an M:SS pace string."""
    if pd.isna(min_per_mi):
        return "-"
    minutes = int(min_per_mi)
    seconds = round((min_per_mi - minutes) * 60)
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes}:{seconds:02d}"


def load_runs_2026() -> pd.DataFrame:
    fact = pd.read_parquet(GOLD_DIR / "fact_activity.parquet")
    dim_date = pd.read_parquet(GOLD_DIR / "dim_date.parquet")
    dim_activity_type = pd.read_parquet(GOLD_DIR / "dim_activity_type.parquet")

    runs = (
        fact.merge(dim_activity_type, on="type_key")
        .merge(dim_date, on="date_key")
        .loc[lambda df: (df["type_name"] == "Run") & (df["year"] == 2026)]
        .copy()
    )

    runs["distance_mi"] = runs["distance_km"] * MILES_PER_KM
    runs["pace_min_per_mi"] = runs["moving_time_min"] / runs["distance_mi"]
    runs["elevation_gain_ft"] = runs["elevation_gain_m"] * FEET_PER_METER
    runs["elev_ft_per_mi"] = runs["elevation_gain_ft"] / runs["distance_mi"]
    return runs.sort_values("date")


def weekly_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to training weeks (Mon-Sun). Weekly pace is total time / total
    distance, so it isn't skewed by short vs long runs."""
    weekly = (
        runs.set_index("date")
        .resample("W-SUN")
        .agg(
            runs=("distance_mi", "size"),
            total_mi=("distance_mi", "sum"),
            longest_mi=("distance_mi", "max"),
            total_time_min=("moving_time_min", "sum"),
            avg_hr=("avg_heartrate", "mean"),
        )
    )
    weekly = weekly[weekly["runs"] > 0]
    weekly["avg_pace_min_per_mi"] = weekly["total_time_min"] / weekly["total_mi"]
    return weekly.reset_index()


def load_splits() -> pd.DataFrame:
    path = SILVER_DIR / "splits.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["activity_id", "mile", "pace_min_per_mi"])
    return pd.read_parquet(path)


def mp_miles_by_long_run(runs: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """For each long run, count the per-mile splits that land in the marathon-
    pace band.

    Some Apple-Health-imported runs have a GPS/import gap where Strava's splits
    (and streams) lose distance mid-run. When the splits don't reconcile with the
    run's total distance we can't trust the MP count, so those are flagged
    unreliable rather than reported as zero.
    """
    long_runs = runs[runs["distance_mi"] >= LONG_RUN_MI]
    rows = []
    for _, lr in long_runs.iterrows():
        s = splits[splits["activity_id"] == lr["activity_id"]]
        mp = s[s["pace_min_per_mi"].between(MP_LOW, MP_HIGH)]
        missing_mi = lr["distance_mi"] - s["distance_mi"].sum()
        rows.append(
            {
                "date": lr["date"],
                "distance_mi": lr["distance_mi"],
                "mp_miles": len(mp),
                "has_splits": len(s) > 0,
                "reliable": missing_mi <= 0.5,
                "pct_missing": max(0.0, missing_mi / lr["distance_mi"] * 100),
                "mp_paces": [format_pace(p) for p in mp["pace_min_per_mi"]],
            }
        )
    return pd.DataFrame(rows).sort_values("date", ascending=False)


def render_report(runs: pd.DataFrame, weekly: pd.DataFrame, mp_long: pd.DataFrame) -> str:
    total_mi = runs["distance_mi"].sum()
    longest = runs.loc[runs["distance_mi"].idxmax()]
    lines = []
    lines.append("# Marathon Training — Marquette 2026")
    lines.append("")
    lines.append(f"_Last updated: {date.today().isoformat()}_")
    lines.append("")
    lines.append("## Goal")
    lines.append("")
    lines.append("- **Race:** Marquette, MI — Labor Day weekend 2026")
    lines.append("- **Target:** finish under **3:30:00**")
    goal_finish_min = GOAL_PACE_MIN_PER_MI * MARATHON_MI
    lines.append(
        f"- **Goal race pace:** **{format_pace(GOAL_PACE_MIN_PER_MI)} min/mi** "
        f"→ projected finish **{int(goal_finish_min // 60)}:{int(goal_finish_min % 60):02d}:"
        f"{round((goal_finish_min % 1) * 60):02d}**"
    )
    lines.append(
        f"- **Sub-3:30 ceiling:** must average faster than "
        f"**{format_pace(SUB_330_PACE_MIN_PER_MI)} min/mi**"
    )
    lines.append("")
    lines.append("## 2026 so far")
    lines.append("")
    lines.append(f"- **Runs:** {len(runs)}")
    lines.append(f"- **Total distance:** {total_mi:.1f} mi")
    lines.append(
        f"- **Longest run:** {longest['distance_mi']:.1f} mi "
        f"on {longest['date'].date()} @ {format_pace(longest['pace_min_per_mi'])}/mi"
    )
    lines.append("")
    lines.append("## Weekly progression")
    lines.append("")
    lines.append("| Week ending | Runs | Miles | Longest | Avg pace | Avg HR |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, w in weekly.iterrows():
        avg_hr = f"{w['avg_hr']:.0f}" if pd.notna(w["avg_hr"]) else "-"
        lines.append(
            f"| {w['date'].date()} | {int(w['runs'])} | {w['total_mi']:.1f} | "
            f"{w['longest_mi']:.1f} | {format_pace(w['avg_pace_min_per_mi'])} | {avg_hr} |"
        )
    lines.append("")
    long_runs = runs[runs["distance_mi"] >= LONG_RUN_MI]
    lines.append(f"## Long runs (≥ {LONG_RUN_MI} mi)")
    lines.append("")
    lines.append("Elevation matters here — Marquette long runs carry real climbing.")
    lines.append("")
    lines.append("| Date | Miles | Pace | Elev gain (ft) | ft/mi | Avg HR |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in long_runs.iloc[::-1].iterrows():
        avg_hr = f"{r['avg_heartrate']:.0f}" if pd.notna(r["avg_heartrate"]) else "-"
        lines.append(
            f"| {r['date'].date()} | {r['distance_mi']:.1f} | "
            f"{format_pace(r['pace_min_per_mi'])} | {r['elevation_gain_ft']:.0f} | "
            f"{r['elev_ft_per_mi']:.0f} | {avg_hr} |"
        )
    lines.append("")
    lines.append("## Marathon-pace miles in long runs")
    lines.append("")
    lines.append(
        f"Miles run within ±{round(MP_BAND * 60)}s of the {format_pace(MP_TARGET_MIN_PER_MI)}"
        " MP target, from per-mile splits. Watch this count trend up as the build progresses."
    )
    lines.append("")
    lines.append("| Date | Long run (mi) | MP miles | MP splits |")
    lines.append("|---|---:|---:|---|")
    for _, r in mp_long.iterrows():
        if not r["has_splits"]:
            mp_count, paces = "—", "_splits pending_"
        elif not r["reliable"]:
            mp_count = "?"
            paces = f"⚠️ _unreliable — {r['pct_missing']:.0f}% distance missing (GPS/import gap)_"
        elif r["mp_paces"]:
            mp_count, paces = r["mp_miles"], ", ".join(r["mp_paces"])
        else:
            mp_count, paces = r["mp_miles"], "—"
        lines.append(
            f"| {r['date'].date()} | {r['distance_mi']:.1f} | {mp_count} | {paces} |"
        )
    lines.append("")
    lines.append("## Recent runs")
    lines.append("")
    lines.append("| Date | Miles | Time (min) | Pace | Elev (ft) | Avg HR |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in runs.tail(12).iloc[::-1].iterrows():
        avg_hr = f"{r['avg_heartrate']:.0f}" if pd.notna(r["avg_heartrate"]) else "-"
        lines.append(
            f"| {r['date'].date()} | {r['distance_mi']:.1f} | "
            f"{r['moving_time_min']:.0f} | {format_pace(r['pace_min_per_mi'])} | "
            f"{r['elevation_gain_ft']:.0f} | {avg_hr} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    runs = load_runs_2026()
    weekly = weekly_summary(runs)
    mp_long = mp_miles_by_long_run(runs, load_splits())
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(runs, weekly, mp_long))
    print(f"Wrote {REPORT_PATH.relative_to(Path.cwd())}")
    print(f"{len(runs)} runs, {runs['distance_mi'].sum():.1f} mi in 2026")


if __name__ == "__main__":
    main()
