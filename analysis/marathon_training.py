"""Marathon training analysis for the 2026 Marquette race (goal: sub-3:30).

Runs, splits, sleep, and daily recovery metrics come from the Apple Health
gold facts (the watch's sensor-fused data — accurate through the GPS dropouts
that corrupt Strava's API). Pace is reported in min/mi throughout. Renders a
living markdown report to docs/marathon-training.md.
"""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from pipeline.config import GOLD_DIR

MARATHON_MI = 26.2188
LONG_RUN_MI = 10  # threshold for what counts as a long run
RACE_DATE = "2026-09-05"  # Marquette, Saturday of Labor Day weekend

# Goal this year: finish, sub-3:30 (~8:00/mi). This is a debut marathon -
# "say I did it" is the point; chasing a time goal is next year's build. The
# fitness data suggests capable of faster, which just means 3:30 has margin.
GOAL_TIME = "3:30"
GOAL_PACE_MIN_PER_MI = (3 * 60 + 30) / MARATHON_MI  # ~8:01/mi

# Marathon-pace work inside long runs is trained at ~7:45 (faster than the 3:30
# race pace, which gives buffer). Band is tight to keep MP honest vs threshold.
MP_TARGET_MIN_PER_MI = 7 + 45 / 60
MP_LOW, MP_HIGH = 7 + 35 / 60, 7 + 55 / 60

# Anything faster than the MP band inside a run is tempo/threshold work.
TEMPO_CEILING = MP_LOW

REPORT_PATH = Path(__file__).resolve().parent.parent / "docs" / "marathon-training.md"


def format_pace(min_per_mi: float) -> str:
    if pd.isna(min_per_mi):
        return "-"
    minutes = int(min_per_mi)
    seconds = round((min_per_mi - minutes) * 60)
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes}:{seconds:02d}"


def load() -> dict[str, pd.DataFrame]:
    dim_date = pd.read_parquet(GOLD_DIR / "dim_date.parquet")
    runs = (
        pd.read_parquet(GOLD_DIR / "fact_run.parquet")
        .merge(dim_date, on="date_key")
        .sort_values("date")
    )
    return {
        "runs": runs,
        "runs_2026": runs[runs["year"] == 2026].copy(),
        "splits": pd.read_parquet(GOLD_DIR / "fact_run_split.parquet"),
        "sleep": pd.read_parquet(GOLD_DIR / "fact_sleep_night.parquet").merge(
            dim_date, on="date_key"
        ),
        "daily": pd.read_parquet(GOLD_DIR / "fact_daily_health.parquet").merge(
            dim_date, on="date_key"
        ),
    }


def weekly_summary(runs: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        runs.set_index("date")
        .resample("W-SUN")
        .agg(
            runs=("distance_mi", "size"),
            total_mi=("distance_mi", "sum"),
            longest_mi=("distance_mi", "max"),
            total_time_min=("duration_min", "sum"),
            avg_hr=("avg_heartrate", "mean"),
        )
    )
    weekly = weekly[weekly["runs"] > 0]
    weekly["avg_pace_min_per_mi"] = weekly["total_time_min"] / weekly["total_mi"]
    return weekly.reset_index()


def quality_miles(splits: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Label each split mile: MP (in the marathon-pace band), tempo (faster
    than the tempo ceiling), or easy."""
    s = splits.merge(
        runs[["workout_id", "date", "distance_mi", "indoor"]],
        on="workout_id",
        suffixes=("", "_run"),
    )
    s["label"] = "easy"
    s.loc[s["pace_min_per_mi"] < TEMPO_CEILING, "label"] = "tempo"
    s.loc[s["pace_min_per_mi"].between(MP_LOW, MP_HIGH), "label"] = "MP"
    return s


def rolling_mean(daily: pd.DataFrame, col: str, days: int, end: date) -> float:
    window = daily[
        (daily["date"].dt.date > end - timedelta(days=days))
        & (daily["date"].dt.date <= end)
    ]
    return window[col].mean()


def render(d: dict[str, pd.DataFrame]) -> str:
    runs = d["runs_2026"]
    weekly = weekly_summary(runs)
    labeled = quality_miles(d["splits"], d["runs"])
    labeled_2026 = labeled[labeled["date"].dt.year == 2026]
    daily = d["daily"]
    sleep = d["sleep"]
    today = runs["date"].max().date()
    weeks_out = (pd.Timestamp(RACE_DATE).date() - today).days / 7

    total_mi = runs["distance_mi"].sum()
    longest = runs.loc[runs["distance_mi"].idxmax()]

    L = []
    L.append("# Marathon Training — Marquette 2026")
    L.append("")
    L.append(f"_Last updated: {date.today().isoformat()} · source: Apple Health_")
    L.append("")
    L.append("## Goal")
    L.append("")
    L.append(f"- **Race:** Marquette, MI — Saturday {RACE_DATE} (~{weeks_out:.0f} weeks out)")
    L.append(
        f"- **Goal:** finish, sub-**{GOAL_TIME}** ({format_pace(GOAL_PACE_MIN_PER_MI)}/mi). "
        "Debut marathon — the point is completing it; time goal is next year."
    )
    L.append(
        f"- **MP work is trained at {format_pace(MP_TARGET_MIN_PER_MI)}/mi** — "
        f"faster than {GOAL_TIME} race pace, so goal pace has built-in buffer."
    )
    L.append("")

    L.append("## 2026 so far")
    L.append("")
    L.append(f"- **Runs:** {len(runs)} ({int(runs['indoor'].sum())} treadmill)")
    L.append(f"- **Total distance:** {total_mi:.1f} mi")
    L.append(
        f"- **Longest run:** {longest['distance_mi']:.1f} mi on "
        f"{longest['date'].date()} @ {format_pace(longest['pace_min_per_mi'])}/mi"
    )
    mp_total = (labeled_2026["label"] == "MP").sum()
    tempo_total = (labeled_2026["label"] == "tempo").sum()
    L.append(f"- **Quality miles:** {mp_total} at marathon pace, {tempo_total} at tempo or faster")
    L.append("")

    L.append("## Weekly progression")
    L.append("")
    L.append("| Week ending | Runs | Miles | Longest | Avg pace | Avg HR | MP mi | Tempo mi |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    wk_labels = (
        labeled_2026.set_index("date")
        .groupby([pd.Grouper(freq="W-SUN"), "label"])
        .size()
        .unstack(fill_value=0)
    )
    for _, w in weekly.iterrows():
        avg_hr = f"{w['avg_hr']:.0f}" if pd.notna(w["avg_hr"]) else "-"
        wl = wk_labels.loc[w["date"]] if w["date"] in wk_labels.index else {}
        L.append(
            f"| {w['date'].date()} | {int(w['runs'])} | {w['total_mi']:.1f} | "
            f"{w['longest_mi']:.1f} | {format_pace(w['avg_pace_min_per_mi'])} | {avg_hr} | "
            f"{wl.get('MP', 0)} | {wl.get('tempo', 0)} |"
        )
    L.append("")

    L.append(f"## Long runs (≥ {LONG_RUN_MI} mi)")
    L.append("")
    L.append("| Date | Miles | Pace | Elev (ft) | MP miles | MP splits | Temp °F |")
    L.append("|---|---:|---:|---:|---:|---|---:|")
    long_runs = runs[runs["distance_mi"] >= LONG_RUN_MI]
    for _, r in long_runs.iloc[::-1].iterrows():
        s = labeled[labeled["workout_id"] == r["workout_id"]]
        mp = s[s["label"] == "MP"]
        paces = ", ".join(format_pace(p) for p in mp["pace_min_per_mi"]) if len(mp) else "—"
        temp = f"{r['weather_temp_f']:.0f}" if pd.notna(r["weather_temp_f"]) else "-"
        L.append(
            f"| {r['date'].date()} | {r['distance_mi']:.1f} | "
            f"{format_pace(r['pace_min_per_mi'])} | {r['elevation_gain_ft']:.0f} | "
            f"{len(mp)} | {paces} | {temp} |"
        )
    L.append("")

    L.append("## Treadmill / tempo sessions (last 8)")
    L.append("")
    L.append("| Date | Miles | Avg pace | Fastest mile | Avg HR | Max HR |")
    L.append("|---|---:|---:|---:|---:|---:|")
    treadmill = runs[runs["indoor"]].tail(8)
    for _, r in treadmill.iloc[::-1].iterrows():
        s = labeled[labeled["workout_id"] == r["workout_id"]]
        fastest = s["pace_min_per_mi"].min() if len(s) else float("nan")
        L.append(
            f"| {r['date'].date()} | {r['distance_mi']:.1f} | "
            f"{format_pace(r['pace_min_per_mi'])} | {format_pace(fastest)} | "
            f"{r['avg_heartrate']:.0f} | {r['max_heartrate']:.0f} |"
        )
    L.append("")

    L.append("## Recovery & readiness")
    L.append("")
    s7 = sleep[sleep["date"].dt.date > today - timedelta(days=7)]
    s28 = sleep[sleep["date"].dt.date > today - timedelta(days=28)]
    if len(s28):
        L.append(
            f"- **Sleep:** {s7['asleep_min'].mean() / 60:.1f} h/night last 7d "
            f"(28d avg {s28['asleep_min'].mean() / 60:.1f} h) — "
            f"deep {s7['deep_min'].mean():.0f} min, REM {s7['rem_min'].mean():.0f} min"
        )
    rhr7, rhr28 = rolling_mean(daily, "resting_hr", 7, today), rolling_mean(daily, "resting_hr", 28, today)
    hrv7, hrv28 = rolling_mean(daily, "hrv_ms", 7, today), rolling_mean(daily, "hrv_ms", 28, today)
    L.append(
        f"- **Resting HR:** {rhr7:.0f} bpm (7d) vs {rhr28:.0f} (28d) — "
        f"{'↓ good' if rhr7 <= rhr28 else '↑ watch for fatigue'}"
    )
    L.append(
        f"- **HRV:** {hrv7:.0f} ms (7d) vs {hrv28:.0f} (28d) — "
        f"{'↑ good' if hrv7 >= hrv28 else '↓ watch for fatigue'}"
    )
    vo2 = daily.dropna(subset=["vo2max"])
    if len(vo2):
        latest = vo2.iloc[-1]
        L.append(
            f"- **VO2Max:** {latest['vo2max']:.1f} ({latest['date'].date()})"
        )
    hrr = daily.dropna(subset=["hr_recovery_1min"])
    if len(hrr):
        hrr30 = hrr[hrr["date"].dt.date > today - timedelta(days=30)]
        if len(hrr30):
            L.append(f"- **HR recovery (1 min):** {hrr30['hr_recovery_1min'].mean():.0f} bpm avg last 30d")
    wt = daily.dropna(subset=["body_mass_lb"])
    if len(wt):
        wt26 = wt[wt["date"].dt.year == 2026]
        jan = wt26[wt26["date"].dt.month == 1]["body_mass_lb"].mean()
        recent = wt26.tail(14)["body_mass_lb"].mean()
        L.append(
            f"- **Weight:** {recent:.0f} lb (down from {jan:.0f} in Jan, "
            f"−{jan - recent:.0f} lb) — lighter = free pace; trending toward ~168 target"
        )
    L.append("")

    L.append("## Race-day outlook (honest read)")
    L.append("")
    lr16 = (runs["distance_mi"] >= 16).sum()
    mp = quality_miles(d["splits"], d["runs"])
    mp = mp[(mp["date"].dt.year == 2026) & (mp["label"] == "MP")]
    mp_hr = mp["avg_heartrate"].mean()
    vo2_latest = vo2.iloc[-1]["vo2max"] if len(vo2) else float("nan")
    for line in [
        f"**On track for sub-{GOAL_TIME}, with margin.** The 20-miler is banked, "
        f"{lr16} runs of 16+ mi, VO2Max {vo2_latest:.0f}, and MP work at "
        f"{format_pace(MP_TARGET_MIN_PER_MI)}/mi — 15s/mi faster than "
        f"{format_pace(GOAL_PACE_MIN_PER_MI)} goal pace, so the target has cushion.",
        f"**But that MP pace isn't 'free' yet:** those miles run at ~{mp_hr:.0f} bpm "
        "(threshold, not aerobic MP) — so 7:45 for a full 26.2 is unproven. "
        f"Sub-{GOAL_TIME} at ~8:00/mi sits at an easier effort and is the realistic play.",
        "**Watch items:** volume is moderate (30–45 mpw — right-sized for this goal, "
        "not faster); fueling is untested at race distance; sleep ~6 h is the weak spot; "
        "HRV has drifted down slightly off the June peak. None are red flags for finishing.",
        "**Bottom line:** for a debut where the goal is to finish strong, the data says "
        "you're in good shape and ahead of where sub-3:30 requires.",
    ]:
        L.append(f"- {line}")
    L.append("")
    L.append("![training dashboard](charts/dashboard.png)")
    L.append("")

    L.append("## Recent runs")
    L.append("")
    L.append("| Date | Miles | Pace | Avg HR | Power | Elev (ft) | Where |")
    L.append("|---|---:|---:|---:|---:|---:|---|")
    for _, r in runs.tail(12).iloc[::-1].iterrows():
        pw = f"{r['avg_power_w']:.0f}W" if pd.notna(r["avg_power_w"]) else "-"
        elev = f"{r['elevation_gain_ft']:.0f}" if pd.notna(r["elevation_gain_ft"]) else "-"
        L.append(
            f"| {r['date'].date()} | {r['distance_mi']:.1f} | "
            f"{format_pace(r['pace_min_per_mi'])} | {r['avg_heartrate']:.0f} | {pw} | "
            f"{elev} | {'treadmill' if r['indoor'] else 'outdoor'} |"
        )
    L.append("")
    return "\n".join(L)


def main() -> None:
    d = load()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render(d))
    print(f"Wrote {REPORT_PATH.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
