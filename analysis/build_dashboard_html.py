"""Generate a self-contained, interactive HTML training dashboard.

Reads the Apple Health gold facts, computes every series, and writes a single
HTML file with the data embedded as JSON and charts drawn as inline SVG (no
external assets — publishes cleanly as an Artifact). Refresh it by re-running
after a data refresh:
    uv run python -m pipeline.run_apple_health
    uv run python -m analysis.build_dashboard_html
then re-publish the file to the same Artifact URL.
"""

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from analysis.charts import PLAN_WEEKLY, RACE_WEIGHT_TARGET, compute_decoupling
from analysis.marathon_training import (
    GOAL_PACE_MIN_PER_MI,
    GOAL_TIME,
    LONG_RUN_MI,
    MP_TARGET_MIN_PER_MI,
    RACE_DATE,
    format_pace,
    load,
    quality_miles,
    weekly_summary,
)

OUT = Path(__file__).resolve().parent.parent / "docs" / "dashboard.html"


def _series(daily, col):
    d = daily.dropna(subset=[col]).sort_values("date")
    d = d[d["date"].dt.year == 2026]
    roll = d.set_index("date")[col].rolling("14D").mean()
    return [
        {"d": r["date"].date().isoformat(), "v": round(float(r[col]), 1),
         "r": round(float(roll.loc[r["date"]]), 1)}
        for _, r in d.iterrows()
    ]


def compute_data() -> dict:
    d = load()
    runs = d["runs_2026"]
    weekly = weekly_summary(runs)
    labeled = quality_miles(d["splits"], d["runs"])
    labeled26 = labeled[labeled["date"].dt.year == 2026]
    daily = d["daily"]
    sleep = d["sleep"]
    today = runs["date"].max().date()

    wk = (labeled26.set_index("date").groupby([pd.Grouper(freq="W-SUN"), "label"])
          .size().unstack(fill_value=0))

    def roll_mean(df, col, days):
        w = df[(df["date"].dt.date > today - timedelta(days=days)) & (df["date"].dt.date <= today)]
        return float(w[col].mean())

    lr = runs[runs["distance_mi"] >= LONG_RUN_MI]
    longest = runs.loc[runs["distance_mi"].idxmax()]
    mp = labeled26[labeled26["label"] == "MP"]
    vo2 = daily.dropna(subset=["vo2max"])
    wt = daily.dropna(subset=["body_mass_lb"])
    wt26 = wt[wt["date"].dt.year == 2026]
    jan_w = wt26[wt26["date"].dt.month == 1]["body_mass_lb"].mean()
    recent_w = wt26.tail(14)["body_mass_lb"].mean()
    s7 = sleep[sleep["date"].dt.date > today - timedelta(days=7)]

    last_full = weekly.iloc[-2] if len(weekly) >= 2 else weekly.iloc[-1]
    weeks_out = (pd.Timestamp(RACE_DATE).date() - today).days / 7
    mp_hr = mp["avg_heartrate"].mean()
    vo2_latest = float(vo2.iloc[-1]["vo2max"]) if len(vo2) else None

    kpis = [
        {"label": "Weeks to race", "value": f"{weeks_out:.0f}", "sub": f"Sat {RACE_DATE}"},
        {"label": "Longest run", "value": f"{longest['distance_mi']:.0f} mi",
         "sub": f"{format_pace(longest['pace_min_per_mi'])}/mi · {longest['date'].date()}"},
        {"label": "Last full week", "value": f"{last_full['total_mi']:.0f} mi",
         "sub": f"peak {weekly['total_mi'].max():.0f} mi"},
        {"label": "VO₂max", "value": f"{vo2_latest:.0f}", "sub": "↑ from 51 in spring", "tone": "good"},
        {"label": "Resting HR", "value": f"{roll_mean(daily,'resting_hr',7):.0f}", "sub": "bpm · 7-day", "tone": "good"},
        {"label": "HRV", "value": f"{roll_mean(daily,'hrv_ms',7):.0f}",
         "sub": "ms · drifting down", "tone": "warn"},
        {"label": "Weight", "value": f"{recent_w:.0f} lb", "sub": f"−{jan_w-recent_w:.0f} since Jan"},
        {"label": "Sleep", "value": f"{s7['asleep_min'].mean()/60:.1f} h", "sub": "7-day · weak spot", "tone": "warn"},
    ]

    outlook = [
        ("On track for sub-3:30, with margin.",
         f"20-miler banked, {(runs['distance_mi']>=16).sum()} runs of 16+ mi, VO₂max {vo2_latest:.0f}, "
         f"and MP work at {format_pace(MP_TARGET_MIN_PER_MI)}/mi — 15s/mi faster than "
         f"{format_pace(GOAL_PACE_MIN_PER_MI)} goal pace.", "good"),
        ("But that MP pace isn't free yet.",
         f"Those miles run at ~{mp_hr:.0f} bpm — threshold, not aerobic MP. 7:45 for a full 26.2 is "
         f"unproven; sub-3:30 at ~8:00/mi is the realistic, sustainable play.", "warn"),
        ("Watch items.",
         "Volume is moderate (30–45 mpw, right-sized for this goal); fueling untested at race distance; "
         "sleep ~6 h; HRV drifting off the June peak. None are red flags for finishing.", "warn"),
        ("Bottom line.",
         "For a debut where the goal is to finish strong, the data says you're in good shape and ahead of "
         "where sub-3:30 requires. The volume ramp is the work left, not the fitness.", "good"),
    ]

    return {
        "meta": {
            "race_date": RACE_DATE, "weeks_out": round(weeks_out, 1),
            "goal_time": GOAL_TIME, "goal_pace": format_pace(GOAL_PACE_MIN_PER_MI),
            "today": today.isoformat(), "generated": date.today().isoformat(),
        },
        "kpis": kpis,
        "outlook": outlook,
        "weekly_actual": [{"d": r["date"].date().isoformat(), "mi": round(float(r["total_mi"]), 1)}
                          for _, r in weekly.iterrows()],
        "weekly_plan": [{"d": k, "mi": v} for k, v in PLAN_WEEKLY.items()],
        "long_runs": [{"d": r["date"].date().isoformat(), "mi": round(float(r["distance_mi"]), 1)}
                      for _, r in lr.iterrows()],
        "quality": [{"d": i.date().isoformat(), "mp": int(row.get("MP", 0)), "tempo": int(row.get("tempo", 0))}
                    for i, row in wk.iterrows()],
        "decoupling": [{"d": r["date"].date().isoformat(), "pct": round(float(r["decoupling_pct"]), 1)}
                       for _, r in compute_decoupling(labeled26).iterrows()],
        "pace_hr": [{"pace": round(float(r["pace_min_per_mi"]), 2), "hr": round(float(r["avg_heartrate"])),
                     "m": int(r["date"].month), "d": r["date"].date().isoformat()}
                    for _, r in runs[(~runs["indoor"]) & runs["avg_heartrate"].notna()
                                     & (runs["distance_mi"] >= 3) & (runs["pace_min_per_mi"] < 11)].iterrows()],
        "weight": _series(daily, "body_mass_lb"),
        "rhr": _series(daily, "resting_hr"),
        "hrv": _series(daily, "hrv_ms"),
        "weight_target": RACE_WEIGHT_TARGET,
        "long_peak": 21,
    }


def build_html() -> Path:
    data = compute_data()
    template = (Path(__file__).resolve().parent / "dashboard_template.html").read_text()
    html = template.replace("/*__DATA__*/", json.dumps(data))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    return OUT


if __name__ == "__main__":
    print(f"Wrote {build_html()}")
