"""Silver layer over the Apple Health bronze tables.

Produces:
- ah_runs.parquet         cleaned, deduped running workouts (all sources/years)
- ah_run_splits.parquet   per-mile splits reconstructed from the watch's
                          sensor-fused distance samples (accurate through GPS
                          dropouts), with pause time subtracted and HR/power
                          joined per mile
- ah_sleep_nights.parquet nightly sleep with stage minutes
- ah_daily.parquet        one row per local day of recovery/activity metrics
"""

import json

import numpy as np
import pandas as pd

from pipeline.bronze.build_apple_health import AH_DIR, RECORDS_DIR
from pipeline.config import LOCAL_TZ, SILVER_DIR

MILE_M = 1609.344

# When two runs overlap in time, keep the higher-priority source.
SOURCE_PRIORITY = ["Watch", "Nike Run Club", "Strava"]

# Consecutive segments of one physical run (a mid-run pause or GPS drop splits
# the watch recording) are stitched back together when the gap between them is
# under this many minutes. A doubles day (AM/PM runs) is hours apart, so it
# stays separate.
STITCH_GAP_MIN = 35

SUM_COLS = ["duration_min", "distance_mi", "active_kcal", "steps", "elevation_gain_ft"]
WMEAN_COLS = [
    "avg_heartrate", "avg_power_w", "avg_gct_ms", "avg_vert_osc_cm",
    "avg_stride_m", "avg_mets",
]
FIRST_COLS = ["source_name", "indoor", "local_date", "weather_temp_f", "weather_humidity_pct"]


def _priority(source: str) -> int:
    for i, key in enumerate(SOURCE_PRIORITY):
        if key.lower() in source.lower():
            return i
    return len(SOURCE_PRIORITY)


def _read_records(rtype: str) -> pd.DataFrame:
    df = pd.read_parquet(RECORDS_DIR / f"{rtype}.parquet")
    df["start"] = pd.to_datetime(df["start_date"], utc=True, format="%Y-%m-%d %H:%M:%S %z")
    df["end"] = pd.to_datetime(df["end_date"], utc=True, format="%Y-%m-%d %H:%M:%S %z")
    return df


def build_runs() -> pd.DataFrame:
    w = pd.read_parquet(AH_DIR / "workouts.parquet")
    runs = w[w["activity_type"] == "Running"].copy()

    runs["start"] = pd.to_datetime(runs["start_date"], utc=True, format="%Y-%m-%d %H:%M:%S %z")
    runs["end"] = pd.to_datetime(runs["end_date"], utc=True, format="%Y-%m-%d %H:%M:%S %z")
    runs["local_date"] = runs["start"].dt.tz_convert(LOCAL_TZ).dt.date

    numeric = {
        "duration_min": "duration_min",
        "DistanceWalkingRunning_sum": "distance_mi",
        "HeartRate_average": "avg_heartrate",
        "HeartRate_maximum": "max_heartrate",
        "RunningPower_average": "avg_power_w",
        "RunningGroundContactTime_average": "avg_gct_ms",
        "RunningVerticalOscillation_average": "avg_vert_osc_cm",
        "RunningStrideLength_average": "avg_stride_m",
        "ActiveEnergyBurned_sum": "active_kcal",
        "StepCount_sum": "steps",
        "avg_mets": "avg_mets",
    }
    for src, dst in numeric.items():
        runs[dst] = pd.to_numeric(runs.get(src), errors="coerce")

    runs["indoor"] = runs["indoor"] == "1"
    # "36569 cm" -> feet
    runs["elevation_gain_ft"] = (
        runs["elevation_ascended"].str.extract(r"([\d.]+)")[0].astype(float) / 30.48
    )
    runs["weather_temp_f"] = runs["weather_temp"].str.extract(r"([\d.]+)")[0].astype(float)
    runs["weather_humidity_pct"] = (
        runs["weather_humidity"].str.extract(r"([\d.]+)")[0].astype(float) / 100
    )
    runs["pace_min_per_mi"] = runs["duration_min"] / runs["distance_mi"]

    # Dedupe overlapping recordings of the same physical run across sources.
    runs = runs.sort_values(["start"]).reset_index(drop=True)
    runs["prio"] = runs["source_name"].map(_priority)
    keep = []
    for idx in runs.sort_values(["prio", "start"]).index:
        row = runs.loc[idx]
        overlaps = [
            k for k in keep
            if row["start"] < runs.loc[k, "end"] and row["end"] > runs.loc[k, "start"]
        ]
        if not overlaps:
            keep.append(idx)
    runs = runs.loc[sorted(keep)].reset_index(drop=True)

    out = _stitch_segments(runs)
    out.to_parquet(SILVER_DIR / "ah_runs.parquet", index=False)
    return out


def _wmean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def _stitch_segments(runs: pd.DataFrame) -> pd.DataFrame:
    """Merge consecutive segments of the same physical run into one row.

    Records the constituent workout ids and per-segment [start, end] spans as
    JSON so the split reconstruction can gather samples across the whole run
    and treat the inter-segment gaps as pauses.
    """
    runs = runs.sort_values("start").reset_index(drop=True)
    group_id, groups = 0, []
    prev = None
    for _, row in runs.iterrows():
        if prev is not None:
            gap_min = (row["start"] - prev["end"]).total_seconds() / 60
            same_run = (
                row["source_name"] == prev["source_name"]
                and row["local_date"] == prev["local_date"]
                and gap_min <= STITCH_GAP_MIN
            )
            if not same_run:
                group_id += 1
        groups.append(group_id)
        prev = row
    runs["group"] = groups

    merged = []
    for _, seg in runs.groupby("group"):
        seg = seg.sort_values("start")
        row = {
            "workout_id": int(seg["workout_id"].iloc[0]),
            "segment_ids": json.dumps([int(x) for x in seg["workout_id"]]),
            "segments": json.dumps(
                [[s.isoformat(), e.isoformat()] for s, e in zip(seg["start"], seg["end"])]
            ),
            "n_segments": len(seg),
            "start": seg["start"].min(),
            "end": seg["end"].max(),
            "max_heartrate": seg["max_heartrate"].max(),
        }
        for col in SUM_COLS:
            row[col] = seg[col].sum(min_count=1)
        for col in WMEAN_COLS:
            row[col] = _wmean(seg[col], seg["duration_min"])
        for col in FIRST_COLS:
            row[col] = seg[col].iloc[0]
        row["pace_min_per_mi"] = row["duration_min"] / row["distance_mi"]
        merged.append(row)

    cols = [
        "workout_id", "segment_ids", "segments", "n_segments", "source_name",
        "start", "end", "local_date", "indoor", "duration_min", "distance_mi",
        "pace_min_per_mi", "avg_heartrate", "max_heartrate", "avg_power_w",
        "avg_gct_ms", "avg_vert_osc_cm", "avg_stride_m", "active_kcal", "steps",
        "avg_mets", "elevation_gain_ft", "weather_temp_f", "weather_humidity_pct",
    ]
    return pd.DataFrame(merged)[cols].sort_values("start").reset_index(drop=True)


def _pause_intervals(events: pd.DataFrame, workout_id: int) -> list[tuple]:
    ev = events[
        (events["workout_id"] == workout_id)
        & (events["event_type"].isin(["Pause", "Resume"]))
    ].sort_values("date")
    pauses, pause_start = [], None
    for _, e in ev.iterrows():
        if e["event_type"] == "Pause":
            pause_start = e["date"]
        elif pause_start is not None:
            pauses.append((pause_start, e["date"]))
            pause_start = None
    return pauses


def _moving_time_s(t0, t1, pauses) -> float:
    total = (t1 - t0).total_seconds()
    for p0, p1 in pauses:
        overlap = (min(t1, p1) - max(t0, p0)).total_seconds()
        if overlap > 0:
            total -= overlap
    return total


def build_run_splits(runs: pd.DataFrame) -> pd.DataFrame:
    dist = _read_records("DistanceWalkingRunning")
    hr = _read_records("HeartRate")
    power = _read_records("RunningPower")
    events = pd.read_parquet(AH_DIR / "workout_events.parquet")
    events["date"] = pd.to_datetime(events["date"], utc=True, format="%Y-%m-%d %H:%M:%S %z")

    dist["value"] = dist["value"].astype(float)
    hr["value"] = hr["value"].astype(float)
    power["value"] = power["value"].astype(float)

    all_rows = []
    for _, run in runs.iterrows():
        segs = json.loads(run["segments"])

        # Build one distance track across the run's segments. Multiple sources
        # log distance for a run (watch GPS outdoors, GymKit belt on a
        # treadmill, phone), and the best source differs per segment - so pick
        # the largest-distance source *within each segment* and concatenate.
        seg_values, seg_times = [], []
        for seg_start, seg_end in segs:
            s0 = pd.Timestamp(seg_start).tz_convert("UTC")
            s1 = pd.Timestamp(seg_end).tz_convert("UTC")
            d = dist[(dist["end"] >= s0) & (dist["end"] <= s1)]
            if d.empty:
                continue
            best = d.groupby("source_name")["value"].sum().idxmax()
            d = d[d["source_name"] == best].sort_values("end")
            vals = d["value"].to_numpy()
            if d["unit"].iloc[0] == "km":
                vals = vals * 0.621371
            seg_values.append(vals)
            seg_times.append(d["end"].to_numpy())
        if not seg_values:
            continue
        values = np.concatenate(seg_values)
        times = np.concatenate(seg_times)
        cum_mi = values.cumsum()

        n_miles = int(cum_mi[-1])
        if n_miles == 0:
            continue
        boundaries = np.searchsorted(cum_mi, np.arange(1, n_miles + 1))
        boundary_times = pd.to_datetime(times[np.minimum(boundaries, len(times) - 1)], utc=True)

        # Pauses = each segment's own pause events, plus the gaps between
        # stitched segments (so a mile spanning the gap isn't counted as moving).
        pauses = []
        for seg_id in json.loads(run["segment_ids"]):
            pauses.extend(_pause_intervals(events, seg_id))
        for (_, prev_end), (next_start, _) in zip(segs, segs[1:]):
            pauses.append(
                (pd.Timestamp(prev_end).tz_convert("UTC"), pd.Timestamp(next_start).tz_convert("UTC"))
            )

        run_hr = hr[(hr["end"] >= run["start"]) & (hr["end"] <= run["end"])]
        run_pw = power[(power["end"] >= run["start"]) & (power["end"] <= run["end"])]

        prev_t = run["start"]
        for mile, t in enumerate(boundary_times, start=1):
            moving_s = _moving_time_s(prev_t, t, pauses)
            seg_hr = run_hr[(run_hr["end"] > prev_t) & (run_hr["end"] <= t)]["value"]
            seg_pw = run_pw[(run_pw["end"] > prev_t) & (run_pw["end"] <= t)]["value"]
            all_rows.append(
                {
                    "workout_id": run["workout_id"],
                    "mile": mile,
                    "moving_time_min": moving_s / 60,
                    "pace_min_per_mi": moving_s / 60,
                    "avg_heartrate": seg_hr.mean() if len(seg_hr) else None,
                    "avg_power_w": seg_pw.mean() if len(seg_pw) else None,
                }
            )
            prev_t = t

    splits = pd.DataFrame(all_rows)
    splits.to_parquet(SILVER_DIR / "ah_run_splits.parquet", index=False)
    return splits


def build_sleep_nights() -> pd.DataFrame:
    sleep = _read_records("SleepAnalysis")
    sleep["stage"] = sleep["value"].str.removeprefix("HKCategoryValueSleepAnalysis")
    sleep["minutes"] = (sleep["end"] - sleep["start"]).dt.total_seconds() / 60
    # A night belongs to the morning it ends on; shifting back 12h maps
    # evening-to-morning sessions onto one "night of" date.
    sleep["night_of"] = (
        sleep["end"].dt.tz_convert(LOCAL_TZ) - pd.Timedelta(hours=12)
    ).dt.date

    per = (
        sleep.pivot_table(
            index=["night_of", "source_name"],
            columns="stage",
            values="minutes",
            aggfunc="sum",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    stage_cols = [c for c in ["AsleepCore", "AsleepDeep", "AsleepREM", "AsleepUnspecified"] if c in per]
    per["asleep_min"] = per[stage_cols].sum(axis=1)
    per["has_stages"] = per.get("AsleepDeep", pd.Series(0, index=per.index)).fillna(0) > 0

    # One source per night: prefer stage-bearing (watch) over in-bed-only apps.
    per = per.sort_values(["night_of", "has_stages", "asleep_min"], ascending=[True, False, False])
    nights = per.drop_duplicates("night_of", keep="first").copy()

    nights = nights.rename(
        columns={
            "AsleepCore": "core_min",
            "AsleepDeep": "deep_min",
            "AsleepREM": "rem_min",
            "Awake": "awake_min",
            "InBed": "in_bed_min",
        }
    )
    keep = [
        c for c in [
            "night_of", "source_name", "asleep_min", "core_min", "deep_min",
            "rem_min", "awake_min", "in_bed_min",
        ] if c in nights
    ]
    nights = nights[keep].reset_index(drop=True)
    nights.to_parquet(SILVER_DIR / "ah_sleep_nights.parquet", index=False)
    return nights


def _daily(rtype: str, how: str) -> pd.Series:
    df = _read_records(rtype)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["day"] = df["start"].dt.tz_convert(LOCAL_TZ).dt.date
    if how == "sum_max_source":
        # Phone and watch both log; summing would double-count, so sum per
        # source and keep the larger (the primary recorder that day).
        per = df.groupby(["day", "source_name"])["value"].sum()
        return per.groupby("day").max()
    if how == "mean":
        return df.groupby("day")["value"].mean()
    return df.groupby("day")["value"].last()


def build_daily() -> pd.DataFrame:
    parts = {
        "steps": _daily("StepCount", "sum_max_source"),
        "active_kcal": _daily("ActiveEnergyBurned", "sum_max_source"),
        "exercise_min": _daily("AppleExerciseTime", "sum_max_source"),
        "resting_hr": _daily("RestingHeartRate", "mean"),
        "hrv_ms": _daily("HeartRateVariabilitySDNN", "mean"),
        "respiratory_rate": _daily("RespiratoryRate", "mean"),
        "spo2": _daily("OxygenSaturation", "mean"),
        "vo2max": _daily("VO2Max", "last"),
        "body_mass_lb": _daily("BodyMass", "last"),
        "wrist_temp_f": _daily("AppleSleepingWristTemperature", "mean"),
        "hr_recovery_1min": _daily("HeartRateRecoveryOneMinute", "mean"),
        "time_in_daylight_min": _daily("TimeInDaylight", "sum_max_source"),
    }
    daily = pd.DataFrame(parts).rename_axis("day").reset_index()
    daily.to_parquet(SILVER_DIR / "ah_daily.parquet", index=False)
    return daily


def build_apple_health_silver() -> None:
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    runs = build_runs()
    print(f"ah_runs: {len(runs)} rows")
    splits = build_run_splits(runs)
    print(f"ah_run_splits: {len(splits)} rows across {splits['workout_id'].nunique()} runs")
    nights = build_sleep_nights()
    print(f"ah_sleep_nights: {len(nights)} nights")
    daily = build_daily()
    print(f"ah_daily: {len(daily)} days")


if __name__ == "__main__":
    build_apple_health_silver()
