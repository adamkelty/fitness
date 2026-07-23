"""Gold star-schema facts built from the Apple Health silver tables.

Grain of each fact:
- fact_run          one row per (deduped) run, all sources/years
- fact_run_split    one row per run-mile
- fact_sleep_night  one row per night
- fact_daily_health one row per local day

All facts carry date_key (yyyymmdd int) joining to the shared dim_date.
"""

import pandas as pd

from pipeline.config import GOLD_DIR, SILVER_DIR


def _date_key(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates).dt.strftime("%Y%m%d").astype(int)


def build_dim_date(*date_series: pd.Series) -> pd.DataFrame:
    all_dates = pd.concat([pd.to_datetime(s) for s in date_series])
    dates = pd.date_range(all_dates.min(), all_dates.max(), freq="D")
    df = pd.DataFrame({"date": dates})
    df["date_key"] = df["date"].dt.strftime("%Y%m%d").astype(int)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.month_name()
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.day_name()
    df["week_of_year"] = df["date"].dt.isocalendar().week
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = df["date"].dt.dayofweek >= 5
    return df[
        ["date_key", "date", "year", "month", "month_name", "day",
         "day_of_week", "week_of_year", "quarter", "is_weekend"]
    ]


def build_fact_run(runs: pd.DataFrame) -> pd.DataFrame:
    fact = runs.copy()
    fact["date_key"] = _date_key(fact["local_date"])
    return fact[
        ["workout_id", "date_key", "source_name", "indoor", "n_segments",
         "distance_mi", "duration_min", "pace_min_per_mi", "avg_heartrate",
         "max_heartrate", "avg_power_w", "avg_gct_ms", "avg_vert_osc_cm",
         "avg_stride_m", "active_kcal", "steps", "elevation_gain_ft",
         "weather_temp_f", "weather_humidity_pct"]
    ]


def build_fact_sleep(nights: pd.DataFrame) -> pd.DataFrame:
    fact = nights.copy()
    fact["date_key"] = _date_key(fact["night_of"])
    return fact.drop(columns=["night_of"])


def build_fact_daily(daily: pd.DataFrame) -> pd.DataFrame:
    fact = daily.copy()
    fact["date_key"] = _date_key(fact["day"])
    return fact.drop(columns=["day"])


def build_health_gold() -> dict[str, pd.DataFrame]:
    runs = pd.read_parquet(SILVER_DIR / "ah_runs.parquet")
    splits = pd.read_parquet(SILVER_DIR / "ah_run_splits.parquet")
    nights = pd.read_parquet(SILVER_DIR / "ah_sleep_nights.parquet")
    daily = pd.read_parquet(SILVER_DIR / "ah_daily.parquet")

    # dim_date must span every fact table, including the Strava-based
    # fact_activity (which reaches back before Apple Health data starts).
    date_inputs = [runs["local_date"], nights["night_of"], daily["day"]]
    strava_activities = SILVER_DIR / "activities.parquet"
    if strava_activities.exists():
        strava_dates = pd.read_parquet(strava_activities, columns=["start_date"])
        date_inputs.append(strava_dates["start_date"].dt.tz_localize(None).dt.date)

    tables = {
        "dim_date": build_dim_date(*date_inputs),
        "fact_run": build_fact_run(runs),
        "fact_run_split": splits,  # already at run-mile grain w/ workout_id FK
        "fact_sleep_night": build_fact_sleep(nights),
        "fact_daily_health": build_fact_daily(daily),
    }

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(GOLD_DIR / f"{name}.parquet", index=False)
    return tables


if __name__ == "__main__":
    for name, table in build_health_gold().items():
        print(f"gold {name}: {len(table)} rows")
