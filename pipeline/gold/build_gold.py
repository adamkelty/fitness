import pandas as pd

from pipeline.config import GOLD_DIR, SILVER_DIR

# Strava gear ids are prefixed 'b' for bikes and 'g' for shoes/other gear.
GEAR_TYPE_PREFIXES = {"b": "Bike", "g": "Shoe"}


def build_dim_date(start_dates: pd.Series) -> pd.DataFrame:
    dates = pd.date_range(start_dates.min().date(), start_dates.max().date(), freq="D")
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
        [
            "date_key",
            "date",
            "year",
            "month",
            "month_name",
            "day",
            "day_of_week",
            "week_of_year",
            "quarter",
            "is_weekend",
        ]
    ]


def build_dim_gear(silver_gear: pd.DataFrame) -> pd.DataFrame:
    df = silver_gear.copy()
    df["gear_type"] = df["gear_id"].str[0].map(GEAR_TYPE_PREFIXES).fillna("Other")
    df.insert(0, "gear_key", range(1, len(df) + 1))
    return df[["gear_key", "gear_id", "name", "brand_name", "model_name", "gear_type"]]


def build_dim_activity_type(silver_activities: pd.DataFrame) -> pd.DataFrame:
    type_names = sorted(silver_activities["type"].dropna().unique())
    return pd.DataFrame({"type_key": range(1, len(type_names) + 1), "type_name": type_names})


def build_fact_activity(
    silver_activities: pd.DataFrame,
    dim_gear: pd.DataFrame,
    dim_activity_type: pd.DataFrame,
) -> pd.DataFrame:
    df = silver_activities.copy()
    df["date_key"] = df["start_date"].dt.strftime("%Y%m%d").astype(int)

    df = df.merge(dim_gear[["gear_id", "gear_key"]], on="gear_id", how="left")
    df = df.merge(
        dim_activity_type[["type_name", "type_key"]],
        left_on="type",
        right_on="type_name",
        how="left",
    )

    return df[
        [
            "activity_id",
            "date_key",
            "gear_key",
            "type_key",
            "distance_km",
            "moving_time_min",
            "elapsed_time_min",
            "elevation_gain_m",
            "avg_speed_kmh",
            "max_speed_kmh",
            "avg_heartrate",
            "max_heartrate",
            "calories",
            "suffer_score",
            "kudos_count",
            "achievement_count",
            "commute",
            "trainer",
        ]
    ]


def build_gold() -> dict[str, pd.DataFrame]:
    silver_activities = pd.read_parquet(SILVER_DIR / "activities.parquet")
    silver_gear = pd.read_parquet(SILVER_DIR / "gear.parquet")

    dim_date = build_dim_date(silver_activities["start_date"])
    dim_gear = build_dim_gear(silver_gear)
    dim_activity_type = build_dim_activity_type(silver_activities)
    fact_activity = build_fact_activity(silver_activities, dim_gear, dim_activity_type)

    tables = {
        "dim_date": dim_date,
        "dim_gear": dim_gear,
        "dim_activity_type": dim_activity_type,
        "fact_activity": fact_activity,
    }

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(GOLD_DIR / f"{name}.parquet", index=False)

    return tables


if __name__ == "__main__":
    for name, table in build_gold().items():
        print(f"gold {name}: {len(table)} rows")
