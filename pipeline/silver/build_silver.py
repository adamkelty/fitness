import pandas as pd

from pipeline.config import BRONZE_DIR, SILVER_DIR

ACTIVITY_COLUMNS = {
    "id": "activity_id",
    "name": "name",
    "type": "type",
    "start_date": "start_date",
    "distance": "distance_m",
    "moving_time": "moving_time_s",
    "elapsed_time": "elapsed_time_s",
    "total_elevation_gain": "elevation_gain_m",
    "average_speed": "avg_speed_mps",
    "max_speed": "max_speed_mps",
    "average_heartrate": "avg_heartrate",
    "max_heartrate": "max_heartrate",
    "suffer_score": "suffer_score",
    "kudos_count": "kudos_count",
    "achievement_count": "achievement_count",
    "commute": "commute",
    "trainer": "trainer",
    "gear_id": "gear_id",
}

GEAR_COLUMNS = {
    "id": "gear_id",
    "name": "name",
    "brand_name": "brand_name",
    "model_name": "model_name",
    "distance": "lifetime_distance_m",
}


def _select_and_rename(df: pd.DataFrame, columns: dict) -> pd.DataFrame:
    present = {source: target for source, target in columns.items() if source in df.columns}
    return df[list(present)].rename(columns=present)


def build_silver_activities(bronze_activities: pd.DataFrame) -> pd.DataFrame:
    df = _select_and_rename(bronze_activities, ACTIVITY_COLUMNS)

    df["start_date"] = pd.to_datetime(df["start_date"], utc=True)
    df["distance_km"] = df.pop("distance_m") / 1000
    df["moving_time_min"] = df.pop("moving_time_s") / 60
    df["elapsed_time_min"] = df.pop("elapsed_time_s") / 60
    df["avg_speed_kmh"] = df.pop("avg_speed_mps") * 3.6
    df["max_speed_kmh"] = df.pop("max_speed_mps") * 3.6

    df = df.drop_duplicates(subset="activity_id")
    return df


def build_silver_gear(bronze_gear: pd.DataFrame) -> pd.DataFrame:
    df = _select_and_rename(bronze_gear, GEAR_COLUMNS)
    if "lifetime_distance_m" in df.columns:
        df["lifetime_distance_km"] = df.pop("lifetime_distance_m") / 1000

    df = df.drop_duplicates(subset="gear_id")
    return df


def build_silver() -> tuple[pd.DataFrame, pd.DataFrame]:
    bronze_activities = pd.read_parquet(BRONZE_DIR / "activities.parquet")
    bronze_gear = pd.read_parquet(BRONZE_DIR / "gear.parquet")

    activities_df = build_silver_activities(bronze_activities)
    gear_df = build_silver_gear(bronze_gear)

    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    activities_df.to_parquet(SILVER_DIR / "activities.parquet", index=False)
    gear_df.to_parquet(SILVER_DIR / "gear.parquet", index=False)

    return activities_df, gear_df


if __name__ == "__main__":
    activities_df, gear_df = build_silver()
    print(f"silver activities: {len(activities_df)} rows")
    print(f"silver gear: {len(gear_df)} rows")
