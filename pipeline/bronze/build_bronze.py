from datetime import datetime, timezone

import pandas as pd

from pipeline.config import BRONZE_DIR
from pipeline.extract.strava import get_activities, get_gear


def build_bronze() -> tuple[pd.DataFrame, pd.DataFrame]:
    activities = get_activities()
    activities_df = pd.json_normalize(activities)
    activities_df["ingested_at"] = datetime.now(timezone.utc)

    gear_ids = sorted(activities_df.get("gear_id", pd.Series(dtype=str)).dropna().unique())
    gear = [get_gear(gear_id) for gear_id in gear_ids]
    gear_df = pd.json_normalize(gear)
    gear_df["ingested_at"] = datetime.now(timezone.utc)

    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    activities_df.to_parquet(BRONZE_DIR / "activities.parquet", index=False)
    gear_df.to_parquet(BRONZE_DIR / "gear.parquet", index=False)

    return activities_df, gear_df


if __name__ == "__main__":
    activities_df, gear_df = build_bronze()
    print(f"bronze activities: {len(activities_df)} rows")
    print(f"bronze gear: {len(gear_df)} rows")
