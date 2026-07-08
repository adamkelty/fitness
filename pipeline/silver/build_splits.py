import pandas as pd

from pipeline.bronze.build_details import SPLITS_PATH
from pipeline.config import SILVER_DIR

MILES_PER_KM = 0.621371
FEET_PER_METER = 3.28084


def build_splits() -> pd.DataFrame:
    splits = pd.read_parquet(SPLITS_PATH)

    df = pd.DataFrame(
        {
            "activity_id": splits["activity_id"],
            "mile": splits["split"],
            "distance_mi": splits["distance_m"] / 1609.344,
            "moving_time_min": splits["moving_time_s"] / 60,
            "avg_heartrate": splits["average_heartrate"],
            "elev_change_ft": splits["elevation_difference_m"] * FEET_PER_METER,
        }
    )
    df["pace_min_per_mi"] = df["moving_time_min"] / df["distance_mi"]

    # Drop the tiny partial final split each run ends on (< a third of a mile);
    # its pace is a meaningless artifact of the leftover distance.
    df = df[df["distance_mi"] >= 0.33]

    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SILVER_DIR / "splits.parquet", index=False)
    return df


if __name__ == "__main__":
    df = build_splits()
    print(f"silver splits: {len(df)} mile-rows across {df['activity_id'].nunique()} runs")
