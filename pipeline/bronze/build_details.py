"""Backfill per-activity detail (splits) and streams for runs.

This is separate from the main pipeline because it makes one Strava read call
per activity and is bound by the read rate limit (100 / 15 min). It is:

- **Incremental**: activity detail never changes once recorded, so already-
  fetched activity_ids are skipped. Only new runs cost a call after the first
  backfill.
- **Budgeted**: stops after READ_BUDGET calls (or on a 429) and saves progress,
  so a large first backfill just needs a few reruns ~15 min apart.

Splits are pulled for every run; streams (per-second) only for "quality"
sessions — treadmill tempos and long runs — where mile splits are too coarse.
"""

import pandas as pd

from pipeline.config import BRONZE_DIR, READ_BUDGET
from pipeline.extract.strava import RateLimitError, StravaReader

LONG_RUN_M = 10 * 1609.344  # 10 miles in meters (matches the analysis threshold)

SPLITS_PATH = BRONZE_DIR / "splits.parquet"
STREAMS_PATH = BRONZE_DIR / "streams.parquet"


def _fetched_ids(path) -> set[int]:
    if not path.exists():
        return set()
    return set(pd.read_parquet(path, columns=["activity_id"])["activity_id"].unique())


def _append(path, new_rows: list[dict]) -> None:
    if not new_rows:
        return
    df = pd.DataFrame(new_rows)
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _splits_rows(activity_id: int, detail: dict) -> list[dict]:
    rows = []
    for i, s in enumerate(detail.get("splits_standard") or [], start=1):
        rows.append(
            {
                "activity_id": activity_id,
                "split": i,
                "distance_m": s.get("distance"),
                "elapsed_time_s": s.get("elapsed_time"),
                "moving_time_s": s.get("moving_time"),
                "elevation_difference_m": s.get("elevation_difference"),
                "average_speed_mps": s.get("average_speed"),
                "average_heartrate": s.get("average_heartrate"),
            }
        )
    return rows


def _stream_rows(activity_id: int, streams: dict) -> list[dict]:
    def series(key):
        return (streams.get(key) or {}).get("data") or []

    time, distance = series("time"), series("distance")
    hr, vel, alt = series("heartrate"), series("velocity_smooth"), series("altitude")
    rows = []
    for i in range(len(time)):
        rows.append(
            {
                "activity_id": activity_id,
                "time_s": time[i],
                "distance_m": distance[i] if i < len(distance) else None,
                "heartrate": hr[i] if i < len(hr) else None,
                "velocity_smooth_mps": vel[i] if i < len(vel) else None,
                "altitude_m": alt[i] if i < len(alt) else None,
            }
        )
    return rows


def build_details() -> None:
    activities = pd.read_parquet(BRONZE_DIR / "activities.parquet")
    runs = activities[activities["type"] == "Run"]

    split_targets = [int(i) for i in runs["id"] if int(i) not in _fetched_ids(SPLITS_PATH)]
    quality = runs[(runs["trainer"]) | (runs["distance"] >= LONG_RUN_M)]
    stream_targets = [
        int(i) for i in quality["id"] if int(i) not in _fetched_ids(STREAMS_PATH)
    ]

    print(f"splits to fetch: {len(split_targets)} | streams to fetch: {len(stream_targets)}")
    if not split_targets and not stream_targets:
        print("nothing to fetch — all runs already backfilled")
        return

    reader = StravaReader()
    new_splits, new_streams = [], []
    stopped = False

    try:
        for activity_id in split_targets:
            if reader.window_used >= READ_BUDGET:
                stopped = True
                break
            new_splits.extend(_splits_rows(activity_id, reader.get_activity_detail(activity_id)))

        if not stopped:
            for activity_id in stream_targets:
                if reader.window_used >= READ_BUDGET:
                    stopped = True
                    break
                new_streams.extend(
                    _stream_rows(activity_id, reader.get_activity_streams(activity_id))
                )
    except RateLimitError:
        stopped = True
        print("hit Strava rate limit — saving progress")
    finally:
        _append(SPLITS_PATH, new_splits)
        _append(STREAMS_PATH, new_streams)

    print(
        f"fetched {len(new_splits)} split rows, {len(new_streams)} stream rows "
        f"(read usage {reader.window_used}/100 this window)"
    )
    if stopped:
        print("Budget/limit reached — rerun in ~15 min to continue the backfill.")


if __name__ == "__main__":
    build_details()
