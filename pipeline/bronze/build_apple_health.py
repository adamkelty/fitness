"""Parse the Apple Health export.xml into bronze parquet tables.

The export is ~2GB of XML, so this streams it with ElementTree.iterparse
(clearing elements as it goes) instead of loading the document. One pass
produces:

- workouts.parquet         one row per workout, with WorkoutStatistics pivoted
                           into columns and useful MetadataEntry keys promoted
- workout_events.parquet   pause/resume/segment events per workout
- records/{Type}.parquet   per-sample Record rows, one file per record type,
                           written in chunks so memory stays bounded

Apple Health is the source of truth here: its distance samples come from the
watch's sensor-fused estimate, so they stay accurate through the GPS dropouts
that corrupt Strava's API splits/streams.
"""

import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.config import BRONZE_DIR, DATA_DIR

APPLE_HEALTH_DIR = DATA_DIR / "apple_health"
EXPORT_ZIP = APPLE_HEALTH_DIR / "export.zip"
EXTRACTED_DIR = APPLE_HEALTH_DIR / "apple_health_export"
EXPORT_XML = EXTRACTED_DIR / "export.xml"
AH_DIR = BRONZE_DIR / "apple_health"
RECORDS_DIR = AH_DIR / "records"

FLUSH_ROWS = 200_000

# Record types worth landing (short name, without the HK...Identifier prefix).
KEEP_RECORD_TYPES = {
    # per-second workout samples
    "HeartRate",
    "DistanceWalkingRunning",
    "DistanceCycling",
    "RunningSpeed",
    "RunningPower",
    "RunningVerticalOscillation",
    "RunningStrideLength",
    "RunningGroundContactTime",
    "PhysicalEffort",
    # daily activity
    "StepCount",
    "ActiveEnergyBurned",
    "BasalEnergyBurned",
    "FlightsClimbed",
    "TimeInDaylight",
    "AppleExerciseTime",
    # recovery / readiness
    "SleepAnalysis",
    "HeartRateVariabilitySDNN",
    "RestingHeartRate",
    "RespiratoryRate",
    "OxygenSaturation",
    "AppleSleepingWristTemperature",
    "HeartRateRecoveryOneMinute",
    "WalkingHeartRateAverage",
    # fitness markers
    "VO2Max",
    # body composition
    "BodyMass",
    "BodyMassIndex",
    "BodyFatPercentage",
    "LeanBodyMass",
}

RECORD_FIELDS = ["type", "source_name", "unit", "start_date", "end_date", "value"]

WORKOUT_META_KEYS = {
    "HKIndoorWorkout": "indoor",
    "HKElevationAscended": "elevation_ascended",
    "HKWeatherTemperature": "weather_temp",
    "HKWeatherHumidity": "weather_humidity",
    "HKAverageMETs": "avg_mets",
    "HKTimeZone": "timezone",
}


def _short_type(hk_type: str) -> str:
    for prefix in (
        "HKQuantityTypeIdentifier",
        "HKCategoryTypeIdentifier",
        "HKWorkoutActivityType",
    ):
        if hk_type.startswith(prefix):
            return hk_type[len(prefix) :]
    return hk_type


class ChunkedWriter:
    """Appends dict-rows per record type, flushing to parquet in chunks."""

    def __init__(self) -> None:
        self.buffers: dict[str, list[dict]] = {}
        self.writers: dict[str, pq.ParquetWriter] = {}

    def add(self, rtype: str, row: dict) -> None:
        buf = self.buffers.setdefault(rtype, [])
        buf.append(row)
        if len(buf) >= FLUSH_ROWS:
            self._flush(rtype)

    def _flush(self, rtype: str) -> None:
        buf = self.buffers.get(rtype)
        if not buf:
            return
        table = pa.Table.from_pylist(buf, schema=pa.schema(
            [(f, pa.string()) for f in RECORD_FIELDS]
        ))
        if rtype not in self.writers:
            RECORDS_DIR.mkdir(parents=True, exist_ok=True)
            self.writers[rtype] = pq.ParquetWriter(
                RECORDS_DIR / f"{rtype}.parquet", table.schema
            )
        self.writers[rtype].write_table(table)
        buf.clear()

    def close(self) -> None:
        for rtype in list(self.buffers):
            self._flush(rtype)
        for writer in self.writers.values():
            writer.close()


def _parse_workout(elem: ET.Element, workout_id: int) -> tuple[dict, list[dict]]:
    row = {
        "workout_id": workout_id,
        "activity_type": _short_type(elem.get("workoutActivityType", "")),
        "duration_min": elem.get("duration"),
        "source_name": elem.get("sourceName"),
        "start_date": elem.get("startDate"),
        "end_date": elem.get("endDate"),
    }
    events = []
    for child in elem:
        if child.tag == "MetadataEntry":
            key = child.get("key")
            if key in WORKOUT_META_KEYS:
                row[WORKOUT_META_KEYS[key]] = child.get("value")
        elif child.tag == "WorkoutStatistics":
            stat = _short_type(child.get("type", ""))
            for agg in ("sum", "average", "minimum", "maximum"):
                value = child.get(agg)
                if value is not None:
                    row[f"{stat}_{agg}"] = value
            unit = child.get("unit")
            if unit:
                row[f"{stat}_unit"] = unit
        elif child.tag == "WorkoutEvent":
            events.append(
                {
                    "workout_id": workout_id,
                    "event_type": child.get("type", "").removeprefix(
                        "HKWorkoutEventType"
                    ),
                    "date": child.get("date"),
                    "duration_min": child.get("duration"),
                }
            )
    return row, events


def _ensure_extracted() -> None:
    """Unzip a freshly-dropped export.zip if it hasn't been extracted yet."""
    if EXPORT_XML.exists():
        return
    if not EXPORT_ZIP.exists():
        raise FileNotFoundError(
            f"No export found. Drop an Apple Health export at {EXPORT_ZIP}"
        )
    with zipfile.ZipFile(EXPORT_ZIP) as zf:
        zf.extractall(APPLE_HEALTH_DIR)


def _cleanup_raw_export() -> None:
    """Delete the raw export (~2.7GB extracted) now that bronze parquet has
    the data we need (~60MB) - nothing reads the raw XML/GPX after this."""
    if EXTRACTED_DIR.exists():
        shutil.rmtree(EXTRACTED_DIR)
    if EXPORT_ZIP.exists():
        EXPORT_ZIP.unlink()


def build_apple_health() -> None:
    _ensure_extracted()
    writer = ChunkedWriter()
    workouts: list[dict] = []
    workout_events: list[dict] = []
    workout_id = 0
    n_records = 0

    for event, elem in ET.iterparse(EXPORT_XML, events=("end",)):
        if elem.tag == "Record":
            rtype = _short_type(elem.get("type", ""))
            if rtype in KEEP_RECORD_TYPES:
                writer.add(
                    rtype,
                    {
                        "type": rtype,
                        "source_name": elem.get("sourceName"),
                        "unit": elem.get("unit"),
                        "start_date": elem.get("startDate"),
                        "end_date": elem.get("endDate"),
                        "value": elem.get("value"),
                    },
                )
                n_records += 1
        elif elem.tag == "Workout":
            workout_id += 1
            row, events = _parse_workout(elem, workout_id)
            workouts.append(row)
            workout_events.extend(events)
        else:
            continue
        elem.clear()  # keep memory flat while streaming

    writer.close()

    import pandas as pd

    AH_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(workouts).to_parquet(AH_DIR / "workouts.parquet", index=False)
    pd.DataFrame(workout_events).to_parquet(
        AH_DIR / "workout_events.parquet", index=False
    )
    print(f"workouts: {len(workouts)} | events: {len(workout_events)} | records kept: {n_records}")

    _cleanup_raw_export()
    print(f"cleaned up raw export ({EXPORT_ZIP.name}, {EXTRACTED_DIR.name}/) - bronze parquet retained")


if __name__ == "__main__":
    build_apple_health()
