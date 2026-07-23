"""Refresh the Apple Health side of the pipeline after a new export.

Usage: drop a fresh export.zip at data/apple_health/export.zip, then run
    uv run python -m pipeline.run_apple_health

The raw export (~2.7GB unzipped: XML + GPX routes) is auto-extracted, parsed
into bronze parquet (~60MB), then deleted - only the compact parquet sticks
around. Nothing needs the raw files after this runs, so there's no need to
keep re-accumulating multi-GB exports locally.
"""

from pipeline.bronze.build_apple_health import build_apple_health
from pipeline.gold.build_health_gold import build_health_gold
from pipeline.silver.build_apple_health import build_apple_health_silver


def main() -> None:
    build_apple_health()
    build_apple_health_silver()
    for name, table in build_health_gold().items():
        print(f"gold {name}: {len(table)} rows")


if __name__ == "__main__":
    main()
