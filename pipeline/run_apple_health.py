"""Refresh the Apple Health side of the pipeline after a new export.

Usage: drop a fresh export.zip in data/apple_health/, unzip it, then run
    uv run python -m pipeline.run_apple_health
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
