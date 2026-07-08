from pipeline.bronze.build_bronze import build_bronze
from pipeline.gold.build_gold import build_gold
from pipeline.silver.build_silver import build_silver


def main() -> None:
    bronze_activities, bronze_gear = build_bronze()
    print(f"bronze activities: {len(bronze_activities)} rows")
    print(f"bronze gear: {len(bronze_gear)} rows")

    silver_activities, silver_gear = build_silver()
    print(f"silver activities: {len(silver_activities)} rows")
    print(f"silver gear: {len(silver_gear)} rows")

    for name, table in build_gold().items():
        print(f"gold {name}: {len(table)} rows")


if __name__ == "__main__":
    main()
