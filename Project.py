from __future__ import annotations

from dashboard.data import (
    build_rankings,
    build_snapshot,
    describe_dataset,
    get_latest_period,
    load_art_coverage_data,
)


def main() -> None:
    dataset = load_art_coverage_data()
    latest_year = get_latest_period(dataset)
    latest_snapshot = build_snapshot(dataset, year=latest_year, region="All")
    summary = describe_dataset(dataset)
    leaders = build_rankings(latest_snapshot, ascending=False)[:5]

    print("WHO ART Coverage Project Summary")
    print(f"Records: {summary['records']}")
    print(f"Countries: {summary['countries']}")
    print(f"Years covered: {summary['start_year']} to {summary['end_year']}")
    print(f"Latest average coverage ({latest_year}): {summary['latest_average']}%")
    print(f"Last updated: {summary['updated_on']}")
    print()
    print("Top countries in the latest snapshot:")
    for leader in leaders:
        print(f"- {leader['location']}: {leader['coverage']} ({leader['interval']})")


if __name__ == "__main__":
    main()
