from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data.csv"

NUMERIC_COLUMNS = [
    "Period",
    "FactValueNumeric",
    "FactValueNumericLow",
    "FactValueNumericHigh",
]


@lru_cache(maxsize=1)
def load_art_coverage_data(csv_path: str = str(DATA_PATH)) -> pd.DataFrame:
    dataset = pd.read_csv(csv_path)

    for column in NUMERIC_COLUMNS:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")

    dataset["DateModified"] = pd.to_datetime(dataset["DateModified"], errors="coerce")
    dataset["ParentLocation"] = dataset["ParentLocation"].fillna("Unspecified")
    dataset["Location"] = dataset["Location"].fillna("Unknown")
    dataset["SpatialDimValueCode"] = dataset["SpatialDimValueCode"].fillna("")
    dataset["is_latest_year"] = dataset["IsLatestYear"].astype(str).str.lower().eq("true")

    midpoint = dataset[["FactValueNumericLow", "FactValueNumericHigh"]].mean(axis=1, skipna=True)
    dataset["coverage_value"] = dataset["FactValueNumeric"].fillna(midpoint)
    dataset["coverage_low"] = dataset["FactValueNumericLow"].fillna(dataset["coverage_value"])
    dataset["coverage_high"] = dataset["FactValueNumericHigh"].fillna(dataset["coverage_value"])
    dataset["uncertainty_width"] = (dataset["coverage_high"] - dataset["coverage_low"]).clip(lower=0)
    dataset["uses_midpoint_estimate"] = dataset["FactValueNumeric"].isna() & midpoint.notna()

    dataset = dataset[dataset["Location type"].fillna("Country").eq("Country")].copy()
    dataset = dataset.dropna(subset=["Period", "coverage_value", "Location"])
    dataset["Period"] = dataset["Period"].astype(int)
    dataset["coverage_value"] = dataset["coverage_value"].round(1)
    dataset["coverage_low"] = dataset["coverage_low"].round(1)
    dataset["coverage_high"] = dataset["coverage_high"].round(1)

    return dataset.sort_values(["Period", "ParentLocation", "Location"]).reset_index(drop=True)


def get_latest_period(dataset: pd.DataFrame) -> int:
    return int(dataset["Period"].max())


def get_filter_options(dataset: pd.DataFrame) -> dict[str, object]:
    years = sorted(dataset["Period"].unique().tolist())
    regions = ["All", *sorted(dataset["ParentLocation"].dropna().unique().tolist())]
    previous_year_lookup = {year: years[index - 1] if index > 0 else None for index, year in enumerate(years)}

    return {
        "years": years,
        "regions": regions,
        "default_year": years[-1],
        "previous_year_lookup": previous_year_lookup,
    }


def build_snapshot(dataset: pd.DataFrame, year: int, region: str) -> pd.DataFrame:
    snapshot = dataset[dataset["Period"] == year].copy()
    if region != "All":
        snapshot = snapshot[snapshot["ParentLocation"] == region].copy()

    return snapshot.sort_values("coverage_value", ascending=False).reset_index(drop=True)


def build_year_over_year_changes(dataset: pd.DataFrame, year: int, region: str) -> pd.DataFrame:
    years = sorted(dataset["Period"].unique().tolist())
    if year not in years:
        return pd.DataFrame()

    year_index = years.index(year)
    if year_index == 0:
        return pd.DataFrame()

    current = build_snapshot(dataset, year=year, region=region)[["Location", "ParentLocation", "coverage_value"]]
    previous = build_snapshot(dataset, year=years[year_index - 1], region=region)[["Location", "coverage_value"]]

    merged = current.merge(previous, on="Location", how="inner", suffixes=("_current", "_previous"))
    if merged.empty:
        return merged

    merged["change"] = (merged["coverage_value_current"] - merged["coverage_value_previous"]).round(1)
    return merged.sort_values("change", ascending=False).reset_index(drop=True)


def describe_dataset(dataset: pd.DataFrame) -> dict[str, object]:
    latest_rows = dataset[dataset["Period"] == get_latest_period(dataset)]
    source_values = latest_rows["DataSource"].dropna()
    return {
        "records": int(len(dataset)),
        "countries": int(dataset["Location"].nunique()),
        "regions": int(dataset["ParentLocation"].nunique()),
        "start_year": int(dataset["Period"].min()),
        "end_year": int(dataset["Period"].max()),
        "latest_average": round(float(latest_rows["coverage_value"].mean()), 1),
        "updated_on": dataset["DateModified"].max().strftime("%B %d, %Y"),
        "data_source": source_values.iloc[0] if not source_values.empty else "WHO",
        "midpoint_estimates": int(dataset["uses_midpoint_estimate"].sum()),
    }


def build_metric_cards(
    snapshot: pd.DataFrame,
    year_over_year: pd.DataFrame,
    prior_year: int | None,
) -> list[dict[str, str]]:
    if snapshot.empty:
        return []

    average_coverage = snapshot["coverage_value"].mean()
    median_coverage = snapshot["coverage_value"].median()
    countries_above_target = int((snapshot["coverage_value"] >= 90).sum())

    if prior_year is not None and not year_over_year.empty:
        change_value = year_over_year["change"].mean()
        change_label = f"Avg change vs {prior_year}"
        change_display = f"{change_value:+.1f} pts"
        change_detail = "Comparable countries only"
    else:
        spread = snapshot["coverage_value"].max() - snapshot["coverage_value"].min()
        change_label = "Coverage spread"
        change_display = f"{spread:.1f} pts"
        change_detail = "Difference between highest and lowest country"

    return [
        {
            "label": "Countries in view",
            "value": f"{snapshot['Location'].nunique()}",
            "detail": "WHO country estimates in the selected slice",
        },
        {
            "label": "Average coverage",
            "value": f"{average_coverage:.1f}%",
            "detail": "Mean ART coverage for the selected year",
        },
        {
            "label": "Median coverage",
            "value": f"{median_coverage:.1f}%",
            "detail": "Middle country after sorting by coverage",
        },
        {
            "label": change_label,
            "value": change_display,
            "detail": (
                change_detail
                if countries_above_target == 0
                else f"{countries_above_target} countries are at or above 90%"
            ),
        },
    ]


def summarize_regions(dataset: pd.DataFrame, year: int) -> list[dict[str, str]]:
    regional = (
        dataset[dataset["Period"] == year]
        .groupby("ParentLocation", as_index=False)
        .agg(
            average_coverage=("coverage_value", "mean"),
            countries=("Location", "nunique"),
        )
        .sort_values("average_coverage", ascending=False)
    )

    return [
        {
            "region": row.ParentLocation,
            "average": f"{row.average_coverage:.1f}%",
            "countries": f"{int(row.countries)} countries",
        }
        for row in regional.itertuples()
    ]


def build_rankings(snapshot: pd.DataFrame, ascending: bool) -> list[dict[str, str]]:
    ranked = snapshot.sort_values("coverage_value", ascending=ascending).head(8)
    return [
        {
            "location": row.Location,
            "region": row.ParentLocation,
            "coverage": f"{row.coverage_value:.1f}%",
            "interval": f"{row.coverage_low:.1f}-{row.coverage_high:.1f}%",
        }
        for row in ranked.itertuples()
    ]


def get_time_series(dataset: pd.DataFrame) -> pd.DataFrame:
    regional = (
        dataset.groupby(["Period", "ParentLocation"], as_index=False)
        .agg(
            average_coverage=("coverage_value", "mean"),
            countries=("Location", "nunique"),
        )
        .sort_values(["ParentLocation", "Period"])
    )

    global_average = (
        dataset.groupby("Period", as_index=False)
        .agg(
            average_coverage=("coverage_value", "mean"),
            countries=("Location", "nunique"),
        )
        .assign(ParentLocation="Global average")
    )

    return pd.concat([regional, global_average], ignore_index=True)


def build_insights(
    snapshot: pd.DataFrame,
    year_over_year: pd.DataFrame,
    dataset: pd.DataFrame,
    year: int,
    region: str,
    prior_year: int | None,
) -> list[str]:
    if snapshot.empty:
        return ["No records were available for the selected year and region."]

    insights: list[str] = []
    leader = snapshot.iloc[0]
    trailer = snapshot.iloc[-1]
    region_average = snapshot["coverage_value"].mean()
    median_coverage = snapshot["coverage_value"].median()

    if region == "All":
        leading_region = (
            dataset[dataset["Period"] == year]
            .groupby("ParentLocation")["coverage_value"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
            .iloc[0]
        )
        insights.append(
            f"{leader['Location']} leads the global view at {leader['coverage_value']:.1f}% "
            f"in {year}, while {trailer['Location']} is lowest at {trailer['coverage_value']:.1f}%."
        )
        insights.append(
            f"{leading_region['ParentLocation']} posts the strongest regional average "
            f"in {year} at {leading_region['coverage_value']:.1f}%."
        )
    else:
        insights.append(
            f"{leader['Location']} sets the pace in {region} at {leader['coverage_value']:.1f}% "
            f"in {year}, and the regional mean sits at {region_average:.1f}%."
        )
        insights.append(
            f"The selected region spans from {trailer['coverage_value']:.1f}% "
            f"to {leader['coverage_value']:.1f}%, with a median country at {median_coverage:.1f}%."
        )

    if prior_year is not None and not year_over_year.empty:
        best_change = year_over_year.iloc[0]
        insights.append(
            f"{best_change['Location']} shows the largest comparable gain versus {prior_year}: "
            f"{best_change['change']:+.1f} percentage points."
        )
    else:
        midpoint_count = int(snapshot["uses_midpoint_estimate"].sum())
        insights.append(
            f"{midpoint_count} countries in the current view rely on midpoint estimates "
            f"because WHO published uncertainty bounds without a central point value."
        )

    return insights
