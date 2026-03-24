from dashboard.data import (
    build_snapshot,
    build_year_over_year_changes,
    get_latest_period,
    load_art_coverage_data,
)
from app import app


def test_dataset_loads_and_has_expected_columns():
    dataset = load_art_coverage_data()

    assert not dataset.empty
    assert {"Location", "Period", "coverage_value", "ParentLocation"}.issubset(dataset.columns)
    assert dataset["Period"].min() == 2000
    assert dataset["Period"].max() == 2023


def test_latest_snapshot_contains_country_records():
    dataset = load_art_coverage_data()
    latest_year = get_latest_period(dataset)
    snapshot = build_snapshot(dataset, year=latest_year, region="All")

    assert not snapshot.empty
    assert snapshot["Location"].nunique() >= 100
    assert snapshot["Location"].nunique() == len(snapshot)


def test_year_over_year_changes_are_built_for_latest_view():
    dataset = load_art_coverage_data()
    latest_year = get_latest_period(dataset)
    change_frame = build_year_over_year_changes(dataset, year=latest_year, region="All")

    assert not change_frame.empty
    assert "change" in change_frame.columns


def test_homepage_renders():
    client = app.test_client()
    response = client.get("/")

    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "WHO ART Coverage Dashboard" in page
    assert "Coverage map" in page


def test_health_endpoint_returns_ok():
    client = app.test_client()
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
