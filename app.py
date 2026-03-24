from __future__ import annotations

import io
import os

from flask import Flask, Response, jsonify, render_template, request

from dashboard.charts import chart_to_html
from dashboard.data import (
    DATA_PATH,
    build_insights,
    build_metric_cards,
    build_rankings,
    build_snapshot,
    build_year_over_year_changes,
    describe_dataset,
    get_filter_options,
    get_latest_period,
    get_time_series,
    load_art_coverage_data,
    summarize_regions,
)
from dashboard.visuals import (
    build_change_chart,
    build_coverage_map,
    build_region_trends_chart,
    build_top_locations_chart,
)


def create_app() -> Flask:
    app = Flask(__name__)

    def resolve_filters() -> tuple[object, dict[str, object], int, str]:
        dataset = load_art_coverage_data()
        filter_options = get_filter_options(dataset)

        year = request.args.get("year", type=int) or filter_options["default_year"]
        if year not in filter_options["years"]:
            year = filter_options["default_year"]

        region = request.args.get("region", default="All", type=str) or "All"
        if region not in filter_options["regions"]:
            region = "All"

        return dataset, filter_options, year, region

    @app.route("/")
    def index() -> str:
        dataset, filter_options, year, region = resolve_filters()
        snapshot = build_snapshot(dataset, year=year, region=region)
        prior_year = filter_options["previous_year_lookup"].get(year)
        year_over_year = build_year_over_year_changes(dataset, year=year, region=region)

        context = {
            "dataset_summary": describe_dataset(dataset),
            "filter_options": filter_options,
            "selected_year": year,
            "selected_region": region,
            "metrics": build_metric_cards(snapshot, year_over_year, prior_year),
            "insights": build_insights(snapshot, year_over_year, dataset, year, region, prior_year),
            "region_summary": summarize_regions(dataset, year=year),
            "top_locations": build_rankings(snapshot, ascending=False),
            "bottom_locations": build_rankings(snapshot, ascending=True),
            "map_chart": chart_to_html(build_coverage_map(snapshot, year=year, region=region)),
            "leaders_chart": chart_to_html(build_top_locations_chart(snapshot, year=year, region=region)),
            "trend_chart": chart_to_html(build_region_trends_chart(get_time_series(dataset), region=region)),
            "change_chart": chart_to_html(
                build_change_chart(
                    year_over_year,
                    year=year,
                    region=region,
                    prior_year=prior_year,
                )
            ),
            "latest_period": get_latest_period(dataset),
            "data_path": DATA_PATH.name,
        }

        return render_template("index.html", **context)

    @app.route("/api/summary")
    def summary_api() -> tuple[object, int]:
        dataset, filter_options, year, region = resolve_filters()
        snapshot = build_snapshot(dataset, year=year, region=region)
        prior_year = filter_options["previous_year_lookup"].get(year)
        year_over_year = build_year_over_year_changes(dataset, year=year, region=region)

        payload = {
            "status": "ok",
            "year": year,
            "region": region,
            "filters": {
                "years": filter_options["years"],
                "regions": filter_options["regions"],
            },
            "summary": describe_dataset(dataset),
            "metrics": build_metric_cards(snapshot, year_over_year, prior_year),
            "insights": build_insights(snapshot, year_over_year, dataset, year, region, prior_year),
            "top_locations": build_rankings(snapshot, ascending=False),
            "bottom_locations": build_rankings(snapshot, ascending=True),
        }
        return jsonify(payload), 200

    @app.route("/download/current-view.csv")
    def download_current_view() -> Response:
        dataset, _, year, region = resolve_filters()
        snapshot = build_snapshot(dataset, year=year, region=region)

        export_columns = [
            "Location",
            "ParentLocation",
            "Period",
            "coverage_value",
            "coverage_low",
            "coverage_high",
            "uses_midpoint_estimate",
            "DateModified",
        ]
        export_frame = snapshot[export_columns].rename(
            columns={
                "ParentLocation": "Region",
                "coverage_value": "Coverage",
                "coverage_low": "CoverageLow",
                "coverage_high": "CoverageHigh",
                "uses_midpoint_estimate": "UsesMidpointEstimate",
            }
        )

        buffer = io.StringIO()
        export_frame.to_csv(buffer, index=False)
        filename = f"art-coverage-{region.lower().replace(' ', '-')}-{year}.csv"

        return Response(
            buffer.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.route("/health")
    def health() -> tuple[object, int]:
        dataset = load_art_coverage_data()
        return (
            jsonify(
                {
                    "status": "ok",
                    "records": int(len(dataset)),
                    "latest_period": int(get_latest_period(dataset)),
                }
            ),
            200,
        )

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
