from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


REGION_ORDER = [
    "Africa",
    "Americas",
    "Eastern Mediterranean",
    "Europe",
    "South-East Asia",
    "Western Pacific",
    "Global average",
]

REGION_COLORS = {
    "Africa": "#0b6e69",
    "Americas": "#1d8a99",
    "Eastern Mediterranean": "#c8843f",
    "Europe": "#3d5a80",
    "South-East Asia": "#b56576",
    "Western Pacific": "#6a9a1f",
    "Global average": "#152238",
}

MAP_SCALE = [
    [0.0, "#f8f7f1"],
    [0.2, "#dbe8d8"],
    [0.4, "#a9d4c2"],
    [0.6, "#62b6cb"],
    [0.8, "#1d7ea7"],
    [1.0, "#0b3558"],
]


def _base_layout() -> dict[str, object]:
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(247,249,248,0.85)",
        "font": {"family": "Source Sans 3, Arial, sans-serif", "color": "#12384a", "size": 14},
        "margin": {"l": 24, "r": 24, "t": 24, "b": 24},
    }


def _apply_layout(figure, height: int = 420):
    figure.update_layout(height=height, **_base_layout())
    return figure


def _empty_figure(message: str):
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 16, "color": "#48626f"},
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return _apply_layout(figure, height=360)


def build_coverage_map(snapshot: pd.DataFrame, year: int, region: str):
    if snapshot.empty:
        return _empty_figure("No country coverage values were available for this view.")

    figure = px.choropleth(
        snapshot,
        locations="SpatialDimValueCode",
        locationmode="ISO-3",
        color="coverage_value",
        hover_name="Location",
        hover_data={
            "ParentLocation": True,
            "coverage_value": ":.1f",
            "coverage_low": ":.1f",
            "coverage_high": ":.1f",
            "SpatialDimValueCode": False,
        },
        color_continuous_scale=MAP_SCALE,
        range_color=(0, 100),
        labels={
            "coverage_value": "Coverage (%)",
            "ParentLocation": "Region",
            "coverage_low": "Low",
            "coverage_high": "High",
        },
    )
    figure.update_geos(
        projection_type="natural earth",
        showframe=False,
        showcoastlines=False,
        bgcolor="rgba(0,0,0,0)",
        showcountries=True,
        countrycolor="rgba(18, 56, 74, 0.15)",
    )
    figure.update_layout(
        coloraxis_colorbar={"title": "Coverage (%)"},
        annotations=[
            {
                "text": f"Selected view: {region} - {year}",
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1.08,
                "showarrow": False,
                "font": {"size": 14, "color": "#48626f"},
            }
        ],
    )
    return _apply_layout(figure, height=500)


def build_top_locations_chart(snapshot: pd.DataFrame, year: int, region: str):
    if snapshot.empty:
        return _empty_figure("No country rankings are available for this view.")

    leaders = snapshot.head(12).sort_values("coverage_value", ascending=True)
    figure = px.bar(
        leaders,
        x="coverage_value",
        y="Location",
        color="coverage_value",
        orientation="h",
        text="coverage_value",
        color_continuous_scale=MAP_SCALE,
        labels={"coverage_value": "Coverage (%)", "Location": ""},
    )
    figure.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate="%{y}<br>%{x:.1f}%<extra></extra>",
    )
    figure.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        annotations=[
            {
                "text": f"Highest-performing countries in {region} for {year}",
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1.08,
                "showarrow": False,
                "font": {"size": 14, "color": "#48626f"},
            }
        ],
    )
    figure.update_xaxes(range=[0, 100], ticksuffix="%")
    return _apply_layout(figure, height=460)


def build_region_trends_chart(time_series: pd.DataFrame, region: str):
    if time_series.empty:
        return _empty_figure("No time-series data is available.")

    if region == "All":
        chart_data = time_series[time_series["ParentLocation"].isin(REGION_ORDER)].copy()
    else:
        chart_data = time_series[
            time_series["ParentLocation"].isin([region, "Global average"])
        ].copy()

    figure = px.line(
        chart_data,
        x="Period",
        y="average_coverage",
        color="ParentLocation",
        markers=True,
        color_discrete_map=REGION_COLORS,
        category_orders={"ParentLocation": REGION_ORDER},
        labels={
            "Period": "Year",
            "average_coverage": "Average coverage (%)",
            "ParentLocation": "Series",
        },
    )
    figure.update_traces(
        hovertemplate="%{fullData.name}<br>%{x}: %{y:.1f}%<extra></extra>",
        line={"width": 3},
        marker={"size": 7},
    )
    figure.update_layout(legend_title_text="", hovermode="x unified")
    figure.update_yaxes(range=[0, 100], ticksuffix="%")
    return _apply_layout(figure, height=430)


def build_change_chart(
    year_over_year: pd.DataFrame,
    year: int,
    region: str,
    prior_year: int | None,
):
    if prior_year is None or year_over_year.empty:
        return _empty_figure("Choose a year after the first available observation to compare changes.")

    biggest_gains = year_over_year.nlargest(6, "change")
    biggest_declines = year_over_year.nsmallest(6, "change")
    movers = (
        pd.concat([biggest_gains, biggest_declines], ignore_index=True)
        .drop_duplicates(subset="Location")
        .sort_values("change", ascending=True)
    )
    movers["direction"] = movers["change"].apply(lambda value: "Gain" if value >= 0 else "Decline")

    figure = px.bar(
        movers,
        x="change",
        y="Location",
        orientation="h",
        color="direction",
        color_discrete_map={"Gain": "#0b6e69", "Decline": "#c46a3f"},
        text="change",
        labels={"change": "Percentage-point change", "Location": ""},
    )
    figure.update_traces(
        texttemplate="%{text:+.1f}",
        textposition="outside",
        hovertemplate="%{y}<br>%{x:+.1f} pts<extra></extra>",
    )
    figure.add_vline(x=0, line_dash="dash", line_color="rgba(18, 56, 74, 0.35)")
    figure.update_layout(
        legend_title_text="",
        annotations=[
            {
                "text": f"Comparable movers in {region}, {prior_year} to {year}",
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1.08,
                "showarrow": False,
                "font": {"size": 14, "color": "#48626f"},
            }
        ],
    )
    return _apply_layout(figure, height=430)
