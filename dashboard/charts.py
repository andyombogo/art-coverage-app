from __future__ import annotations

from plotly.io import to_html


def chart_to_html(figure) -> str:
    return to_html(
        figure,
        include_plotlyjs=False,
        full_html=False,
        config={
            "displaylogo": False,
            "responsive": True,
            "modeBarButtonsToRemove": [
                "lasso2d",
                "select2d",
                "toggleSpikelines",
                "hoverClosestCartesian",
                "hoverCompareCartesian",
            ],
        },
    )
