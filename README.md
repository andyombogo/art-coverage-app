# WHO ART Coverage Dashboard

An intuitive Flask dashboard for exploring WHO antiretroviral therapy (ART) coverage estimates by country, region, and year.

## Why this repo is stronger now

- Replaced hard-coded local file paths with a portable data pipeline that reads the checked-in WHO CSV.
- Removed Spark and GeoPandas from the web request path so the app is lighter, faster, and easier to deploy.
- Upgraded the UI from a static image page to an interactive dashboard with filters, KPI cards, a choropleth map, trend lines, movers, and ranked tables.
- Added a health endpoint, JSON summary endpoint, downloadable filtered CSV export, Render blueprint, cleaner deployment commands, and test coverage for core app behavior.

## Dashboard features

- Year and region filters for focused exploration.
- Country-level coverage map using ISO-3 country codes from the WHO dataset.
- Regional trend tracking from 2000 to 2023.
- Year-over-year change view for countries with comparable records.
- Tables for highest- and lowest-coverage geographies.
- One-click export of the filtered country view as CSV.
- JSON summary endpoint for lightweight sharing or integration.
- Methodology panel that explains how missing point estimates are handled.

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000/`.

## Deploy on Render

This repo includes `render.yaml` so you can deploy it as a Render Blueprint.

Manual Render settings if you need them:

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`
- Health check path: `/health`
- Python version: pinned via `.python-version`

Recommended deployment path:

1. In Render, click `New +` -> `Blueprint`.
2. Select `andyombogo/art-coverage-app`.
3. Confirm the imported settings from `render.yaml`.
4. Deploy and wait for the health check on `/health` to pass.

Because this repo already has CI, the Blueprint is set to deploy only after GitHub checks pass.

For a full deploy checklist, see `docs/render-deploy.md`.

## Project structure

- `app.py`: Flask entrypoint and routes
- `dashboard/data.py`: data loading and dashboard summaries
- `dashboard/visuals.py`: Plotly figure builders
- `templates/index.html`: dashboard layout
- `static/styles.css`: app styling

## Data note

The checked-in source file spans 2000 through 2023 and contains 4,656 WHO rows. After filtering to country observations with usable coverage values, the dashboard works with 3,407 records across 146 countries. Some rows publish only low and high intervals without a central estimate; the dashboard uses the midpoint of that interval for comparative views while still exposing the full range in tables and map hovers.

## Portfolio tips

- Add your final Render URL to the GitHub repo website field.
- Capture screenshots from the `#coverage-map`, `#regional-trends`, and `#top-performers` sections.
- Pin this repo on your profile once the live link is up.




