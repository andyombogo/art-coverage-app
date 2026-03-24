# Render Deploy Guide

This repo is ready to deploy on Render as a Python web service.

## What is already configured

- `render.yaml` defines the Blueprint service.
- `Procfile` and the Blueprint both start the app with `gunicorn app:app`.
- `.python-version` pins Python 3.13 for more predictable builds.
- `/health` provides a health check endpoint for Render.
- GitHub Actions validates linting and tests before deploys when checks are required.

## Deploy steps

1. Open Render and click `New +`.
2. Choose `Blueprint`.
3. Select the GitHub repo `andyombogo/art-coverage-app`.
4. Review the imported service settings.
5. Click `Apply`.
6. Wait for the health check on `/health` to pass.
7. Open the generated `.onrender.com` URL and verify the dashboard loads.

## After deployment

- Add the live URL to the GitHub repo website field.
- Add repository topics such as `flask`, `plotly`, `python`, `hiv`, `public-health`, and `render`.
- Capture screenshots from the app for the README and GitHub project card.
- If the free instance spin-down delay feels too slow for demos, upgrade the Render plan.

## Handy routes

- `/` dashboard
- `/health` health check
- `/api/summary?year=2023&region=Africa` JSON summary
- `/download/current-view.csv?year=2023&region=Africa` filtered CSV export
