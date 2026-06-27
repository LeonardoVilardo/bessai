# BESSAi

BESSAi is a local engineering demo for evaluating whether a solar PV user should add a battery.

The MVP has three main parts:

- A deterministic battery simulation written in Python.
- A FastAPI backend that exposes the simulation through HTTP routes.
- A Next.js dashboard that calls the backend and shows the recommendation, dispatch chart, economics, and diagnostics.

The current dashboard is intentionally focused on two demo locations:

- Brasilia, Brazil: `-15.826016, -47.812539`
- West of Bahia, Brazil: `-13.792761, -46.104032`

## Data Sources

### Next-24h Dispatch Forecast

Tomorrow's dispatch chart uses the Open-Meteo Forecast API.

The backend requests hourly:

- `shortwave_radiation`
- `cloud_cover`
- `temperature_2m`

The current horizon is 24 hours. If Open-Meteo is unavailable, the app uses a simple deterministic fallback profile so the demo can still run.

### ML Forecast-Error Diagnostics

The ML model is used only to estimate forecast uncertainty. It does not change the dispatch forecast and it does not calculate financial results.

The current model is:

- Library: `scikit-learn`
- Model class: `GradientBoostingRegressor`
- Target: `actual_solar - forecast_solar`

When possible, training data is built from:

- Forecast rows: Open-Meteo Previous Runs API, using previous-day forecast runs.
- Actual/reanalysis rows: Open-Meteo Historical Weather API.

The matched training rows include:

- forecast irradiance
- cloud cover
- temperature
- hour of day
- day of year
- month
- actual irradiance
- forecast error

If real matched forecast-error rows are unavailable or not useful, the app falls back to clearly labeled synthetic demo training data.

### Annual Economics

Annual savings and payback use NASA POWER hourly historical irradiance when available.

The current default historical window is:

```text
2016-2025
```

For each battery size, the engine simulates each year independently, then reports:

- average annual savings
- P50 annual savings: median weather-year savings
- P90 annual savings: 10th percentile savings, a conservative weather-year estimate

The recommendation currently uses P50 annual savings.

Important: NASA POWER historical data is not a forecast. It is a historical weather-year sample used to make annual economics more realistic than multiplying one forecast day by 365. A production-grade or finance-grade version should support longer climatology windows, such as 20-30 years when available and appropriate.

Fallback order for annual economics:

1. NASA POWER hourly data for `2016-2025`.
2. Single NASA POWER historical year, currently `2024`.
3. Annualized estimate from the current 24-hour forecast window.

## Local Setup

Run these commands from the project root:

```bash
cd /Users/leovilardo/Desktop/Projects/BESSAí
```

Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Install frontend dependencies:

```bash
cd apps/web
npm install
cd ../..
```

Next.js 16 requires Node.js `>=20.9.0`. Check your Node version with:

```bash
node -v
```

## Run Locally

You need two terminal windows.

### Terminal 1: API Backend

From the project root:

```bash
python3 -m uvicorn apps.api.main:app --reload
```

What this means:

- `FastAPI` is the Python web framework used for the backend.
- `apps.api.main:app` points to the `app` object inside `apps/api/main.py`.
- `uvicorn` is the local web server that runs the FastAPI app.
- `--reload` restarts the server when backend files change.

Keep this terminal open. The API should be available at:

```text
http://127.0.0.1:8000
```

You can check the API docs at:

```text
http://127.0.0.1:8000/docs
```

### Terminal 2: Web Dashboard

From the project root:

```bash
cd apps/web
npm run dev
```

Then open:

```text
http://localhost:3000
```

The dashboard calls the API at `http://127.0.0.1:8000` by default.

## API Routes

- `GET /health`
  - Simple health check. Returns `{"status": "ok"}`.

- `POST /simulate`
  - Runs one deterministic simulation for one scenario and one battery size.

- `POST /optimize`
  - Runs the battery-size sweep for `0, 5, 10, 13.5, 15, 20, 30 kWh`.
  - Returns the recommended size, comparison table, dispatch series, assumptions, and diagnostics.

- `POST /prefetch-annual-economics`
  - Downloads or loads cached NASA POWER historical irradiance for the selected location.
  - This makes the next optimization faster for the same location.

POST routes default to the Brasilia scenario if no request body is provided.

## Local Scripts

Run the deterministic engine demo:

```bash
python3 packages/engine/run_brasilia_demo.py
```

Run the ML forecast-error demo:

```bash
python3 packages/ml/run_forecast_error_demo.py
```

## Current Modelling Assumptions

- Time step: hourly.
- PV output: irradiance factor times PV size times system efficiency.
- Default system efficiency: `0.8`.
- Battery round-trip efficiency: `0.9`.
- Battery charges from PV surplus.
- Battery discharges when load exceeds PV and during the peak tariff window.
- Peak tariff window: `18:00-21:00`.
- Export credit: configurable value per exported kWh, default `0`.
- Battery sizes: `0, 5, 10, 13.5, 15, 20, 30 kWh`.
- Annual economics: NASA POWER `2016-2025` when available, with single-year and representative-day fallbacks.
- ML: forecast-error uncertainty only; dispatch uses the raw Open-Meteo forecast.
- Financial results come from deterministic simulation, not ML.

## Troubleshooting

If `uvicorn` is not found, use:

```bash
python3 -m uvicorn apps.api.main:app --reload
```

If port `8000` is already in use:

```bash
python3 -m uvicorn apps.api.main:app --reload --port 8001
```

Then start the web dashboard with:

```bash
cd apps/web
NEXT_PUBLIC_API_URL=http://127.0.0.1:8001 npm run dev
```

If port `3000` is already in use:

```bash
cd apps/web
npm run dev -- --port 3001
```

Then open:

```text
http://localhost:3001
```

If the first optimization is slow, it is probably downloading NASA POWER historical data for the selected location. Downloaded data is cached under `outputs/cache/`, which is ignored by Git.

If live API calls fail, the app should still run with fallback data, but annual economics may fall back to a less realistic estimate.
