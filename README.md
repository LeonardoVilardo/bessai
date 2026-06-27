# BESSAi

BESSAi is a small engineering demo for evaluating whether a solar user should add a battery. The current MVP uses a deterministic hourly battery simulation, a simple battery-size optimizer, annual historical irradiance economics, and an ML forecast-error uncertainty demo.

## Setup

Python:

```bash
python3 -m pip install -r requirements.txt
```

Web dashboard:

```bash
cd apps/web
npm install
```

Next.js 16 requires Node.js `>=20.9.0`.

## Run The API

```bash
uvicorn apps.api.main:app --reload
```

If `uvicorn` is not on your PATH, use:

```bash
python3 -m uvicorn apps.api.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Run The Web Dashboard

In a second terminal:

```bash
cd apps/web
npm run dev
```

Then open:

```text
http://localhost:3000
```

The dashboard calls `http://127.0.0.1:8000` by default. To use another backend URL:

```bash
NEXT_PUBLIC_API_URL=http://127.0.0.1:8001 npm run dev
```

## API Routes

- `GET /health`
- `POST /simulate`
- `POST /optimize`
- `POST /prefetch-annual-economics`

POST routes default to the Brasilia demo scenario if no request body is provided. The dashboard currently offers two focused demo locations: Brasilia and West of Bahia. The prefetch route warms the local NASA POWER cache for the selected location so optimisation can reuse the same historical years.

## Local Demos

```bash
python3 packages/engine/run_brasilia_demo.py
python3 packages/ml/run_forecast_error_demo.py
```

## Current Assumptions

- Forecast source: Open-Meteo, with a simple fallback profile.
- Annual economics: NASA POWER hourly historical irradiance across a recent 10-year sample when available, with P50/P90 annual savings. Falls back to a single historical year, then representative 24 h annualisation if needed.
- Export credit: configurable value per exported kWh, defaulting to `0`. The app does not hardcode country-specific net-metering policy.
- Time step: hourly.
- Battery sizes: `0, 5, 10, 13.5, 15, 20, 30 kWh`.
- ML forecast-error model: trains on matched Open-Meteo previous-day forecast runs and Open-Meteo historical weather data when available, with a clearly labeled synthetic fallback. In the MVP it reports forecast uncertainty only; dispatch uses the raw Open-Meteo forecast.
