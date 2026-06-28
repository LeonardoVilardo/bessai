# BESSAi Plan

## App Concept

BESSAi is a web app that helps solar PV users decide whether adding a battery energy storage system is financially and operationally worthwhile.

The app combines Open-Meteo short-term solar forecasts, NASA POWER historical irradiance, a simple scikit-learn forecast-error regression model, and a deterministic battery simulation to recommend a battery size, estimate payback, and show a practical daily dispatch strategy.

The long-term product can become general, but the current MVP demo is intentionally focused on two Brazilian locations: Brasilia and West of Bahia.

## Product Positioning

BESSAi should feel like an engineering decision-support tool, not a black-box consumer calculator.

The MVP should answer three questions clearly:

1. How large should my battery be?
2. How much money could it save each year?
3. How would the battery behave tomorrow under uncertain solar forecasts?

The strongest demo angle is transparency: the user can see the inputs, assumptions, dispatch chart, forecast uncertainty estimate, and recommendation explanation.

## Core System Loop

The MVP should use this loop as its default operating model, while keeping the implementation flexible if a simpler working version is cleaner.

1. Fetch the next-24h solar forecast from the Open-Meteo Forecast API.
   - Requested hourly variables: `shortwave_radiation`, `cloud_cover`, and `temperature_2m`.
   - `shortwave_radiation` is the raw irradiance input for tomorrow's dispatch chart.
   - If the API fails, stop the simulation and show a clear error. Do not fabricate weather data.
2. Estimate forecast error using a scikit-learn regression model.
   - Current model class: `GradientBoostingRegressor`.
   - Target: `actual_solar - forecast_solar`.
   - Real training source: an offline artifact built from Open-Meteo Previous Runs API matched against Open-Meteo Historical Weather API.
   - The `/optimize` route should load the artifact; it should not train from remote archives during the user request.
   - If the real artifact is unavailable or invalid, mark ML uncertainty as unavailable. Do not train on synthetic rows.
3. Convert predicted forecast error into a forecast-uncertainty diagnostic.
   - Current displayed uncertainty: absolute predicted forecast error in W/m2.
   - The MVP does not use ML to overwrite the dispatch forecast.
4. Simulate PV generation and load from the raw Open-Meteo forecast.
   - PV output = irradiance factor * PV size * system efficiency.
   - Load comes from a preset or custom hourly profile.
5. Simulate battery dispatch hour by hour.
   - Track state of charge, PV used directly, battery charge, battery discharge, grid import, and PV export.
6. Compute short-horizon dispatch metrics.
   - Representative 24h savings.
   - Grid import.
   - Exported PV.
   - Final state of charge.
   - Backup hours at the configured critical load.
7. Compute annual economics using historical irradiance.
   - Preferred source: NASA POWER hourly `ALLSKY_SFC_SW_DWN`.
   - Current MVP years: `2001-2025`.
   - Each year is simulated independently.
   - The app reports average, P50, and P90 annual savings.
8. Repeat the annual simulation for multiple battery sizes.
   - Current sizes: `0, 5, 10, 13.5, 15, 20, 30 kWh`.
9. Select the best battery size using a simple blended score.
   - Savings, payback, and outage resilience are scored deterministically.
   - Only payback-qualified batteries are recommended when any option has payback <= 10 years.
10. Display recommendation, dispatch plan, diagnostics, and assumptions.
    - Show what data source was used.
    - Show which NASA historical years were used and whether any years failed to load.
    - Show ML as an uncertainty diagnostic, not as a hidden financial or dispatch decision.

## Target Users

- Residential solar PV owners considering a battery.
- Small commercial solar users with peak tariffs or reliability concerns.
- Medium-sized farms or agro-industrial users with solar PV, reliability concerns, or peak-tariff exposure.
- Solar installers who need a quick pre-sales sizing estimate.
- Energy analysts evaluating simple battery economics.
- Demo audience: people who understand solar but do not want to read an optimization notebook.

## MVP Features

- Scenario input form:
  - focused location selector for Brasilia and West of Bahia
  - existing PV size in kWp
  - average daily consumption in kWh/day
  - critical load in kW
  - battery cost per kWh
  - grid tariff per kWh
  - optional peak tariff per kWh
  - outage duration in hours/month
- Default demo scenario for Brasilia, Brazil, with a second West of Bahia scenario.
- Hourly PV forecast for tomorrow.
- Simple load profile generated from daily consumption.
- Deterministic battery simulation at hourly time steps.
- Common discrete battery sizes tested by default: 0, 5, 10, 13.5, 15, 20, and 30 kWh.
- Recommended battery size.
- Estimated annual savings.
- Estimated payback period.
- Outage backup hours.
- Tomorrow dispatch chart:
  - PV generation
  - load
  - battery state of charge
  - grid import
- ML forecast-error uncertainty estimate using regression.
- Plain-English recommendation explanation.
- README with assumptions and local setup instructions.

## Later Features

- Saved scenarios and accounts using Supabase.
- Multiple tariff windows and demand charges.
- Real historical bill import.
- Seasonal/monthly load profiles.
- Battery degradation and replacement costs.
- Exported reports.
- More sophisticated optimizer.
- Multiple forecast providers.
- Real archived forecast dataset.
- Country-specific tariff presets.
- Installer-facing comparison mode for multiple battery products.

## Non-Goals

- No authentication in the MVP.
- No database in the MVP.
- No PDF reports.
- No tariff scraping.
- No payments.
- No complex policy modelling.
- No real-time battery control.
- No claim of certified engineering design.
- No black-box financial output from ML.
- No overcomplicated dispatch optimization.

## Screens

### Dashboard

Primary screen for the MVP.

- Left or top: scenario input form.
- Right or lower: recommendation cards and chart.
- Cards:
  - recommended battery size
  - annual savings
  - payback period
  - backup hours
- Chart:
  - hourly PV forecast
  - forecast-error uncertainty estimate
  - load
  - battery state of charge
  - grid import
- Before vs after battery comparison card:
  - without battery:
    - estimated grid cost
    - outage exposure
  - with battery:
    - estimated grid cost
    - outage exposure
  - difference:
    - savings
    - extra backup hours
- Explanation panel:
  - why this size was recommended
  - how forecast uncertainty affects confidence
  - key assumptions

### Scenario Summary

Small read-only section showing the active assumptions and location.

### Forecast And Dispatch Detail

Optional MVP subsection or tab if the dashboard becomes crowded.

- raw forecast
- forecast error estimate
- uncertainty estimate
- dispatch table

## Tech Stack

### Frontend

- Next.js: React framework used for the local web dashboard.
- TypeScript: typed frontend code.
- Tailwind CSS: utility CSS used for layout and styling.
- Recharts: React charting library used for the dispatch chart.

### Backend

- FastAPI: Python web framework used to expose HTTP API routes such as `/simulate` and `/optimize`.
- Uvicorn: local ASGI server used to run the FastAPI app during development.
- Python 3.11 or newer.
- Pydantic: request/response validation and typed API input models.

### Modelling

- NumPy: arrays and deterministic simulation math.
- Pandas: optional later analysis; not required for the current core runtime.

### ML

- scikit-learn: machine learning library used by the forecast-error module.
- Current model: `GradientBoostingRegressor`.
- Target: `actual_solar - forecast_solar`.
- Output used by the app: forecast uncertainty based on absolute predicted forecast error.
- Output not used by the app: financial results. All financial outputs stay deterministic.

### Data APIs

- Open-Meteo Forecast API:
  - Used for next-24h dispatch.
  - Variables: `shortwave_radiation`, `cloud_cover`, `temperature_2m`.
- Open-Meteo Previous Runs API:
  - Used by the offline ML builder to fetch archived previous-day forecasts.
- Open-Meteo Historical Weather API:
  - Used as the actual/reanalysis comparison dataset for ML forecast-error training.
- NASA POWER Hourly API:
  - Used for annual economics.
  - Variable: `ALLSKY_SFC_SW_DWN`.
  - Current years: `2001-2025`.
  - Cached locally under `outputs/cache/`.
  - Optional bundled demo cache can live under `packages/engine/data/cache/` for the two fixed locations.

### Optional Later

- Supabase for saved scenarios and auth.

## Proposed Project Structure

```text
apps/
  web/
    Next.js frontend
  api/
    FastAPI backend
packages/
  engine/
    deterministic battery simulation and optimization
  ml/
    forecast-error modelling
notebooks/
  exploration and validation
README.md
bessai-plan.md
```

## Engineering Assumptions

- Hourly time steps are enough for the MVP.
- PV output = irradiance factor * PV size * system efficiency.
- Default system efficiency: 0.8.
- Battery round-trip efficiency: 0.9.
- Battery charges only from PV surplus in the MVP.
- Battery discharges when load exceeds PV and during high-tariff hours.
- Battery keeps a fixed reserve in the current MVP.
- Forecast uncertainty is reported to the user but does not yet change reserve or dispatch.
- Forecast uncertainty should use smoothed seasonal windows rather than exact hour buckets when the training sample is small.
- Current offline ML metadata stores a rolling 5-week daylight uncertainty summary by week-of-year.
- Battery state of charge is bounded between 0 and capacity.
- Load profile can be generated from average daily consumption with a simple morning/evening shape.
- Annual savings should use NASA POWER hourly historical irradiance for `2001-2025` when available.
- Historical annual economics simulate each year independently, then report average, P50, and P90 annual savings.
- If some NASA POWER years fail, use the real years that loaded and report the missing years.
- If no NASA POWER years load, stop annual economics and show a clear error.
- For the current two-location demo, NASA POWER data should be pre-cacheable with `packages/engine/build_historical_cache.py`.
- Runtime cache in `outputs/cache/` should remain uncommitted; bundled cache data should only be committed intentionally for a deployable portfolio demo.
- NASA POWER historical data is not a forecast. It is used to estimate likely annual economics from historical weather years.
- The current 25-year range is acceptable for the MVP. If a longer valid hourly range becomes available, support up to a practical cap such as 50 years.
- Payback = installed battery cost / annual savings.
- Outage backup hours = usable reserved energy / critical load.
- Financial results must come from deterministic simulation, not ML.
- Currency should be configurable, with BRL as the default for the Brasilia demo.
- Peak tariff should default to 18:00 to 21:00 for the MVP.
- Forecast data should use live Open-Meteo. If it is unavailable, the app should fail clearly instead of using fake weather.

## Optimization Approach

Test common discrete battery sizes by default:

```text
0, 5, 10, 13.5, 15, 20, 30 kWh
```

This is more realistic for an MVP than continuous sizing and maps better to actual product choices. Later versions can support custom battery sizes or finer grid search.

For each size:

1. Simulate hourly dispatch.
2. Estimate grid import cost without battery.
3. Estimate grid import cost with battery.
4. Calculate savings.
5. Calculate payback.
6. Calculate backup resilience.
7. Score the battery size.

Initial score:

```text
savings_score = annual_savings / max_annual_savings
resilience_score = min(outage_backup_hours, 8) / 8
payback_score = max(0, 1 - payback_years / 10)
final_score = 0.5 * savings_score + 0.3 * resilience_score + 0.2 * payback_score
```

If annual savings are zero, payback is unavailable, or a denominator is zero, the optimizer should handle the value safely and avoid selecting a battery for impossible economics. Backup hours are capped at 8 hours for scoring so the optimizer does not automatically favor the largest battery once useful resilience has been reached.

MVP recommendation rule:

- If one or more battery sizes have payback <= 10 years, recommend the highest-scoring option from that payback-qualified set.
- If no battery size has payback <= 10 years, recommend the highest-scoring option overall and clearly mark it as financially weak.
- The explanation should state whether the selected battery passed the payback threshold.

The exact score should remain simple and explainable. A slightly smaller battery with strong payback may be better than the largest battery.

MVP default optimization target: blended score combining annual savings, payback, and outage resilience.

## ML Role

The ML model should not directly predict financial outcomes. In the MVP, it should estimate solar forecast error and present forecast uncertainty, while deterministic dispatch uses the raw Open-Meteo forecast.

Core flow:

```text
raw forecast -> ML forecast-error estimate -> forecast uncertainty
```

Then:

- raw forecast drives dispatch in the MVP
- forecast-error estimate informs the user about uncertainty
- deterministic simulation calculates savings, payback, and backup hours
- optimization selects a battery size from deterministic simulation outputs

Suggested model note if real forecast-error training data is unavailable:

```text
Forecast-error model unavailable because real matched forecast-error training data could not be loaded or validated.
```

This note should be visible but not alarming. It belongs in a small model note, not as a large warning.

## ML Forecast-Error Model

The ML model should estimate solar forecast error:

```text
forecast_error = actual_solar - forecast_solar
```

Current implemented model:

- Library: `scikit-learn`.
- Model class: `GradientBoostingRegressor`.
- Training window: offline artifact built from the most recent 365 days of Open-Meteo Previous Runs when available.
- Validation: last 20% holdout rows, reported as raw forecast MAE and model residual MAE.
- Guardrail: if the real artifact is missing, invalid, or not useful, mark ML uncertainty unavailable and keep dispatch on raw Open-Meteo forecast.

Implemented inputs:

- raw forecast solar or irradiance proxy
- hour of day
- day of year
- cloud cover
- temperature
- month
- daylight flag

Output:

```text
forecast_uncertainty = P90 historical daylight forecast error from a rolling seasonal window
```

The app currently displays P90 and mean absolute forecast-error uncertainty. This is designed to show seasonal reliability differences, such as dry-season weeks being more predictable than rainy-season weeks, without overfitting exact hour-level noise. It does not overwrite the raw forecast.

The MVP should train on real matched forecast-error rows when practical:

- Preferred practical source: Open-Meteo Previous Runs API for archived previous-day forecasts.
- Current actual/reanalysis comparison source: Open-Meteo Historical Weather API.
- Build local artifacts with `packages/ml/build_forecast_error_artifact.py` so repeated demos do not hammer APIs.
- Validate that matched rows contain measurable forecast error; if they do not, fall back instead of pretending the model learned useful error.

If real forecast-error data is not available, the MVP should not train a synthetic model. It should keep the interface ready for real archived forecast data later and clearly show that ML diagnostics are unavailable.

The ML model is informational in the MVP. It must not invent annual savings or payback, and it should not override the dispatch forecast until validation is strong enough to justify that behavior.

## API Endpoints

### GET /health

Returns a simple backend health check:

```json
{"status": "ok"}
```

### POST /simulate

Runs a deterministic battery simulation for a single scenario and battery size.

Returns:

- hourly dispatch
- savings estimate
- backup hours
- assumptions

### POST /optimize

Runs simulations for the default battery size set: 0, 5, 10, 13.5, 15, 20, and 30 kWh.

Returns:

- recommended size
- all candidate results
- annual savings
- average, P50, and P90 annual savings when multi-year NASA data is available
- payback
- backup hours
- dispatch series for the recommended battery
- annual simulation diagnostics
- ML forecast-error diagnostics
- explanation

### POST /prefetch-annual-economics

Fetches or loads cached NASA POWER hourly irradiance for the selected location and years.

Current years:

```text
2001-2025
```

This endpoint exists so a developer or future explicit UI action can warm the annual-economics cache. The dashboard should not automatically prefetch all historical data on page load.

## Default Demo Scenario

- Location: Brasilia, Brazil
- Latitude: -15.826016
- Longitude: -47.812539
- Secondary demo location: West of Bahia, Brazil at -13.792761, -46.104032
- PV size: 4 kWp
- Average consumption: 18 kWh/day
- Critical load: 1.5 kW
- Battery cost: 2500 BRL/kWh
- Grid tariff: 0.95 BRL/kWh
- Peak tariff: 1.50 BRL/kWh
- Peak tariff window: 18:00 to 21:00
- Outage duration: 8 hours/month
- Currency: BRL

## MVP Defaults For Open Questions

- Optimization target: blended score combining savings, payback, and outage resilience.
- Battery sizes: common discrete sizes first: 0, 5, 10, 13.5, 15, 20, and 30 kWh.
- Currency: configurable, with BRL as the Brasilia demo default.
- Peak tariff window: 18:00 to 21:00.
- Annual economics: simulate NASA POWER hourly historical irradiance for `2001-2025` when available.
- Historical economics fallback: use the real NASA years that load; if none load, show an error.
- Short-term dispatch forecast: live Open-Meteo Forecast API. If unavailable, show an error.
- ML transparency: show the training source and uncertainty estimate calmly; if real training data is unavailable, say so directly.
- Tone: installer/analyst-friendly, but still understandable to a homeowner.

## MVP Build Order

1. Planning file.
2. Core Python simulation that runs the Brasilia default scenario.
3. Optimization over common discrete battery sizes.
4. Forecast provider wrapper with clear API failure handling.
5. FastAPI app with `/health`, `/simulate`, and `/optimize`.
6. Next.js dashboard with input form, results cards, dispatch chart, and explanation.
7. ML forecast-error uncertainty pipeline.
8. UI polish and README.

## Checklist

- [x] Create `bessai-plan.md`.
- [x] Create base project structure.
- [x] Build deterministic simulation engine.
- [x] Add default Brasilia scenario.
- [x] Add representative load profile generator.
- [x] Add optimization over common discrete battery sizes.
- [x] Add blended scoring for savings, resilience, and payback.
- [x] Add forecast provider wrapper.
- [x] Add forecast-error ML pipeline with real Open-Meteo previous-run training rows and unavailable-state handling.
- [x] Connect ML forecast-error diagnostics to deterministic engine demo.
- [x] Add FastAPI endpoints.
- [x] Add NASA POWER multi-year historical economics path.
- [x] Add Next.js dashboard.
- [x] Add Recharts dispatch visualization.
- [x] Add before vs after battery comparison card.
- [x] Add recommendation explanation generator.
- [x] Add README setup instructions.
- [x] Verify backend runs locally.
- [x] Verify frontend runs locally.
- [x] Verify `npm run build` works.

## Open Questions

- How should the score weights change for residential users versus installer or analyst users?
- Should the default battery sizes become market-specific once the user selects a country?
- Should annual economics use more than the current `2001-2025` range if a longer valid hourly source is available?
- Should P90 be calculated from annual savings, monthly savings, or a more formal climatology method?
- Should the two supported locations ship with prebuilt NASA POWER cache files, or should each developer fetch them locally?
- Should outage exposure be shown as unserved critical-load hours, lost energy, or a simpler risk score?
- Should the first UI expose score components, or keep them behind an explanation drawer?

## Product Notes

- The name BESSAi is strong because it combines BESS with AI and hints at a localized first demo. The accented final character may create avoidable friction in package names, URLs, and terminal paths, so the codebase should probably use `bessai` while the product UI can display `BESSAi` or `BESSAí`.
- The MVP should not hide behind ML. The impressive part is a credible loop: forecast, uncertainty estimate, dispatch, economics, explanation.
- The app should not fabricate weather data. If a required data source is unavailable, it should fail clearly and explain which source failed.
- The UI should look like an operational energy tool: compact, clear, and data-forward.
- Implementation should prefer the simplest working version when constraints appear. The defaults in this plan guide the MVP but are not rigid long-term product decisions.
