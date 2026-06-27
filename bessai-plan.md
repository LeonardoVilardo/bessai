# BESSAi Plan

## App Concept

BESSAi is a web app that helps solar PV users decide whether adding a battery energy storage system is financially and operationally worthwhile.

The app combines hourly solar forecasts, historical solar patterns, a simple forecast-error regression model, and a deterministic battery simulation to recommend a battery size, estimate payback, and show a practical daily dispatch strategy.

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

1. Fetch solar forecast for the next 24 to 48 hours.
2. Estimate forecast error using ML regression.
3. Estimate forecast uncertainty from predicted forecast error.
4. Simulate PV generation and load from the raw forecast.
5. Simulate battery dispatch hour by hour.
6. Compute savings, payback, and resilience.
7. Repeat for multiple battery sizes.
8. Select the best battery size using a simple blended score.
9. Display recommendation and dispatch plan.

## Target Users

- Residential solar PV owners considering a battery.
- Small commercial solar users with peak tariffs or reliability concerns.
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

- Next.js
- TypeScript
- Tailwind CSS
- Recharts

### Backend

- FastAPI
- Python 3.11
- Pydantic

### Modelling

- NumPy
- Pandas

### ML

- scikit-learn regression model
- target: actual_solar - forecast_solar
- output: forecast uncertainty based on predicted forecast error

### Data APIs

- Open-Meteo forecast API for near-term weather and solar-related variables.
- NASA POWER historical solar data if practical.
- Synthetic forecast-error training data as a documented fallback.

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
- Battery keeps a reserve when forecast uncertainty is high.
- Battery state of charge is bounded between 0 and capacity.
- Load profile can be generated from average daily consumption with a simple morning/evening shape.
- Annual savings can be estimated from representative-day or representative-period simulation, then annualized.
- Payback = installed battery cost / annual savings.
- Outage backup hours = usable reserved energy / critical load.
- Financial results must come from deterministic simulation, not ML.
- Currency should be configurable, with BRL as the default for the Brasilia demo.
- Peak tariff should default to 18:00 to 21:00 for the MVP.
- Forecast data should use live Open-Meteo when available and cached or sample fallback data when live data is unavailable.

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

Suggested model note if synthetic forecast-error data is used:

```text
Forecast-error model trained on synthetic demo data. Designed to be replaced with archived forecast data.
```

This note should be visible but not alarming. It belongs in a small model note, not as a large warning.

## ML Forecast-Error Model

The ML model should estimate solar forecast error:

```text
forecast_error = actual_solar - forecast_solar
```

Inputs can include:

- raw forecast solar or irradiance proxy
- hour of day
- day of year
- cloud cover
- temperature
- humidity
- recent forecast bias if available

Output:

```text
forecast_uncertainty = abs(predicted_error)
```

The MVP should train on real matched forecast-error rows when practical:

- Preferred practical source: Open-Meteo Previous Runs API for archived previous-day forecasts.
- Actual comparison source: Open-Meteo Historical Weather API or NASA POWER irradiance.
- Cache downloaded rows locally so repeated demos do not hammer APIs.
- Validate that matched rows contain measurable forecast error; if they do not, fall back instead of pretending the model learned useful error.

If real forecast-error data is not immediately available, the MVP should use a clean synthetic pipeline:

- Generate plausible forecast errors.
- Train a regression model.
- Label it clearly but calmly as synthetic demo training.
- Keep the interface ready for real archived forecast data later.

The ML model is informational in the MVP. It must not invent annual savings or payback, and it should not override the dispatch forecast until validation is strong enough to justify that behavior.

## API Endpoints

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
- payback
- backup hours
- explanation

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
- Annualization: representative week if practical. If not, one representative day is acceptable, clearly labelled as annualized from a representative profile.
- Forecast data: live Open-Meteo when available, with cached or sample fallback data so the demo always works.
- ML transparency: show the training source and uncertainty estimate calmly; mention synthetic training data in a small model note if fallback data is used.
- Tone: installer/analyst-friendly, but still understandable to a homeowner.

## MVP Build Order

1. Planning file.
2. Core Python simulation that runs the Brasilia default scenario.
3. Optimization over common discrete battery sizes.
4. Forecast provider wrapper with a deterministic fallback.
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
- [x] Add forecast-error ML pipeline with real Open-Meteo previous-run training rows and synthetic fallback.
- [x] Connect ML forecast-error diagnostics to deterministic engine demo.
- [x] Add FastAPI endpoints.
- [x] Add NASA POWER full-year historical economics fallback path.
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
- Should the representative week be generated synthetically or based on a recent historical NASA POWER week?
- How much cached/sample forecast data should ship with the demo?
- Should outage exposure be shown as unserved critical-load hours, lost energy, or a simpler risk score?
- Should the first UI expose score components, or keep them behind an explanation drawer?

## Product Notes

- The name BESSAi is strong because it combines BESS with AI and hints at a localized first demo. The accented final character may create avoidable friction in package names, URLs, and terminal paths, so the codebase should probably use `bessai` while the product UI can display `BESSAi` or `BESSAí`.
- The MVP should not hide behind ML. The impressive part is a credible loop: forecast, uncertainty estimate, dispatch, economics, explanation.
- A deterministic fallback forecast is important. The app should still demo well without live API availability.
- The UI should look like an operational energy tool: compact, clear, and data-forward.
- Implementation should prefer the simplest working version when constraints appear. The defaults in this plan guide the MVP but are not rigid long-term product decisions.
