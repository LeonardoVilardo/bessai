"""Minimal FastAPI backend for BESSAi."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from packages.engine.core import (
    DEFAULT_HISTORICAL_YEARS,
    Scenario,
    calculate_metrics,
    extract_clock_hours,
    fetch_open_meteo_forecast,
    fetch_nasa_power_hourly_irradiance,
    generate_load_profile,
    optimize_battery_size_with_multi_year_history,
    run_dispatch_case,
    simulate_battery_dispatch,
    simulate_pv_generation,
    validate_custom_load_shape,
)
from packages.ml.forecast_error import ForecastScenario, get_corrected_shortwave_forecast


app = FastAPI(title="BESSAi API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScenarioInput(BaseModel):
    location_name: str = "Brasilia, Brazil"
    latitude: float = -15.826016
    longitude: float = -47.812539
    timezone: str = "America/Sao_Paulo"
    pv_size_kwp: float = Field(default=4.0, gt=0)
    average_daily_consumption_kwh: float = Field(default=18.0, ge=0)
    critical_load_kw: float = Field(default=1.5, gt=0)
    load_profile_type: str = "residential_evening"
    custom_load_shape: list[float] | None = None
    battery_cost_per_kwh: float = Field(default=2500.0, ge=0)
    grid_tariff_per_kwh: float = Field(default=0.95, ge=0)
    peak_tariff_per_kwh: float = Field(default=1.50, ge=0)
    export_credit_per_kwh: float = Field(default=0.0, ge=0)
    peak_start_hour: int = Field(default=18, ge=0, le=23)
    peak_end_hour: int = Field(default=21, ge=0, le=24)
    outage_duration_hours_per_month: float = Field(default=8.0, ge=0)
    currency: str = "BRL"


class SimulateRequest(BaseModel):
    scenario: ScenarioInput = Field(default_factory=ScenarioInput)
    battery_size_kwh: float = Field(default=10.0, ge=0)


class OptimizeRequest(BaseModel):
    scenario: ScenarioInput = Field(default_factory=ScenarioInput)


class PrefetchAnnualEconomicsRequest(BaseModel):
    scenario: ScenarioInput = Field(default_factory=ScenarioInput)


def build_scenario(input_data: ScenarioInput, battery_size_kwh: float = 10.0) -> Scenario:
    custom_load_shape = tuple(input_data.custom_load_shape) if input_data.custom_load_shape is not None else None
    daily_consumption = input_data.average_daily_consumption_kwh
    if input_data.load_profile_type == "custom":
        validated_shape, _ = validate_custom_load_shape(custom_load_shape)
        daily_consumption = float(np.sum(validated_shape))

    return Scenario(
        location_name=input_data.location_name,
        latitude=input_data.latitude,
        longitude=input_data.longitude,
        timezone=input_data.timezone,
        pv_size_kwp=input_data.pv_size_kwp,
        average_daily_consumption_kwh=daily_consumption,
        critical_load_kw=input_data.critical_load_kw,
        load_profile_type=input_data.load_profile_type,
        custom_load_shape=custom_load_shape,
        battery_capacity_kwh=battery_size_kwh,
        battery_cost_per_kwh=input_data.battery_cost_per_kwh,
        grid_tariff_per_kwh=input_data.grid_tariff_per_kwh,
        peak_tariff_per_kwh=input_data.peak_tariff_per_kwh,
        export_credit_per_kwh=input_data.export_credit_per_kwh,
        peak_start_hour=input_data.peak_start_hour,
        peak_end_hour=input_data.peak_end_hour,
        outage_duration_hours_per_month=input_data.outage_duration_hours_per_month,
        currency=input_data.currency,
    )


def scenario_response(input_data: ScenarioInput, scenario: Scenario) -> dict[str, Any]:
    response = input_data.model_dump()
    response["average_daily_consumption_kwh"] = scenario.average_daily_consumption_kwh
    return response


def load_forecast(scenario: Scenario) -> tuple[dict[str, list[Any]], str]:
    try:
        return fetch_open_meteo_forecast(scenario), "Open-Meteo"
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Open-Meteo forecast data is unavailable, so dispatch simulation cannot run "
                f"with real forecast data. Reason: {exc}"
            ),
        ) from exc


def nullable_round(value: Any, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def fetch_available_historical_years(scenario: Scenario) -> tuple[list[dict[str, list[Any]]], list[int]]:
    historical_years: list[dict[str, list[Any]]] = []
    failed_years: list[int] = []
    for year in DEFAULT_HISTORICAL_YEARS:
        try:
            historical_years.append(fetch_nasa_power_hourly_irradiance(scenario, year))
        except Exception:
            failed_years.append(int(year))
    return historical_years, failed_years


def dispatch_series(
    times: list[str],
    pv_generation: np.ndarray,
    load: np.ndarray,
    dispatch: dict[str, np.ndarray | float],
) -> list[dict[str, float | str]]:
    soc = np.asarray(dispatch["soc"], dtype=float)
    grid_import = np.asarray(dispatch["grid_import"], dtype=float)
    battery_charge = np.asarray(dispatch["battery_charge"], dtype=float)
    battery_discharge = np.asarray(dispatch["battery_discharge"], dtype=float)
    pv_exported = np.asarray(dispatch["pv_exported"], dtype=float)

    return [
        {
            "time": times[index],
            "pv_generation_kwh": round(float(pv_generation[index]), 4),
            "load_kwh": round(float(load[index]), 4),
            "battery_soc_kwh": round(float(soc[index]), 4),
            "grid_import_kwh": round(float(grid_import[index]), 4),
            "battery_charge_kwh": round(float(battery_charge[index]), 4),
            "battery_discharge_kwh": round(float(battery_discharge[index]), 4),
            "pv_exported_kwh": round(float(pv_exported[index]), 4),
        }
        for index in range(len(times))
    ]


def candidate_row(case: dict[str, Any]) -> dict[str, Any]:
    payback = case["payback_years"]
    return {
        "battery_size_kwh": case["battery_size_kwh"],
        "grid_import_kwh": case["grid_import_kwh"],
        "no_battery_exported_pv_kwh": case["no_battery_exported_pv_kwh"],
        "exported_pv_kwh": case["exported_pv_kwh"],
        "daily_cost_without_battery": case["daily_cost_without_battery"],
        "daily_cost_with_battery": case["daily_cost_with_battery"],
        "representative_24h_savings": case["daily_savings"],
        "annualised_savings_estimate": case["annual_savings"],
        "average_annual_savings": case.get("average_annual_savings"),
        "p50_annual_savings": case.get("p50_annual_savings"),
        "p90_annual_savings": case.get("p90_annual_savings"),
        "annual_cost_without_battery": case.get("annual_cost_without_battery"),
        "annual_cost_with_battery": case.get("annual_cost_with_battery"),
        "battery_cost": case["battery_cost"],
        "payback_years": payback if np.isfinite(payback) else None,
        "outage_backup_hours": case["outage_backup_hours"],
        "score": case["final_score"],
        "passes_payback_threshold": bool(case.get("passes_payback_threshold", False)),
    }


def annual_simulation_diagnostics(
    case: dict[str, Any],
    source: str,
    annualization_factor: float = 1.0,
) -> dict[str, float | str]:
    metrics = case["metrics"]
    return {
        "source": source,
        "annual_pv_generation_kwh": round(float(metrics["total_pv_kwh"]) * annualization_factor, 2),
        "annual_load_kwh": round(float(metrics["total_load_kwh"]) * annualization_factor, 2),
        "annual_grid_import_without_battery_kwh": round(
            float(metrics["no_battery_grid_import_kwh"]) * annualization_factor,
            2,
        ),
        "annual_grid_import_with_battery_kwh": round(
            float(metrics["grid_import_kwh"]) * annualization_factor,
            2,
        ),
        "annual_exported_without_battery_kwh": round(
            float(metrics["no_battery_exported_pv_kwh"]) * annualization_factor,
            2,
        ),
        "annual_exported_with_battery_kwh": round(
            float(metrics["exported_pv_kwh"]) * annualization_factor,
            2,
        ),
        "self_consumed_pv_kwh": round(float(metrics["self_consumed_pv_kwh"]) * annualization_factor, 2),
        "self_consumption_ratio": round(float(metrics["self_consumption_ratio"]), 4),
        "average_annual_savings": round(float(case.get("average_annual_savings", case["annual_savings"])), 2),
        "p50_annual_savings": round(float(case.get("p50_annual_savings", case["annual_savings"])), 2),
        "p90_annual_savings": round(float(case.get("p90_annual_savings", case["annual_savings"])), 2),
    }


def ml_diagnostics(ml_forecast: dict[str, Any]) -> dict[str, float | int | str]:
    return {
        "model_source": ml_forecast["training_source"],
        "training_rows": ml_forecast["training_rows"],
        "training_period": ml_forecast["training_period"],
        "mae_before_correction_w_m2": nullable_round(ml_forecast["mae_before_correction_w_m2"]),
        "mae_after_correction_w_m2": nullable_round(ml_forecast["mae_after_correction_w_m2"]),
        "mean_forecast_error_w_m2": nullable_round(ml_forecast["mean_forecast_error_w_m2"]),
        "mean_forecast_uncertainty_w_m2": nullable_round(ml_forecast["mean_forecast_uncertainty_w_m2"]),
        "max_forecast_uncertainty_w_m2": nullable_round(ml_forecast["max_forecast_uncertainty_w_m2"]),
        "evaluation_basis": ml_forecast["evaluation_basis"],
        "correction_applied": bool(ml_forecast["ml_correction_applied"]),
        "correction_reason": ml_forecast["ml_correction_reason"],
    }


def custom_load_shape_note(scenario: Scenario) -> str:
    if scenario.load_profile_type != "custom":
        return "Preset load profile used."
    return validate_custom_load_shape(scenario.custom_load_shape)[1]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/simulate")
def simulate(request: SimulateRequest = Body(default_factory=SimulateRequest)) -> dict[str, Any]:
    scenario = build_scenario(request.scenario, request.battery_size_kwh)
    forecast, forecast_source = load_forecast(scenario)
    times = list(forecast["time"])
    shortwave = np.array(forecast["shortwave_radiation_w_m2"], dtype=float)
    clock_hours = extract_clock_hours(times)
    load = generate_load_profile(
        scenario.average_daily_consumption_kwh,
        clock_hours,
        scenario.load_profile_type,
        scenario.custom_load_shape,
    )
    pv_generation = simulate_pv_generation(shortwave, scenario)
    dispatch = simulate_battery_dispatch(pv_generation, load, scenario)
    metrics = calculate_metrics(pv_generation, load, dispatch, scenario, clock_hours)

    return {
        "forecast_source": forecast_source,
        "scenario": scenario_response(request.scenario, scenario),
        "battery_size_kwh": request.battery_size_kwh,
        "metrics": metrics,
        "dispatch": dispatch_series(times, pv_generation, load, dispatch),
    }


@app.post("/prefetch-annual-economics")
def prefetch_annual_economics(
    request: PrefetchAnnualEconomicsRequest = Body(default_factory=PrefetchAnnualEconomicsRequest),
) -> dict[str, Any]:
    scenario = build_scenario(request.scenario)
    fetched_years: list[int] = []
    failed_years: list[int] = []

    historical_years, failed_years = fetch_available_historical_years(scenario)
    fetched_years = [int(year["year"]) for year in historical_years]

    return {
        "status": "ready" if not failed_years else "partial",
        "source": "NASA POWER hourly historical irradiance",
        "years": fetched_years,
        "failed_years": failed_years,
        "note": (
            "Historical annual economics cache is ready for this location."
            if not failed_years
            else "Some historical years could not be cached; optimisation will use the real years that are available."
        ),
    }


@app.post("/optimize")
def optimize(request: OptimizeRequest = Body(default_factory=OptimizeRequest)) -> dict[str, Any]:
    scenario = build_scenario(request.scenario)
    try:
        ml_forecast = get_corrected_shortwave_forecast(
            ForecastScenario(
                location_name=scenario.location_name,
                latitude=scenario.latitude,
                longitude=scenario.longitude,
                timezone=scenario.timezone,
            )
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Open-Meteo forecast data is unavailable, so optimization cannot run "
                f"with real dispatch forecast data. Reason: {exc}"
            ),
        ) from exc
    if ml_forecast["training_source"] == "unavailable":
        forecast_source = f"{ml_forecast['forecast_source']} raw forecast (ML uncertainty unavailable)"
    else:
        forecast_source = f"{ml_forecast['forecast_source']} raw forecast + ML uncertainty estimate"
    times = list(ml_forecast["time"])
    shortwave = np.array(ml_forecast["selected_shortwave_radiation_w_m2"], dtype=float)
    clock_hours = extract_clock_hours(times)
    load = generate_load_profile(
        scenario.average_daily_consumption_kwh,
        clock_hours,
        scenario.load_profile_type,
        scenario.custom_load_shape,
    )
    pv_generation = simulate_pv_generation(shortwave, scenario)

    economics_source = "NASA POWER hourly historical irradiance unavailable"
    economics_note = "NASA POWER annual historical irradiance unavailable; recommendation not calculated."
    annual_economics_year: int | None = None
    annual_economics_years: list[int] = []
    diagnostics_annualization_factor = 1.0
    historical_years, failed_historical_years = fetch_available_historical_years(scenario)
    if not historical_years:
        raise HTTPException(
            status_code=503,
            detail=(
                "NASA POWER historical irradiance is unavailable for the selected location, "
                "so annual savings and payback cannot be calculated from real historical data."
            ),
        )

    cases, recommended = optimize_battery_size_with_multi_year_history(scenario, historical_years)
    annual_economics_years = [int(year["year"]) for year in historical_years]
    economics_source = (
        f"NASA POWER hourly historical irradiance, "
        f"{annual_economics_years[0]}-{annual_economics_years[-1]}"
    )
    economics_note = (
        "Annual savings and payback use real NASA POWER hourly historical irradiance. "
        "Recommendation uses P50 annual savings; P90 is the 10th-percentile conservative weather-year estimate. "
        "Dispatch chart uses next-24h Open-Meteo forecast."
    )
    if failed_historical_years:
        economics_note += f" Missing historical years: {failed_historical_years}."

    recommended_size = recommended["battery_size_kwh"]
    recommended_case = run_dispatch_case(scenario, recommended_size, pv_generation, load, clock_hours)
    recommended_dispatch = recommended_case["dispatch"]

    return {
        "forecast_source": forecast_source,
        "ml_correction_applied": bool(ml_forecast["ml_correction_applied"]),
        "ml_correction_reason": ml_forecast["ml_correction_reason"],
        "ml_training_source": ml_forecast["training_source"],
        "ml_training_rows": ml_forecast["training_rows"],
        "ml_training_period": ml_forecast["training_period"],
        "ml_model_note": ml_forecast["model_note"],
        "economics_source": economics_source,
        "economics_note": economics_note,
        "model_diagnostics": {
            "annual_simulation": annual_simulation_diagnostics(
                recommended,
                economics_source,
                diagnostics_annualization_factor,
            ),
            "ml": ml_diagnostics(ml_forecast),
        },
        "assumptions": {
            "annual_economics_year": annual_economics_year,
            "annual_economics_years": annual_economics_years,
            "forecast_source_for_dispatch": forecast_source,
            "ml_correction_applied": bool(ml_forecast["ml_correction_applied"]),
            "ml_correction_reason": ml_forecast["ml_correction_reason"],
            "battery_round_trip_efficiency": scenario.battery_round_trip_efficiency,
            "reserve_soc_fraction": scenario.reserve_soc_fraction,
            "load_profile_type": scenario.load_profile_type,
            "custom_load_shape_note": custom_load_shape_note(scenario),
            "export_credit_modeled": True,
            "export_credit_per_kwh": scenario.export_credit_per_kwh,
            "export_credit_note": "Surplus PV is credited using the configured export credit. No country-specific policy is hardcoded.",
        },
        "scenario": scenario_response(request.scenario, scenario),
        "recommendation": candidate_row(recommended),
        "comparison": [candidate_row(case) for case in cases],
        "dispatch": dispatch_series(times, pv_generation, load, recommended_dispatch),
    }
