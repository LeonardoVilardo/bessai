"""Core deterministic BESSAi simulation and optimization logic."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
NASA_POWER_HOURLY_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
CACHE_DIR = OUTPUT_DIR / "cache"
BUNDLED_CACHE_DIR = Path(__file__).resolve().parent / "data" / "cache"
MVP_BATTERY_SIZES_KWH = (0.0, 5.0, 10.0, 13.5, 15.0, 20.0, 30.0)
ANNUALIZATION_DAYS = 365
BACKUP_HOURS_SCORE_CAP = 8.0
PAYBACK_RECOMMENDATION_THRESHOLD_YEARS = 10.0
DEFAULT_HISTORICAL_YEAR = 2024
DEFAULT_HISTORICAL_YEARS = tuple(range(2001, 2026))
DEFAULT_LOAD_PROFILE_TYPE = "residential_evening"
LOAD_PROFILE_TYPES = ("residential_evening", "residential_daytime", "small_business", "flat", "custom")


@dataclass(frozen=True)
class Scenario:
    location_name: str = "Brasilia, Brazil"
    latitude: float = -15.826016
    longitude: float = -47.812539
    timezone: str = "America/Sao_Paulo"
    pv_size_kwp: float = 4.0
    average_daily_consumption_kwh: float = 18.0
    critical_load_kw: float = 1.5
    battery_capacity_kwh: float = 10.0
    initial_soc_fraction: float = 0.5
    min_soc_fraction: float = 0.1
    reserve_soc_fraction: float = 0.2
    battery_round_trip_efficiency: float = 0.90
    system_efficiency: float = 0.80
    grid_tariff_per_kwh: float = 0.95
    peak_tariff_per_kwh: float = 1.50
    export_credit_per_kwh: float = 0.95
    minimum_monthly_bill_kwh: float = 100.0
    battery_cost_per_kwh: float = 2500.0
    tariff_mode: str = "flat"
    peak_start_hour: int = 18
    peak_end_hour: int = 21
    outage_duration_hours_per_month: float = 8.0
    currency: str = "BRL"
    load_profile_type: str = DEFAULT_LOAD_PROFILE_TYPE
    custom_load_shape: tuple[float, ...] | None = None


def fetch_open_meteo_forecast(scenario: Scenario, hours: int = 24) -> dict[str, list[Any]]:
    """Fetch hourly shortwave radiation from Open-Meteo."""
    params = {
        "latitude": scenario.latitude,
        "longitude": scenario.longitude,
        "hourly": "shortwave_radiation",
        "timezone": scenario.timezone,
        "forecast_hours": hours,
    }
    url = f"{OPEN_METEO_URL}?{urlencode(params)}"

    with urlopen(url, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    hourly = payload.get("hourly", {})
    times = hourly.get("time")
    radiation = hourly.get("shortwave_radiation")

    if not times or not radiation:
        raise RuntimeError("Open-Meteo response did not include hourly shortwave radiation.")

    return {
        "time": times[:hours],
        "shortwave_radiation_w_m2": radiation[:hours],
    }


def fetch_nasa_power_hourly_irradiance(
    scenario: Scenario,
    year: int = DEFAULT_HISTORICAL_YEAR,
) -> dict[str, list[Any]]:
    """Fetch one historical year of hourly solar irradiance from NASA POWER.

    NASA POWER returns ALLSKY_SFC_SW_DWN hourly values in Wh/m^2. For an hourly
    PV model this can be treated like average W/m^2 over the hour.
    """
    runtime_cache_path = nasa_power_cache_path(scenario, year)
    for cache_path in nasa_power_cache_paths(scenario, year):
        if cache_path.exists():
            return json.loads(cache_path.read_text())

    params = {
        "parameters": "ALLSKY_SFC_SW_DWN",
        "community": "RE",
        "longitude": scenario.longitude,
        "latitude": scenario.latitude,
        "start": f"{year}0101",
        "end": f"{year}1231",
        "format": "JSON",
        "time-standard": "LST",
    }
    url = f"{NASA_POWER_HOURLY_URL}?{urlencode(params)}"

    with urlopen(url, timeout=45) as response:
        payload = json.loads(response.read().decode("utf-8"))

    values = payload.get("properties", {}).get("parameter", {}).get("ALLSKY_SFC_SW_DWN", {})
    fill_value = payload.get("header", {}).get("fill_value", -999.0)
    if not values:
        raise RuntimeError("NASA POWER response did not include ALLSKY_SFC_SW_DWN data.")

    times: list[str] = []
    irradiance: list[float] = []
    for key in sorted(values):
        value = float(values[key])
        if value == fill_value:
            value = 0.0
        times.append(key)
        irradiance.append(max(0.0, value))

    result = {
        "time": times,
        "shortwave_radiation_w_m2": irradiance,
        "year": year,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    runtime_cache_path.write_text(json.dumps(result))
    return result


def nasa_power_cache_filename(scenario: Scenario, year: int) -> str:
    lat = f"{scenario.latitude:.4f}".replace("-", "m").replace(".", "p")
    lon = f"{scenario.longitude:.4f}".replace("-", "m").replace(".", "p")
    return f"nasa_power_hourly_{lat}_{lon}_{year}.json"


def nasa_power_cache_path(scenario: Scenario, year: int) -> Path:
    return CACHE_DIR / nasa_power_cache_filename(scenario, year)


def nasa_power_bundled_cache_path(scenario: Scenario, year: int) -> Path:
    return BUNDLED_CACHE_DIR / nasa_power_cache_filename(scenario, year)


def nasa_power_cache_paths(scenario: Scenario, year: int) -> tuple[Path, Path]:
    return (
        nasa_power_cache_path(scenario, year),
        nasa_power_bundled_cache_path(scenario, year),
    )


def extract_clock_hours(times: list[str]) -> list[int]:
    """Extract local hour-of-day values from Open-Meteo timestamp strings."""
    clock_hours = []
    for timestamp in times:
        if len(timestamp) == 10 and timestamp.isdigit():
            clock_hours.append(int(timestamp[-2:]))
        else:
            clock_hours.append(datetime.fromisoformat(timestamp).hour)
    return clock_hours


def validate_custom_load_shape(custom_load_shape: tuple[float, ...] | list[float] | None) -> tuple[np.ndarray, str]:
    """Return valid custom hourly kWh values or a flat fallback."""
    fallback = np.ones(24, dtype=float)
    if custom_load_shape is None:
        return fallback, "Custom hourly load missing; using flat 1 kWh/hour fallback."

    values = np.array(custom_load_shape, dtype=float)
    if values.shape != (24,):
        return fallback, "Custom hourly load must contain 24 values; using flat 1 kWh/hour fallback."
    if not np.all(np.isfinite(values)):
        return fallback, "Custom hourly load contains non-finite values; using flat 1 kWh/hour fallback."
    if np.any(values < 0):
        return fallback, "Custom hourly load contains negative values; using flat 1 kWh/hour fallback."
    if float(np.sum(values)) <= 0:
        return fallback, "Custom hourly load sums to zero; using flat 1 kWh/hour fallback."

    return values, "Custom hourly kWh profile provided; daily consumption is the sum of the 24 hours."


def load_shape_by_hour(
    load_profile_type: str,
    custom_load_shape: tuple[float, ...] | list[float] | None = None,
) -> np.ndarray:
    """Return a 24-hour relative load shape for the selected profile."""
    if load_profile_type == "residential_daytime":
        return np.array(
            [
                0.42,
                0.38,
                0.36,
                0.35,
                0.38,
                0.58,
                0.92,
                1.12,
                1.08,
                1.00,
                0.96,
                0.92,
                0.98,
                1.04,
                1.02,
                0.96,
                0.92,
                0.98,
                1.08,
                1.12,
                1.02,
                0.86,
                0.64,
                0.48,
            ],
            dtype=float,
        )
    if load_profile_type == "small_business":
        return np.array(
            [
                0.22,
                0.20,
                0.20,
                0.20,
                0.22,
                0.35,
                0.72,
                1.10,
                1.35,
                1.48,
                1.52,
                1.50,
                1.46,
                1.44,
                1.42,
                1.36,
                1.18,
                0.78,
                0.48,
                0.34,
                0.30,
                0.28,
                0.26,
                0.24,
            ],
            dtype=float,
        )
    if load_profile_type == "custom":
        return validate_custom_load_shape(custom_load_shape)[0]
    if load_profile_type == "flat":
        return np.ones(24, dtype=float)
    return np.array(
        [
            0.45,
            0.40,
            0.38,
            0.36,
            0.38,
            0.55,
            0.85,
            1.05,
            0.95,
            0.80,
            0.70,
            0.68,
            0.72,
            0.76,
            0.82,
            0.92,
            1.08,
            1.28,
            1.45,
            1.50,
            1.34,
            1.10,
            0.78,
            0.55,
        ],
        dtype=float,
    )


def generate_load_profile(
    daily_kwh: float,
    clock_hours: list[int],
    load_profile_type: str = DEFAULT_LOAD_PROFILE_TYPE,
    custom_load_shape: tuple[float, ...] | list[float] | None = None,
) -> np.ndarray:
    """Generate a simple load profile for the selected 24-hour window."""
    base_shape_by_hour = load_shape_by_hour(load_profile_type, custom_load_shape)
    selected_shape = np.array([base_shape_by_hour[hour] for hour in clock_hours], dtype=float)
    if load_profile_type == "custom":
        return selected_shape
    return selected_shape / selected_shape.sum() * daily_kwh


def generate_repeated_load_profile(
    daily_kwh: float,
    clock_hours: list[int],
    load_profile_type: str = DEFAULT_LOAD_PROFILE_TYPE,
    custom_load_shape: tuple[float, ...] | list[float] | None = None,
) -> np.ndarray:
    """Generate hourly load for multi-day periods from the same daily shape."""
    base_shape_by_hour = load_shape_by_hour(load_profile_type, custom_load_shape)
    if load_profile_type == "custom":
        return np.array([base_shape_by_hour[hour] for hour in clock_hours], dtype=float)
    hourly_load_by_hour = base_shape_by_hour / base_shape_by_hour.sum() * daily_kwh
    return np.array([hourly_load_by_hour[hour] for hour in clock_hours], dtype=float)


def simulate_pv_generation(shortwave_radiation_w_m2: np.ndarray, scenario: Scenario) -> np.ndarray:
    """Convert shortwave radiation into hourly PV generation in kWh."""
    irradiance_factor = np.clip(shortwave_radiation_w_m2 / 1000.0, 0.0, None)
    return irradiance_factor * scenario.pv_size_kwp * scenario.system_efficiency


def hourly_tariffs(scenario: Scenario, clock_hours: list[int]) -> np.ndarray:
    tariffs = np.full(len(clock_hours), scenario.grid_tariff_per_kwh, dtype=float)
    if scenario.tariff_mode == "flat":
        return tariffs

    for index, hour in enumerate(clock_hours):
        if scenario.peak_start_hour <= hour < scenario.peak_end_hour:
            tariffs[index] = scenario.peak_tariff_per_kwh
    return tariffs


def minimum_bill_for_period(scenario: Scenario, hours: int) -> float:
    period_days = hours / 24
    annual_minimum_bill = scenario.minimum_monthly_bill_kwh * scenario.grid_tariff_per_kwh * 12
    return annual_minimum_bill * period_days / ANNUALIZATION_DAYS


def simulate_battery_dispatch(
    pv_generation_kwh: np.ndarray,
    load_kwh: np.ndarray,
    scenario: Scenario,
) -> dict[str, np.ndarray | float]:
    """Simulate PV self-consumption and battery dispatch hour by hour."""
    hours = len(load_kwh)
    soc = np.zeros(hours, dtype=float)
    grid_import = np.zeros(hours, dtype=float)
    battery_charge = np.zeros(hours, dtype=float)
    battery_discharge = np.zeros(hours, dtype=float)
    pv_used_directly = np.zeros(hours, dtype=float)
    pv_exported = np.zeros(hours, dtype=float)
    curtailed_pv = np.zeros(hours, dtype=float)

    charge_efficiency = math.sqrt(scenario.battery_round_trip_efficiency)
    discharge_efficiency = math.sqrt(scenario.battery_round_trip_efficiency)
    current_soc = scenario.battery_capacity_kwh * scenario.initial_soc_fraction
    min_soc = scenario.battery_capacity_kwh * scenario.min_soc_fraction
    reserve_soc = scenario.battery_capacity_kwh * scenario.reserve_soc_fraction
    discharge_floor = max(min_soc, reserve_soc)

    for hour in range(hours):
        pv = pv_generation_kwh[hour]
        load = load_kwh[hour]
        direct_pv = min(pv, load)
        remaining_load = load - direct_pv
        surplus_pv = pv - direct_pv

        if surplus_pv > 0 and scenario.battery_capacity_kwh > 0:
            free_capacity = scenario.battery_capacity_kwh - current_soc
            storable_from_pv = free_capacity / charge_efficiency if charge_efficiency else 0.0
            charge_from_pv = min(surplus_pv, storable_from_pv)
            current_soc += charge_from_pv * charge_efficiency
            battery_charge[hour] = charge_from_pv
            surplus_pv -= charge_from_pv

        if remaining_load > 0 and scenario.battery_capacity_kwh > 0:
            available_from_battery = max(0.0, current_soc - discharge_floor)
            deliverable_to_load = available_from_battery * discharge_efficiency
            discharge_to_load = min(remaining_load, deliverable_to_load)
            current_soc -= discharge_to_load / discharge_efficiency
            remaining_load -= discharge_to_load
            battery_discharge[hour] = discharge_to_load

        pv_used_directly[hour] = direct_pv
        grid_import[hour] = remaining_load
        pv_exported[hour] = surplus_pv
        soc[hour] = current_soc

    return {
        "soc": soc,
        "grid_import": grid_import,
        "battery_charge": battery_charge,
        "battery_discharge": battery_discharge,
        "pv_used_directly": pv_used_directly,
        "pv_exported": pv_exported,
        "curtailed_pv": curtailed_pv,
        "min_soc": min_soc,
        "reserve_soc": reserve_soc,
    }


def calculate_metrics(
    pv_generation_kwh: np.ndarray,
    load_kwh: np.ndarray,
    dispatch: dict[str, np.ndarray | float],
    scenario: Scenario,
    clock_hours: list[int],
) -> dict[str, float]:
    tariffs = hourly_tariffs(scenario, clock_hours)
    grid_import = np.asarray(dispatch["grid_import"], dtype=float)
    battery_charge = np.asarray(dispatch["battery_charge"], dtype=float)
    battery_discharge = np.asarray(dispatch["battery_discharge"], dtype=float)
    pv_exported = np.asarray(dispatch["pv_exported"], dtype=float)
    curtailed_pv = np.asarray(dispatch["curtailed_pv"], dtype=float)
    final_soc = float(np.asarray(dispatch["soc"], dtype=float)[-1])
    min_soc = float(dispatch["min_soc"])
    reserve_soc = float(dispatch["reserve_soc"])

    no_battery_grid_import = np.maximum(load_kwh - pv_generation_kwh, 0.0)
    no_battery_exported_pv = np.maximum(pv_generation_kwh - load_kwh, 0.0)
    total_pv = float(np.sum(pv_generation_kwh))
    exported_pv_kwh = float(np.sum(pv_exported))
    no_battery_exported_pv_kwh = float(np.sum(no_battery_exported_pv))
    curtailed_pv_kwh = float(np.sum(curtailed_pv))
    self_consumed_pv_kwh = max(0.0, total_pv - exported_pv_kwh - curtailed_pv_kwh)
    import_cost_without_battery = float(np.sum(no_battery_grid_import * tariffs))
    import_cost_with_battery = float(np.sum(grid_import * tariffs))
    export_credit_without_battery = no_battery_exported_pv_kwh * scenario.export_credit_per_kwh
    export_credit_with_battery = exported_pv_kwh * scenario.export_credit_per_kwh
    raw_cost_without_battery = import_cost_without_battery - export_credit_without_battery
    raw_cost_with_battery = import_cost_with_battery - export_credit_with_battery
    minimum_bill = minimum_bill_for_period(scenario, len(load_kwh))
    billed_cost_without_battery = max(raw_cost_without_battery, minimum_bill)
    billed_cost_with_battery = max(raw_cost_with_battery, minimum_bill)
    import_savings_before_export_credit = import_cost_without_battery - import_cost_with_battery
    export_credit_reduction = export_credit_without_battery - export_credit_with_battery
    raw_energy_savings = raw_cost_without_battery - raw_cost_with_battery
    bill_savings_after_minimum = billed_cost_without_battery - billed_cost_with_battery
    usable_backup_energy = max(0.0, final_soc - min_soc)
    reserve_backup_energy = max(0.0, reserve_soc - min_soc)
    outage_backup_hours = usable_backup_energy / scenario.critical_load_kw if scenario.critical_load_kw else 0.0
    reserve_backup_hours = reserve_backup_energy / scenario.critical_load_kw if scenario.critical_load_kw else 0.0

    return {
        "total_load_kwh": float(np.sum(load_kwh)),
        "total_pv_kwh": total_pv,
        "no_battery_grid_import_kwh": float(np.sum(no_battery_grid_import)),
        "grid_import_kwh": float(np.sum(grid_import)),
        "no_battery_exported_pv_kwh": no_battery_exported_pv_kwh,
        "exported_pv_kwh": exported_pv_kwh,
        "battery_charged_kwh": float(np.sum(battery_charge)),
        "battery_discharged_kwh": float(np.sum(battery_discharge)),
        "battery_shifted_kwh": float(np.sum(battery_discharge)),
        "curtailed_pv_kwh": curtailed_pv_kwh,
        "self_consumed_pv_kwh": self_consumed_pv_kwh,
        "self_consumption_ratio": safe_divide(self_consumed_pv_kwh, total_pv),
        "import_cost_without_battery": import_cost_without_battery,
        "import_cost_with_battery": import_cost_with_battery,
        "export_credit_without_battery": export_credit_without_battery,
        "export_credit_with_battery": export_credit_with_battery,
        "raw_cost_without_battery": raw_cost_without_battery,
        "raw_cost_with_battery": raw_cost_with_battery,
        "minimum_bill": minimum_bill,
        "billed_cost_without_battery": billed_cost_without_battery,
        "billed_cost_with_battery": billed_cost_with_battery,
        "import_savings_before_export_credit": import_savings_before_export_credit,
        "export_credit_reduction": export_credit_reduction,
        "raw_energy_savings": raw_energy_savings,
        "bill_savings_after_minimum": bill_savings_after_minimum,
        "daily_cost_without_battery": billed_cost_without_battery,
        "daily_cost_with_battery": billed_cost_with_battery,
        "daily_savings": bill_savings_after_minimum,
        "outage_backup_hours": outage_backup_hours,
        "reserve_backup_hours": reserve_backup_hours,
        "final_soc_kwh": final_soc,
    }


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def run_dispatch_case(
    base_scenario: Scenario,
    battery_size_kwh: float,
    pv_generation_kwh: np.ndarray,
    load_kwh: np.ndarray,
    clock_hours: list[int],
) -> dict[str, Any]:
    scenario = replace(base_scenario, battery_capacity_kwh=battery_size_kwh)
    dispatch = simulate_battery_dispatch(pv_generation_kwh, load_kwh, scenario)
    metrics = calculate_metrics(pv_generation_kwh, load_kwh, dispatch, scenario, clock_hours)

    annual_savings = max(0.0, metrics["daily_savings"] * ANNUALIZATION_DAYS)
    battery_cost = battery_size_kwh * scenario.battery_cost_per_kwh
    payback_years = battery_cost / annual_savings if battery_cost > 0 and annual_savings > 0 else math.inf

    return {
        "scenario": scenario,
        "dispatch": dispatch,
        "battery_size_kwh": battery_size_kwh,
        "grid_import_kwh": metrics["grid_import_kwh"],
        "no_battery_exported_pv_kwh": metrics["no_battery_exported_pv_kwh"],
        "exported_pv_kwh": metrics["exported_pv_kwh"],
        "daily_cost_without_battery": metrics["daily_cost_without_battery"],
        "daily_cost_with_battery": metrics["daily_cost_with_battery"],
        "daily_savings": metrics["daily_savings"],
        "annual_savings": annual_savings,
        "battery_cost": battery_cost,
        "payback_years": payback_years,
        "outage_backup_hours": metrics["outage_backup_hours"],
        "metrics": metrics,
    }


def add_blended_scores(cases: list[dict[str, Any]]) -> None:
    max_annual_savings = max((case["annual_savings"] for case in cases), default=0.0)

    for case in cases:
        payback_years = case["payback_years"]
        savings_score = safe_divide(case["annual_savings"], max_annual_savings)
        resilience_score = min(case["outage_backup_hours"], BACKUP_HOURS_SCORE_CAP) / BACKUP_HOURS_SCORE_CAP
        payback_score = max(0.0, 1 - payback_years / 10) if math.isfinite(payback_years) else 0.0
        final_score = 0.5 * savings_score + 0.3 * resilience_score + 0.2 * payback_score

        case["savings_score"] = savings_score
        case["resilience_score"] = resilience_score
        case["payback_score"] = payback_score
        case["final_score"] = final_score


def select_recommended_case(cases: list[dict[str, Any]]) -> dict[str, Any]:
    viable_cases = [case for case in cases if case["annual_savings"] > 0 and case["battery_size_kwh"] > 0]
    payback_qualified_cases = [
        case for case in viable_cases if case["payback_years"] <= PAYBACK_RECOMMENDATION_THRESHOLD_YEARS
    ]

    if payback_qualified_cases:
        recommended = max(payback_qualified_cases, key=lambda case: case["final_score"])
    else:
        recommended = next((case for case in cases if case["battery_size_kwh"] == 0), cases[0])

    recommended["passes_payback_threshold"] = bool(payback_qualified_cases) and (
        recommended["payback_years"] <= PAYBACK_RECOMMENDATION_THRESHOLD_YEARS
    )
    recommended["financially_weak_recommendation"] = bool(viable_cases) and not bool(payback_qualified_cases)
    recommended["resilience_only_not_recommended"] = not bool(viable_cases)
    return recommended


def optimize_battery_size(
    base_scenario: Scenario,
    pv_generation_kwh: np.ndarray,
    load_kwh: np.ndarray,
    clock_hours: list[int],
    battery_sizes_kwh: tuple[float, ...] = MVP_BATTERY_SIZES_KWH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases = [
        run_dispatch_case(base_scenario, size, pv_generation_kwh, load_kwh, clock_hours)
        for size in battery_sizes_kwh
    ]
    add_blended_scores(cases)
    recommended = select_recommended_case(cases)

    return cases, recommended


def apply_full_period_annual_metrics(case: dict[str, Any]) -> None:
    metrics = case["metrics"]
    case["annual_savings"] = max(0.0, metrics["daily_savings"])
    case["daily_savings"] = metrics["daily_savings"] / ANNUALIZATION_DAYS
    case["daily_cost_without_battery"] = metrics["daily_cost_without_battery"] / ANNUALIZATION_DAYS
    case["daily_cost_with_battery"] = metrics["daily_cost_with_battery"] / ANNUALIZATION_DAYS
    case["grid_import_kwh"] = metrics["grid_import_kwh"]
    case["no_battery_exported_pv_kwh"] = metrics["no_battery_exported_pv_kwh"]
    case["exported_pv_kwh"] = metrics["exported_pv_kwh"]
    case["payback_years"] = (
        case["battery_cost"] / case["annual_savings"]
        if case["battery_cost"] > 0 and case["annual_savings"] > 0
        else math.inf
    )
    case["annual_cost_without_battery"] = metrics["daily_cost_without_battery"]
    case["annual_cost_with_battery"] = metrics["daily_cost_with_battery"]


def optimize_battery_size_with_annual_history(
    base_scenario: Scenario,
    historical_shortwave_w_m2: np.ndarray,
    historical_clock_hours: list[int],
    battery_sizes_kwh: tuple[float, ...] = MVP_BATTERY_SIZES_KWH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    historical_load = generate_repeated_load_profile(
        base_scenario.average_daily_consumption_kwh,
        historical_clock_hours,
        base_scenario.load_profile_type,
        base_scenario.custom_load_shape,
    )
    historical_pv = simulate_pv_generation(historical_shortwave_w_m2, base_scenario)
    cases = [
        run_dispatch_case(base_scenario, size, historical_pv, historical_load, historical_clock_hours)
        for size in battery_sizes_kwh
    ]

    for case in cases:
        apply_full_period_annual_metrics(case)

    add_blended_scores(cases)
    recommended = select_recommended_case(cases)

    return cases, recommended


def optimize_battery_size_with_multi_year_history(
    base_scenario: Scenario,
    historical_years: list[dict[str, list[Any]]],
    battery_sizes_kwh: tuple[float, ...] = MVP_BATTERY_SIZES_KWH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not historical_years:
        raise RuntimeError("No historical years were provided for multi-year economics.")

    cases: list[dict[str, Any]] = []
    for size in battery_sizes_kwh:
        yearly_cases = []
        for historical in historical_years:
            historical_shortwave = np.array(historical["shortwave_radiation_w_m2"], dtype=float)
            historical_clock_hours = extract_clock_hours(list(historical["time"]))
            historical_load = generate_repeated_load_profile(
                base_scenario.average_daily_consumption_kwh,
                historical_clock_hours,
                base_scenario.load_profile_type,
                base_scenario.custom_load_shape,
            )
            historical_pv = simulate_pv_generation(historical_shortwave, base_scenario)
            year_case = run_dispatch_case(base_scenario, size, historical_pv, historical_load, historical_clock_hours)
            apply_full_period_annual_metrics(year_case)
            year_case["year"] = int(historical["year"])
            yearly_cases.append(year_case)

        annual_savings_values = np.array([case["metrics"]["daily_savings"] for case in yearly_cases], dtype=float)
        average_annual_savings = float(np.mean(annual_savings_values))
        p50_annual_savings = float(np.percentile(annual_savings_values, 50))
        p90_annual_savings = float(np.percentile(annual_savings_values, 10))
        metric_keys = yearly_cases[0]["metrics"].keys()
        average_metrics = {
            key: float(np.mean([case["metrics"][key] for case in yearly_cases]))
            for key in metric_keys
        }

        case = yearly_cases[0].copy()
        case["metrics"] = average_metrics
        case["yearly_annual_savings"] = [
            {"year": case_by_year["year"], "annual_savings": case_by_year["metrics"]["daily_savings"]}
            for case_by_year in yearly_cases
        ]
        case["average_annual_savings"] = average_annual_savings
        case["p50_annual_savings"] = p50_annual_savings
        case["p90_annual_savings"] = p90_annual_savings
        case["annual_savings"] = max(0.0, p50_annual_savings)
        case["daily_savings"] = p50_annual_savings / ANNUALIZATION_DAYS
        case["daily_cost_without_battery"] = average_metrics["daily_cost_without_battery"] / ANNUALIZATION_DAYS
        case["daily_cost_with_battery"] = average_metrics["daily_cost_with_battery"] / ANNUALIZATION_DAYS
        case["grid_import_kwh"] = average_metrics["grid_import_kwh"]
        case["no_battery_exported_pv_kwh"] = average_metrics["no_battery_exported_pv_kwh"]
        case["exported_pv_kwh"] = average_metrics["exported_pv_kwh"]
        case["outage_backup_hours"] = average_metrics["outage_backup_hours"]
        case["annual_cost_without_battery"] = average_metrics["daily_cost_without_battery"]
        case["annual_cost_with_battery"] = average_metrics["daily_cost_with_battery"]
        case["payback_years"] = (
            case["battery_cost"] / case["annual_savings"]
            if case["battery_cost"] > 0 and case["annual_savings"] > 0
            else math.inf
        )
        cases.append(case)

    add_blended_scores(cases)
    recommended = select_recommended_case(cases)

    return cases, recommended


def plot_dispatch(
    times: list[str],
    pv_generation_kwh: np.ndarray,
    load_kwh: np.ndarray,
    dispatch: dict[str, np.ndarray | float],
    scenario: Scenario,
    filename: str = "brasilia_dispatch_24h.png",
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename
    hours = np.arange(len(times))
    soc = np.asarray(dispatch["soc"], dtype=float)
    grid_import = np.asarray(dispatch["grid_import"], dtype=float)
    battery_discharge = np.asarray(dispatch["battery_discharge"], dtype=float)

    fig, ax_left = plt.subplots(figsize=(12, 6))
    ax_left.plot(hours, pv_generation_kwh, label="PV generation", linewidth=2.2, color="#f59e0b")
    ax_left.plot(hours, load_kwh, label="Load", linewidth=2.2, color="#2563eb")
    ax_left.plot(hours, grid_import, label="Grid import", linewidth=2.2, color="#dc2626")
    ax_left.bar(hours, battery_discharge, label="Battery discharge", color="#16a34a", alpha=0.22)
    ax_left.set_ylabel("Energy per hour (kWh)")
    ax_left.set_xlabel("Local forecast hour")
    ax_left.set_xticks(hours[::2])
    ax_left.set_xticklabels([datetime.fromisoformat(times[index]).strftime("%H:%M") for index in hours[::2]])
    ax_left.grid(True, alpha=0.22)

    ax_right = ax_left.twinx()
    ax_right.plot(hours, soc, label="Battery SOC", linewidth=2.5, color="#111827", linestyle="--")
    ax_right.set_ylabel("Battery state of charge (kWh)")
    ax_right.set_ylim(0, max(scenario.battery_capacity_kwh, float(np.max(soc))) * 1.15)

    left_handles, left_labels = ax_left.get_legend_handles_labels()
    right_handles, right_labels = ax_right.get_legend_handles_labels()
    ax_left.legend(left_handles + right_handles, left_labels + right_labels, loc="upper left")

    fig.suptitle(
        f"BESSAi Next 24h Forecast Window - {scenario.location_name} - "
        f"{scenario.battery_capacity_kwh:g} kWh battery"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return output_path


def print_metrics(metrics: dict[str, float], chart_path: Path, scenario: Scenario, forecast_source: str) -> None:
    print("BESSAi deterministic dispatch simulation")
    print(f"Location: {scenario.location_name}")
    print(f"Forecast source: {forecast_source}")
    print(f"PV size: {scenario.pv_size_kwp:.1f} kWp")
    print(f"Battery size: {scenario.battery_capacity_kwh:.1f} kWh")
    print(f"Daily load: {metrics['total_load_kwh']:.2f} kWh")
    print(f"PV generation: {metrics['total_pv_kwh']:.2f} kWh")
    print(f"Grid import: {metrics['grid_import_kwh']:.2f} kWh")
    print(f"PV exported: {metrics['exported_pv_kwh']:.2f} kWh")
    print(f"Battery charged: {metrics['battery_charged_kwh']:.2f} kWh")
    print(f"Battery discharged: {metrics['battery_discharged_kwh']:.2f} kWh")
    print(f"Daily cost without battery: {scenario.currency} {metrics['daily_cost_without_battery']:.2f}")
    print(f"Daily cost with battery: {scenario.currency} {metrics['daily_cost_with_battery']:.2f}")
    print(f"Representative 24h savings: {scenario.currency} {metrics['daily_savings']:.2f}")
    print(f"Final battery SOC: {metrics['final_soc_kwh']:.2f} kWh")
    print(f"Outage backup available at end of day: {metrics['outage_backup_hours']:.1f} hours")
    print(f"Protected reserve backup target: {metrics['reserve_backup_hours']:.1f} hours")
    print(f"Dispatch chart: {chart_path}")


def format_payback(payback_years: float) -> str:
    return f"{payback_years:.1f}" if math.isfinite(payback_years) else "n/a"


def print_optimization_table(cases: list[dict[str, Any]], scenario: Scenario) -> None:
    print("\nBattery size comparison")
    print("Annualised estimate means annualised estimate from current 24h forecast window.")
    print(
        "Size kWh | Grid kWh | Cost no batt | Cost w/ batt | 24h savings | "
        "Annualised est. | Batt cost | Payback | Backup h | Score"
    )
    print("-" * 122)

    for case in cases:
        print(
            f"{case['battery_size_kwh']:8.1f} | "
            f"{case['grid_import_kwh']:8.2f} | "
            f"{scenario.currency} {case['daily_cost_without_battery']:9.2f} | "
            f"{scenario.currency} {case['daily_cost_with_battery']:10.2f} | "
            f"{scenario.currency} {case['daily_savings']:9.2f} | "
            f"{scenario.currency} {case['annual_savings']:13.2f} | "
            f"{scenario.currency} {case['battery_cost']:8.2f} | "
            f"{format_payback(case['payback_years']):>7} | "
            f"{case['outage_backup_hours']:8.1f} | "
            f"{case['final_score']:5.3f}"
        )


def print_recommendation(recommended: dict[str, Any], scenario: Scenario, chart_path: Path) -> None:
    size = recommended["battery_size_kwh"]
    payback = format_payback(recommended["payback_years"])
    payback_threshold = PAYBACK_RECOMMENDATION_THRESHOLD_YEARS
    if recommended["passes_payback_threshold"]:
        selection_reason = (
            f"This size has the strongest blended score ({recommended['final_score']:.3f}) "
            f"among batteries that pass the {payback_threshold:g}-year payback threshold."
        )
    else:
        selection_reason = (
            f"This size has the strongest blended score ({recommended['final_score']:.3f}) "
            f"overall, but no battery size passed the {payback_threshold:g}-year payback threshold."
        )

    print("\nRecommended battery size")
    print(f"{size:g} kWh")
    print(selection_reason)
    print(
        f"Its annualised estimate from the current 24h forecast window is "
        f"{scenario.currency} {recommended['annual_savings']:.0f}/year, "
        f"with payback of {payback} years and "
        f"{recommended['outage_backup_hours']:.1f} hours of end-of-day backup."
    )
    if recommended["passes_payback_threshold"]:
        print(f"Payback threshold: passed ({payback} years <= {payback_threshold:g} years).")
    else:
        print(f"Payback threshold: not passed. Treat this as financially weak unless resilience is the priority.")
    print(
        "The score blends annualised savings, backup resilience capped at 8 hours, "
        "and payback quality using the MVP weights."
    )
    print(f"Recommended dispatch chart: {chart_path}")


def run_optimized_forecast_mode(
    scenario: Scenario,
    times: list[str],
    shortwave_radiation_w_m2: np.ndarray,
    mode_name: str,
    chart_filename: str,
) -> dict[str, Any]:
    clock_hours = extract_clock_hours(times)
    load = generate_load_profile(
        scenario.average_daily_consumption_kwh,
        clock_hours,
        scenario.load_profile_type,
        scenario.custom_load_shape,
    )
    pv_generation = simulate_pv_generation(shortwave_radiation_w_m2, scenario)
    cases, recommended = optimize_battery_size(scenario, pv_generation, load, clock_hours)
    recommended_scenario = recommended["scenario"]
    chart_path = plot_dispatch(
        times,
        pv_generation,
        load,
        recommended["dispatch"],
        recommended_scenario,
        filename=chart_filename,
    )

    return {
        "mode_name": mode_name,
        "times": times,
        "clock_hours": clock_hours,
        "load": load,
        "pv_generation": pv_generation,
        "cases": cases,
        "recommended": recommended,
        "recommended_scenario": recommended_scenario,
        "chart_path": chart_path,
    }


def run_fixed_battery_forecast_mode(
    scenario: Scenario,
    times: list[str],
    shortwave_radiation_w_m2: np.ndarray,
    battery_size_kwh: float,
    mode_name: str,
) -> dict[str, Any]:
    clock_hours = extract_clock_hours(times)
    load = generate_load_profile(
        scenario.average_daily_consumption_kwh,
        clock_hours,
        scenario.load_profile_type,
        scenario.custom_load_shape,
    )
    pv_generation = simulate_pv_generation(shortwave_radiation_w_m2, scenario)
    case = run_dispatch_case(scenario, battery_size_kwh, pv_generation, load, clock_hours)

    return {
        "mode_name": mode_name,
        "times": times,
        "load": load,
        "pv_generation": pv_generation,
        "case": case,
    }


def plot_dispatch_comparison(
    raw_result: dict[str, Any],
    selected_result: dict[str, Any],
    battery_size_kwh: float,
    selected_label: str = "Selected forecast",
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"brasilia_raw_vs_selected_dispatch_{battery_size_kwh:g}kwh.png"
    times = raw_result["times"]
    x = np.arange(len(times))

    raw_case = raw_result["case"]
    selected_case = selected_result["case"]
    raw_grid = np.asarray(raw_case["dispatch"]["grid_import"], dtype=float)
    selected_grid = np.asarray(selected_case["dispatch"]["grid_import"], dtype=float)
    raw_soc = np.asarray(raw_case["dispatch"]["soc"], dtype=float)
    selected_soc = np.asarray(selected_case["dispatch"]["soc"], dtype=float)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax_top.plot(x, raw_result["pv_generation"], label="Raw PV generation", linewidth=2.2, color="#f59e0b")
    ax_top.plot(
        x,
        selected_result["pv_generation"],
        label=f"{selected_label} PV generation",
        linewidth=2.2,
        color="#2563eb",
    )
    ax_top.plot(x, raw_grid, label="Raw grid import", linewidth=2.0, color="#dc2626", linestyle="--")
    ax_top.plot(x, selected_grid, label=f"{selected_label} grid import", linewidth=2.0, color="#16a34a", linestyle="--")
    ax_top.set_ylabel("Energy per hour (kWh)")
    ax_top.grid(True, alpha=0.22)
    ax_top.legend(loc="upper left")

    ax_bottom.plot(x, raw_soc, label="Raw forecast SOC", linewidth=2.2, color="#7c3aed")
    ax_bottom.plot(x, selected_soc, label=f"{selected_label} SOC", linewidth=2.2, color="#111827")
    ax_bottom.set_ylabel("Battery SOC (kWh)")
    ax_bottom.set_xlabel("Local forecast hour")
    ax_bottom.set_xticks(x[::2])
    ax_bottom.set_xticklabels([datetime.fromisoformat(times[index]).strftime("%H:%M") for index in x[::2]])
    ax_bottom.grid(True, alpha=0.22)
    ax_bottom.legend(loc="upper left")

    fig.suptitle(f"BESSAi Raw vs {selected_label} Dispatch - {battery_size_kwh:g} kWh battery")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return output_path


def print_raw_vs_corrected_comparison(
    raw_result: dict[str, Any],
    selected_result: dict[str, Any],
    scenario: Scenario,
    battery_size_kwh: float,
    chart_path: Path,
) -> None:
    print("\nRaw vs selected forecast dispatch comparison")
    print(f"Battery size compared: {battery_size_kwh:g} kWh")
    print("Mode         | PV gen | Grid import | 24h savings | Final SOC | Backup h")
    print("--------------------------------------------------------------------------")

    for result in (raw_result, selected_result):
        case = result["case"]
        metrics = case["metrics"]
        print(
            f"{result['mode_name']:<12} | "
            f"{metrics['total_pv_kwh']:6.2f} | "
            f"{metrics['grid_import_kwh']:11.2f} | "
            f"{scenario.currency} {metrics['daily_savings']:8.2f} | "
            f"{metrics['final_soc_kwh']:9.2f} | "
            f"{metrics['outage_backup_hours']:8.1f}"
        )

    print(f"Comparison chart: {chart_path}")


def run_brasilia_demo() -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    scenario = Scenario()
    forecast_source = "Open-Meteo"
    forecast = fetch_open_meteo_forecast(scenario)

    shortwave = np.array(forecast["shortwave_radiation_w_m2"], dtype=float)
    clock_hours = extract_clock_hours(forecast["time"])
    load = generate_load_profile(
        scenario.average_daily_consumption_kwh,
        clock_hours,
        scenario.load_profile_type,
        scenario.custom_load_shape,
    )
    pv_generation = simulate_pv_generation(shortwave, scenario)
    cases, recommended = optimize_battery_size(scenario, pv_generation, load, clock_hours)
    recommended_scenario = recommended["scenario"]
    chart_path = plot_dispatch(
        forecast["time"],
        pv_generation,
        load,
        recommended["dispatch"],
        recommended_scenario,
        filename=f"brasilia_dispatch_recommended_{recommended_scenario.battery_capacity_kwh:g}kwh.png",
    )

    print_metrics(recommended["metrics"], chart_path, recommended_scenario, forecast_source)
    print_optimization_table(cases, scenario)
    print_recommendation(recommended, scenario, chart_path)

    return cases, recommended, chart_path


def run_brasilia_raw_and_corrected_demo(corrected_forecast: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], Path]:
    scenario = Scenario()
    times = list(corrected_forecast["time"])
    raw_shortwave = np.array(corrected_forecast["raw_shortwave_radiation_w_m2"], dtype=float)
    selected_shortwave = np.array(corrected_forecast["selected_shortwave_radiation_w_m2"], dtype=float)
    selected_label = "Raw forecast + ML uncertainty"

    raw_optimized = run_optimized_forecast_mode(
        scenario,
        times,
        raw_shortwave,
        "Raw",
        "brasilia_dispatch_raw_forecast.png",
    )
    selected_optimized = run_optimized_forecast_mode(
        scenario,
        times,
        selected_shortwave,
        selected_label,
        "brasilia_dispatch_selected_forecast.png",
    )
    battery_size = selected_optimized["recommended"]["battery_size_kwh"]
    raw_fixed = run_fixed_battery_forecast_mode(scenario, times, raw_shortwave, battery_size, "Raw")
    selected_fixed = run_fixed_battery_forecast_mode(
        scenario,
        times,
        selected_shortwave,
        battery_size,
        selected_label,
    )
    comparison_chart = plot_dispatch_comparison(raw_fixed, selected_fixed, battery_size, selected_label)

    print("BESSAi deterministic dispatch simulation")
    print(f"Location: {scenario.location_name}")
    print(f"Forecast source: {corrected_forecast['forecast_source']}")
    print(f"ML training source: {corrected_forecast.get('training_source', 'unknown')}")
    print(f"ML training rows: {corrected_forecast.get('training_rows', 'unknown')}")
    print(f"ML training period: {corrected_forecast.get('training_period', 'unknown')}")
    print(f"Mean forecast uncertainty: {corrected_forecast.get('mean_forecast_uncertainty_w_m2', 'unknown')} W/m2")
    print(f"Max forecast uncertainty: {corrected_forecast.get('max_forecast_uncertainty_w_m2', 'unknown')} W/m2")
    print(f"ML uncertainty note: {corrected_forecast.get('ml_correction_reason', 'unknown')}")
    print(f"Model note: {corrected_forecast['model_note']}")
    print("\nRaw forecast mode recommendation")
    print_recommendation(raw_optimized["recommended"], scenario, raw_optimized["chart_path"])
    print("\nSelected forecast mode recommendation")
    print_recommendation(selected_optimized["recommended"], scenario, selected_optimized["chart_path"])
    print_raw_vs_corrected_comparison(raw_fixed, selected_fixed, scenario, battery_size, comparison_chart)

    return raw_optimized, selected_optimized, comparison_chart
