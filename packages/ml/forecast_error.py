"""Forecast-error regression demo for BESSAi.

The preferred path trains on matched Open-Meteo previous model runs and
historical weather rows. If real matched rows are unavailable or have no
measurable forecast error, the module falls back to synthetic demo training data
and reports that clearly.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HISTORICAL_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
CACHE_DIR = OUTPUT_DIR / "cache"
PREVIOUS_RUNS_PAST_DAYS = 90
PREVIOUS_RUNS_LEAD_DAYS = 1
REAL_MODEL_NOTE = (
    "Forecast-error model trained on matched Open-Meteo previous-day forecast "
    "runs and historical weather data. Annual economics remain deterministic and separate."
)
SYNTHETIC_MODEL_NOTE = (
    "Forecast-error model trained on synthetic demo data because historical "
    "forecast training data was unavailable. Designed to be replaced with archived forecast data."
)


@dataclass(frozen=True)
class ForecastScenario:
    location_name: str = "Brasilia, Brazil"
    latitude: float = -15.826016
    longitude: float = -47.812539
    timezone: str = "America/Sao_Paulo"


def fetch_open_meteo_solar_forecast(
    scenario: ForecastScenario,
    hours: int = 24,
) -> dict[str, list[Any]]:
    params = {
        "latitude": scenario.latitude,
        "longitude": scenario.longitude,
        "hourly": "shortwave_radiation,cloud_cover,temperature_2m",
        "timezone": scenario.timezone,
        "forecast_hours": hours,
    }
    url = f"{OPEN_METEO_URL}?{urlencode(params)}"

    with urlopen(url, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    hourly = payload.get("hourly", {})
    times = hourly.get("time")
    radiation = hourly.get("shortwave_radiation")
    cloud_cover = hourly.get("cloud_cover")
    temperature = hourly.get("temperature_2m")

    if not times or not radiation or not cloud_cover or not temperature:
        raise RuntimeError("Open-Meteo response did not include required hourly forecast variables.")

    return {
        "time": times[:hours],
        "raw_forecast_w_m2": radiation[:hours],
        "cloud_cover_pct": cloud_cover[:hours],
        "temperature_2m_c": temperature[:hours],
    }


def fallback_solar_forecast(hours: int = 24) -> dict[str, list[Any]]:
    start = datetime.now().replace(minute=0, second=0, microsecond=0)
    times = [(start + timedelta(hours=offset)).isoformat() for offset in range(hours)]
    raw_forecast = []
    cloud_cover = []
    temperature = []

    for offset in range(hours):
        hour = (start + timedelta(hours=offset)).hour
        daylight_shape = max(0.0, math.sin(math.pi * (hour - 6) / 12))
        raw_forecast.append(round(850 * daylight_shape, 1))
        cloud_cover.append(35.0)
        temperature.append(24.0)

    return {
        "time": times,
        "raw_forecast_w_m2": raw_forecast,
        "cloud_cover_pct": cloud_cover,
        "temperature_2m_c": temperature,
    }


def parse_times(times: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parsed = [datetime.fromisoformat(timestamp) for timestamp in times]
    hours = np.array([value.hour for value in parsed], dtype=float)
    day_of_year = np.array([value.timetuple().tm_yday for value in parsed], dtype=float)
    month = np.array([value.month for value in parsed], dtype=float)
    return hours, day_of_year, month


def build_features(
    forecast_w_m2: np.ndarray,
    cloud_cover_pct: np.ndarray,
    temperature_2m_c: np.ndarray,
    times: list[str],
) -> np.ndarray:
    hours, day_of_year, month = parse_times(times)
    hour_angle = 2 * np.pi * hours / 24
    year_angle = 2 * np.pi * day_of_year / 365
    return np.column_stack(
        [
            forecast_w_m2,
            forecast_w_m2**2 / 1000,
            cloud_cover_pct,
            temperature_2m_c,
            np.sin(hour_angle),
            np.cos(hour_angle),
            np.sin(year_angle),
            np.cos(year_angle),
            month,
            (forecast_w_m2 > 0).astype(float),
        ]
    )


def create_synthetic_training_data(samples: int = 2500, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    hours = rng.integers(0, 24, size=samples).astype(float)
    day_of_year = rng.integers(1, 366, size=samples)
    month = np.clip(np.ceil(day_of_year / 30.5), 1, 12).astype(int)
    clear_sky_shape = np.maximum(0.0, np.sin(np.pi * (hours - 6) / 12))
    cloud_factor = rng.uniform(0.35, 1.05, size=samples)
    cloud_cover = np.clip((1.05 - cloud_factor) * 100 + rng.normal(0, 10, size=samples), 0, 100)
    temperature = 20 + 10 * clear_sky_shape + rng.normal(0, 2, size=samples)
    forecast_solar = np.clip(900 * clear_sky_shape * cloud_factor + rng.normal(0, 35, size=samples), 0, None)
    synthetic_times = [
        datetime(2025, 1, 1, int(hour)) + timedelta(days=int(day) - 1)
        for hour, day in zip(hours, day_of_year)
    ]
    time_strings = [value.strftime("%Y-%m-%dT%H:%M") for value in synthetic_times]

    morning_bias = np.where((hours >= 7) & (hours <= 10), 30.0, 0.0)
    afternoon_bias = np.where((hours >= 14) & (hours <= 17), -45.0, 0.0)
    high_irradiance_bias = -0.06 * forecast_solar
    cloud_bias = 0.25 * (cloud_cover - 45)
    noise = rng.normal(0, 28, size=samples)
    forecast_error = high_irradiance_bias + morning_bias + afternoon_bias + cloud_bias + noise

    features = build_features(forecast_solar, cloud_cover, temperature, time_strings)
    return features, forecast_error


def cache_path_for_training_data(
    scenario: ForecastScenario,
    source: str = "previous_runs_day1",
    window_label: str = f"past_{PREVIOUS_RUNS_PAST_DAYS}_days",
) -> Path:
    safe_name = "".join(char.lower() if char.isalnum() else "_" for char in scenario.location_name).strip("_")
    return CACHE_DIR / f"open_meteo_forecast_error_{safe_name}_{source}_{window_label}.json"


def fetch_open_meteo_historical_rows(
    url: str,
    scenario: ForecastScenario,
    start_date: str,
    end_date: str,
) -> dict[str, dict[str, float]]:
    params = {
        "latitude": scenario.latitude,
        "longitude": scenario.longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "shortwave_radiation,cloud_cover,temperature_2m",
        "timezone": scenario.timezone,
    }
    with urlopen(f"{url}?{urlencode(params)}", timeout=45) as response:
        payload = json.loads(response.read().decode("utf-8"))

    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    radiation = hourly.get("shortwave_radiation", [])
    cloud_cover = hourly.get("cloud_cover", [])
    temperature = hourly.get("temperature_2m", [])
    if not times or not radiation or not cloud_cover or not temperature:
        raise RuntimeError(f"Open-Meteo response from {url} missed required hourly variables.")

    return {
        time: {
            "shortwave_radiation": float(radiation[index] or 0.0),
            "cloud_cover": float(cloud_cover[index] or 0.0),
            "temperature_2m": float(temperature[index] or 0.0),
        }
        for index, time in enumerate(times)
    }


def fetch_open_meteo_previous_run_rows(
    scenario: ForecastScenario,
    past_days: int = PREVIOUS_RUNS_PAST_DAYS,
    lead_days: int = PREVIOUS_RUNS_LEAD_DAYS,
) -> dict[str, dict[str, float]]:
    suffix = f"previous_day{lead_days}"
    params = {
        "latitude": scenario.latitude,
        "longitude": scenario.longitude,
        "hourly": (
            f"shortwave_radiation_{suffix},"
            f"cloud_cover_{suffix},"
            f"temperature_2m_{suffix}"
        ),
        "timezone": scenario.timezone,
        "past_days": past_days,
        "forecast_days": 1,
    }
    with urlopen(f"{OPEN_METEO_PREVIOUS_RUNS_URL}?{urlencode(params)}", timeout=45) as response:
        payload = json.loads(response.read().decode("utf-8"))

    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    radiation = hourly.get(f"shortwave_radiation_{suffix}", [])
    cloud_cover = hourly.get(f"cloud_cover_{suffix}", [])
    temperature = hourly.get(f"temperature_2m_{suffix}", [])
    if not times or not radiation or not cloud_cover or not temperature:
        raise RuntimeError("Open-Meteo Previous Runs response missed required hourly variables.")

    return {
        time: {
            "shortwave_radiation": float(radiation[index] or 0.0),
            "cloud_cover": float(cloud_cover[index] or 0.0),
            "temperature_2m": float(temperature[index] or 0.0),
        }
        for index, time in enumerate(times)
    }


def date_bounds_from_rows(rows: dict[str, dict[str, float]]) -> tuple[str, str]:
    if not rows:
        raise RuntimeError("No dated rows were available.")
    dates = [datetime.fromisoformat(timestamp).date() for timestamp in rows]
    return min(dates).isoformat(), max(dates).isoformat()


def build_matched_training_rows(
    forecast_rows: dict[str, dict[str, float]],
    actual_rows: dict[str, dict[str, float]],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for time in sorted(set(forecast_rows) & set(actual_rows)):
        forecast = forecast_rows[time]
        actual = actual_rows[time]
        forecast_irradiance = forecast["shortwave_radiation"]
        actual_irradiance = actual["shortwave_radiation"]
        rows.append(
            {
                "time": time,
                "forecast_irradiance": forecast_irradiance,
                "cloud_cover": forecast["cloud_cover"],
                "temperature_2m": forecast["temperature_2m"],
                "actual_irradiance": actual_irradiance,
                "forecast_error": actual_irradiance - forecast_irradiance,
            }
        )
    return rows


def validate_real_training_rows(rows: list[dict[str, float | str]]) -> None:
    if len(rows) < 500:
        raise RuntimeError(f"Only {len(rows)} matched historical rows were available.")

    errors = np.array([float(row["forecast_error"]) for row in rows], dtype=float)
    mean_abs_error = float(np.mean(np.abs(errors)))
    max_abs_error = float(np.max(np.abs(errors)))
    if mean_abs_error < 1.0 or max_abs_error < 10.0:
        raise RuntimeError(
            "Matched historical rows had no measurable forecast error "
            f"(mean absolute error {mean_abs_error:.2f} W/m2)."
        )


def build_real_training_rows(
    scenario: ForecastScenario,
    past_days: int = PREVIOUS_RUNS_PAST_DAYS,
    lead_days: int = PREVIOUS_RUNS_LEAD_DAYS,
) -> list[dict[str, float | str]]:
    cache_path = cache_path_for_training_data(
        scenario,
        source=f"previous_runs_day{lead_days}",
        window_label=f"past_{past_days}_days",
    )
    if cache_path.exists():
        rows = json.loads(cache_path.read_text())
        validate_real_training_rows(rows)
        return rows

    forecast_rows = fetch_open_meteo_previous_run_rows(scenario, past_days, lead_days)
    start_date, end_date = date_bounds_from_rows(forecast_rows)
    actual_rows = fetch_open_meteo_historical_rows(
        OPEN_METEO_HISTORICAL_WEATHER_URL,
        scenario,
        start_date,
        end_date,
    )

    rows = build_matched_training_rows(forecast_rows, actual_rows)
    validate_real_training_rows(rows)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(rows))
    return rows


def features_and_target_from_rows(rows: list[dict[str, float | str]]) -> tuple[np.ndarray, np.ndarray]:
    forecast = np.array([float(row["forecast_irradiance"]) for row in rows], dtype=float)
    cloud_cover = np.array([float(row["cloud_cover"]) for row in rows], dtype=float)
    temperature = np.array([float(row["temperature_2m"]) for row in rows], dtype=float)
    times = [str(row["time"]) for row in rows]
    target = np.array([float(row["forecast_error"]) for row in rows], dtype=float)
    return build_features(forecast, cloud_cover, temperature, times), target


def evaluate_forecast_error_model(
    features: np.ndarray,
    forecast_error: np.ndarray,
) -> dict[str, float]:
    split_index = int(features.shape[0] * 0.8)
    split_index = min(max(split_index, 1), features.shape[0] - 1)
    train_features = features[:split_index]
    train_target = forecast_error[:split_index]
    holdout_features = features[split_index:]
    holdout_target = forecast_error[split_index:]

    holdout_model = GradientBoostingRegressor(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        random_state=7,
    )
    holdout_model.fit(train_features, train_target)
    predicted_error = holdout_model.predict(holdout_features)
    daylight_mask = holdout_features[:, 0] > 1.0
    if np.any(daylight_mask):
        evaluation_target = holdout_target[daylight_mask]
        evaluation_prediction = predicted_error[daylight_mask]
    else:
        evaluation_target = holdout_target
        evaluation_prediction = predicted_error

    return {
        "mae_before_correction_w_m2": float(np.mean(np.abs(evaluation_target))),
        "mae_after_correction_w_m2": float(np.mean(np.abs(evaluation_target - evaluation_prediction))),
        "mean_forecast_error_w_m2": float(np.mean(forecast_error)),
        "evaluation_basis": "last 20% daylight holdout hours",
    }


def train_forecast_error_model(
    scenario: ForecastScenario | None = None,
) -> tuple[GradientBoostingRegressor, dict[str, Any]]:
    scenario = scenario or ForecastScenario()
    try:
        rows = build_real_training_rows(scenario)
        features, forecast_error = features_and_target_from_rows(rows)
        first_time = str(rows[0]["time"])
        last_time = str(rows[-1]["time"])
        metadata = {
            "training_source": "real_open_meteo_previous_runs",
            "model_note": REAL_MODEL_NOTE,
            "training_rows": len(rows),
            "training_period": f"{first_time[:10]} to {last_time[:10]}",
        }
    except Exception as exc:
        features, forecast_error = create_synthetic_training_data()
        metadata = {
            "training_source": "synthetic_fallback",
            "model_note": f"{SYNTHETIC_MODEL_NOTE} Fallback reason: {exc}",
            "training_rows": int(features.shape[0]),
            "training_period": "synthetic demo sample",
        }

    metadata.update(evaluate_forecast_error_model(features, forecast_error))

    model = GradientBoostingRegressor(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        random_state=7,
    )
    model.fit(features, forecast_error)
    return model, metadata


def correct_forecast(
    model: GradientBoostingRegressor,
    times: list[str],
    raw_forecast_w_m2: list[float],
    cloud_cover_pct: list[float],
    temperature_2m_c: list[float],
) -> dict[str, np.ndarray]:
    raw_forecast = np.array(raw_forecast_w_m2, dtype=float)
    cloud_cover = np.array(cloud_cover_pct, dtype=float)
    temperature = np.array(temperature_2m_c, dtype=float)
    features = build_features(raw_forecast, cloud_cover, temperature, times)
    predicted_error = model.predict(features)
    predicted_error = np.where(raw_forecast <= 1.0, 0.0, predicted_error)
    corrected_forecast = np.clip(raw_forecast + predicted_error, 0, None)

    return {
        "raw_forecast_w_m2": raw_forecast,
        "predicted_error_w_m2": predicted_error,
        "corrected_forecast_w_m2": corrected_forecast,
    }


def get_corrected_shortwave_forecast(
    scenario: ForecastScenario | None = None,
    hours: int = 24,
) -> dict[str, Any]:
    scenario = scenario or ForecastScenario()
    forecast_source = "Open-Meteo"

    try:
        forecast = fetch_open_meteo_solar_forecast(scenario, hours)
    except Exception as exc:
        forecast_source = f"fallback sample data ({exc})"
        forecast = fallback_solar_forecast(hours)

    model, metadata = train_forecast_error_model(scenario)
    corrected = correct_forecast(
        model,
        forecast["time"],
        forecast["raw_forecast_w_m2"],
        forecast["cloud_cover_pct"],
        forecast["temperature_2m_c"],
    )
    mae_before = float(metadata["mae_before_correction_w_m2"])
    mae_after = float(metadata["mae_after_correction_w_m2"])
    forecast_uncertainty = np.abs(corrected["predicted_error_w_m2"])
    daylight_uncertainty = forecast_uncertainty[corrected["raw_forecast_w_m2"] > 1.0]
    if daylight_uncertainty.size == 0:
        daylight_uncertainty = forecast_uncertainty
    mean_uncertainty = float(np.mean(daylight_uncertainty))
    max_uncertainty = float(np.max(daylight_uncertainty))
    correction_applied = False
    selected_shortwave = corrected["raw_forecast_w_m2"]
    correction_reason = (
        "ML is used as a forecast-error uncertainty diagnostic only; dispatch uses the raw "
        f"Open-Meteo forecast. Current daylight uncertainty estimate averages {mean_uncertainty:.1f} W/m2."
    )

    return {
        "time": forecast["time"],
        "forecast_source": forecast_source,
        "model_note": metadata["model_note"],
        "training_source": metadata["training_source"],
        "training_rows": metadata["training_rows"],
        "training_period": metadata["training_period"],
        "mae_before_correction_w_m2": metadata["mae_before_correction_w_m2"],
        "mae_after_correction_w_m2": metadata["mae_after_correction_w_m2"],
        "mean_forecast_error_w_m2": metadata["mean_forecast_error_w_m2"],
        "evaluation_basis": metadata["evaluation_basis"],
        "raw_shortwave_radiation_w_m2": corrected["raw_forecast_w_m2"],
        "predicted_error_w_m2": corrected["predicted_error_w_m2"],
        "corrected_shortwave_radiation_w_m2": corrected["corrected_forecast_w_m2"],
        "selected_shortwave_radiation_w_m2": selected_shortwave,
        "forecast_uncertainty_w_m2": forecast_uncertainty,
        "mean_forecast_uncertainty_w_m2": mean_uncertainty,
        "max_forecast_uncertainty_w_m2": max_uncertainty,
        "ml_correction_applied": correction_applied,
        "ml_correction_reason": correction_reason,
    }


def plot_raw_vs_corrected(
    times: list[str],
    raw_forecast_w_m2: np.ndarray,
    corrected_forecast_w_m2: np.ndarray,
    scenario: ForecastScenario,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "brasilia_raw_vs_corrected_forecast.png"
    x = np.arange(len(times))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, raw_forecast_w_m2, label="Raw Open-Meteo forecast", linewidth=2.4, color="#f59e0b")
    ax.plot(x, corrected_forecast_w_m2, label="ML-corrected forecast", linewidth=2.4, color="#2563eb")
    ax.set_title(f"BESSAi Solar Forecast Error Demo - {scenario.location_name}")
    ax.set_xlabel("Local forecast hour")
    ax.set_ylabel("Shortwave radiation (W/m2)")
    ax.set_xticks(x[::2])
    ax.set_xticklabels([datetime.fromisoformat(times[index]).strftime("%H:%M") for index in x[::2]])
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return output_path


def print_forecast_table(times: list[str], corrected: dict[str, np.ndarray]) -> None:
    print("\nForecast correction table")
    print("Hour  | Raw forecast | Predicted error | Corrected forecast")
    print("----------------------------------------------------------")
    for index, timestamp in enumerate(times):
        hour = datetime.fromisoformat(timestamp).strftime("%H:%M")
        raw = corrected["raw_forecast_w_m2"][index]
        error = corrected["predicted_error_w_m2"][index]
        corrected_value = corrected["corrected_forecast_w_m2"][index]
        print(f"{hour} | {raw:12.1f} | {error:15.1f} | {corrected_value:18.1f}")


def run_forecast_error_demo() -> tuple[dict[str, np.ndarray], Path]:
    scenario = ForecastScenario()
    forecast = get_corrected_shortwave_forecast(scenario)
    corrected = {
        "raw_forecast_w_m2": forecast["raw_shortwave_radiation_w_m2"],
        "predicted_error_w_m2": forecast["predicted_error_w_m2"],
        "corrected_forecast_w_m2": forecast["corrected_shortwave_radiation_w_m2"],
    }
    chart_path = plot_raw_vs_corrected(
        list(forecast["time"]),
        corrected["raw_forecast_w_m2"],
        corrected["corrected_forecast_w_m2"],
        scenario,
    )

    print("BESSAi forecast-error ML demo")
    print(f"Location: {scenario.location_name}")
    print(f"Forecast source: {forecast['forecast_source']}")
    print(f"Training source: {forecast['training_source']}")
    print(f"Training rows: {forecast['training_rows']}")
    print(f"Training period: {forecast['training_period']}")
    print(f"Raw forecast MAE: {forecast['mae_before_correction_w_m2']:.1f} W/m2")
    print(f"Model residual MAE: {forecast['mae_after_correction_w_m2']:.1f} W/m2")
    print(f"Mean forecast error: {forecast['mean_forecast_error_w_m2']:.1f} W/m2")
    print(f"Mean forecast uncertainty: {forecast['mean_forecast_uncertainty_w_m2']:.1f} W/m2")
    print(f"Max forecast uncertainty: {forecast['max_forecast_uncertainty_w_m2']:.1f} W/m2")
    print(f"Evaluation basis: {forecast['evaluation_basis']}")
    print(f"ML uncertainty note: {forecast['ml_correction_reason']}")
    print(f"Model note: {forecast['model_note']}")
    print_forecast_table(list(forecast["time"]), corrected)
    print(f"\nForecast chart: {chart_path}")

    return corrected, chart_path
