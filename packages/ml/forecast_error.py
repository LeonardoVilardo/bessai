"""Forecast-error regression diagnostics for BESSAi.

The model trains on matched Open-Meteo previous model runs and historical
weather rows. If real matched rows are unavailable or invalid, diagnostics are
reported as unavailable instead of falling back to synthetic training data.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HISTORICAL_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
OPEN_METEO_HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
CACHE_DIR = OUTPUT_DIR / "cache"
BUNDLED_CACHE_DIR = Path(__file__).resolve().parent / "data" / "cache"
PREVIOUS_RUNS_START_DATE = date(2024, 1, 1)
PREVIOUS_RUNS_LEAD_DAYS = 1
DEFAULT_PREVIOUS_RUNS_PAST_DAYS = 365
DEFAULT_ARTIFACT_START_DATE = date(2024, 1, 1)
DEFAULT_ARTIFACT_END_DATE = date(2025, 12, 31)
SEASONAL_UNCERTAINTY_WINDOW_WEEKS = 2
MIN_IRRADIANCE_FOR_UNCERTAINTY_W_M2 = 50.0
FORECAST_CACHE_TTL_SECONDS = int(os.getenv("BESSAI_FORECAST_CACHE_TTL_SECONDS", "3600"))
FORECAST_CACHE_MAX_STALE_SECONDS = int(os.getenv("BESSAI_FORECAST_CACHE_MAX_STALE_SECONDS", str(7 * 24 * 3600)))
HTTP_HEADERS = {
    "User-Agent": "BESSAi portfolio demo (https://github.com/leovilardo/BESSAi)",
}
REAL_PREVIOUS_RUNS_MODEL_NOTE = (
    "Forecast-error model trained on matched Open-Meteo previous-day forecast "
    "runs and historical weather data. Annual economics remain deterministic and separate."
)
REAL_HISTORICAL_FORECAST_MODEL_NOTE = (
    "Forecast-error model trained on matched Open-Meteo archived forecast and "
    "historical weather data. Annual economics remain deterministic and separate."
)
UNAVAILABLE_MODEL_NOTE = (
    "Forecast-error model unavailable because real matched forecast-error training data "
    "could not be loaded or validated. Dispatch still uses the raw Open-Meteo forecast."
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
    runtime_cache_path = forecast_cache_path(scenario, hours, "runtime")
    cached = load_cached_forecast(runtime_cache_path, max_age_seconds=FORECAST_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached

    params = {
        "latitude": scenario.latitude,
        "longitude": scenario.longitude,
        "hourly": "shortwave_radiation,cloud_cover,temperature_2m",
        "timezone": scenario.timezone,
        "forecast_hours": hours,
    }
    url = f"{OPEN_METEO_URL}?{urlencode(params)}"

    try:
        with urlopen(Request(url, headers=HTTP_HEADERS), timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        fallback = load_cached_forecast(
            runtime_cache_path,
            max_age_seconds=FORECAST_CACHE_MAX_STALE_SECONDS,
            fallback_reason=str(exc),
        )
        if fallback is not None:
            return fallback

        bundled = load_cached_forecast(
            forecast_cache_path(scenario, hours, "bundled"),
            fallback_reason=str(exc),
        )
        if bundled is not None:
            return bundled

        raise RuntimeError(
            "Open-Meteo live forecast request failed and no cached Open-Meteo forecast "
            f"was available for this location. Upstream reason: {exc}"
        ) from exc

    hourly = payload.get("hourly", {})
    times = hourly.get("time")
    radiation = hourly.get("shortwave_radiation")
    cloud_cover = hourly.get("cloud_cover")
    temperature = hourly.get("temperature_2m")

    if not times or not radiation or not cloud_cover or not temperature:
        raise RuntimeError("Open-Meteo response did not include required hourly forecast variables.")

    forecast = {
        "time": times[:hours],
        "raw_forecast_w_m2": radiation[:hours],
        "cloud_cover_pct": cloud_cover[:hours],
        "temperature_2m_c": temperature[:hours],
        "forecast_source": "Open-Meteo live forecast",
    }
    save_cached_forecast(runtime_cache_path, forecast)
    return forecast


def forecast_cache_path(
    scenario: ForecastScenario,
    hours: int = 24,
    target: str = "runtime",
) -> Path:
    root = BUNDLED_CACHE_DIR if target == "bundled" else CACHE_DIR
    return root / f"open_meteo_forecast_{safe_scenario_name(scenario)}_{hours}h.json"


def save_cached_forecast(path: Path, forecast: dict[str, list[Any] | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "forecast": forecast,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_cached_forecast(
    path: Path,
    max_age_seconds: int | None = None,
    fallback_reason: str | None = None,
) -> dict[str, list[Any]] | None:
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        fetched_at_raw = str(payload["fetched_at_utc"])
        fetched_at = datetime.fromisoformat(fetched_at_raw)
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=UTC)
        age_seconds = (datetime.now(UTC) - fetched_at).total_seconds()
        if max_age_seconds is not None and age_seconds > max_age_seconds:
            return None

        forecast = dict(payload["forecast"])
        validate_cached_forecast(forecast)
    except Exception:
        return None

    fetched_label = fetched_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
    if fallback_reason:
        forecast["forecast_source"] = (
            f"Open-Meteo cached forecast from {fetched_label}; live refresh unavailable "
            f"({fallback_reason})"
        )
    else:
        forecast["forecast_source"] = f"Open-Meteo cached forecast from {fetched_label}"
    return forecast


def validate_cached_forecast(forecast: dict[str, Any]) -> None:
    required = ("time", "raw_forecast_w_m2", "cloud_cover_pct", "temperature_2m_c")
    for key in required:
        value = forecast.get(key)
        if not isinstance(value, list) or not value:
            raise RuntimeError(f"Cached forecast missed {key}.")

    length = len(forecast["time"])
    if any(len(forecast[key]) < length for key in required[1:]):
        raise RuntimeError("Cached forecast arrays have inconsistent lengths.")


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


def cache_path_for_training_data(
    scenario: ForecastScenario,
    source: str = "previous_runs_day1",
    window_label: str | None = None,
) -> Path:
    window_label = window_label or previous_runs_window_label()
    safe_name = "".join(char.lower() if char.isalnum() else "_" for char in scenario.location_name).strip("_")
    return CACHE_DIR / f"open_meteo_forecast_error_{safe_name}_{source}_{window_label}.json"


def safe_scenario_name(scenario: ForecastScenario) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in scenario.location_name).strip("_")


def model_artifact_stem(scenario: ForecastScenario) -> str:
    return f"forecast_error_model_{safe_scenario_name(scenario)}"


def model_artifact_paths(scenario: ForecastScenario, target: str = "runtime") -> tuple[Path, Path]:
    root = BUNDLED_CACHE_DIR if target == "bundled" else CACHE_DIR
    stem = model_artifact_stem(scenario)
    return root / f"{stem}.pkl", root / f"{stem}.metadata.json"


def model_artifact_read_paths(scenario: ForecastScenario) -> tuple[tuple[Path, Path], ...]:
    return (
        model_artifact_paths(scenario, "runtime"),
        model_artifact_paths(scenario, "bundled"),
    )


def previous_runs_past_days(today: date | None = None) -> int:
    today = today or date.today()
    return max(1, (today - PREVIOUS_RUNS_START_DATE).days + 1)


def previous_runs_window_label(today: date | None = None) -> str:
    today = today or date.today()
    return f"{PREVIOUS_RUNS_START_DATE.isoformat()}_{today.isoformat()}"


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


def fetch_open_meteo_historical_forecast_rows(
    scenario: ForecastScenario,
    start_date: str,
    end_date: str,
) -> dict[str, dict[str, float]]:
    return fetch_open_meteo_historical_rows(
        OPEN_METEO_HISTORICAL_FORECAST_URL,
        scenario,
        start_date,
        end_date,
    )


def fetch_open_meteo_previous_run_rows(
    scenario: ForecastScenario,
    past_days: int | None = None,
    lead_days: int = PREVIOUS_RUNS_LEAD_DAYS,
) -> dict[str, dict[str, float]]:
    past_days = past_days or previous_runs_past_days()
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


def date_chunks(start_date: date, end_date: date, chunk_days: int) -> list[tuple[date, date]]:
    chunks: list[tuple[date, date]] = []
    current = start_date
    while current <= end_date:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end_date)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks


def build_historical_forecast_training_rows(
    scenario: ForecastScenario,
    start_date: date = DEFAULT_ARTIFACT_START_DATE,
    end_date: date = DEFAULT_ARTIFACT_END_DATE,
    chunk_days: int = 31,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for chunk_start, chunk_end in date_chunks(start_date, end_date, chunk_days):
        forecast_rows = fetch_open_meteo_historical_forecast_rows(
            scenario,
            chunk_start.isoformat(),
            chunk_end.isoformat(),
        )
        actual_rows = fetch_open_meteo_historical_rows(
            OPEN_METEO_HISTORICAL_WEATHER_URL,
            scenario,
            chunk_start.isoformat(),
            chunk_end.isoformat(),
        )
        rows.extend(build_matched_training_rows(forecast_rows, actual_rows))

    rows.sort(key=lambda row: str(row["time"]))
    validate_real_training_rows(rows)
    return rows


def build_previous_runs_training_rows(
    scenario: ForecastScenario,
    past_days: int = DEFAULT_PREVIOUS_RUNS_PAST_DAYS,
    lead_days: int = PREVIOUS_RUNS_LEAD_DAYS,
) -> list[dict[str, float | str]]:
    forecast_rows = fetch_open_meteo_previous_run_rows(scenario, past_days, lead_days)
    start_date, end_date = date_bounds_from_rows(forecast_rows)
    actual_rows = fetch_open_meteo_historical_rows(
        OPEN_METEO_HISTORICAL_WEATHER_URL,
        scenario,
        start_date,
        end_date,
    )
    rows = build_matched_training_rows(forecast_rows, actual_rows)
    rows.sort(key=lambda row: str(row["time"]))
    validate_real_training_rows(rows)
    return rows


def circular_week_distance(left: int, right: int, weeks_per_year: int = 53) -> int:
    distance = abs(left - right)
    return min(distance, weeks_per_year - distance)


def build_seasonal_uncertainty_summary(rows: list[dict[str, float | str]]) -> list[dict[str, float | int]]:
    daylight_errors_by_week: dict[int, list[float]] = {}
    for row in rows:
        timestamp = datetime.fromisoformat(str(row["time"]))
        week = int(timestamp.isocalendar().week)
        forecast_irradiance = float(row["forecast_irradiance"])
        actual_irradiance = float(row["actual_irradiance"])
        if max(forecast_irradiance, actual_irradiance) <= 1.0:
            continue

        error = abs(float(row["forecast_error"]))
        daylight_errors_by_week.setdefault(week, []).append(error)

    summary: list[dict[str, float | int]] = []
    for week in range(1, 54):
        errors: list[float] = []
        for source_week, source_errors in daylight_errors_by_week.items():
            if circular_week_distance(week, source_week) <= SEASONAL_UNCERTAINTY_WINDOW_WEEKS:
                errors.extend(source_errors)
        if not errors:
            continue

        values = np.array(errors, dtype=float)
        summary.append(
            {
                "week_of_year": week,
                "rows": int(values.size),
                "mae_w_m2": float(np.mean(values)),
                "p90_abs_error_w_m2": float(np.percentile(values, 90)),
                "window_weeks_each_side": SEASONAL_UNCERTAINTY_WINDOW_WEEKS,
            }
        )
    return summary


def train_model_from_rows(
    rows: list[dict[str, float | str]],
    training_source: str = "real_open_meteo_previous_runs",
) -> tuple[GradientBoostingRegressor, dict[str, Any]]:
    features, forecast_error = features_and_target_from_rows(rows)
    first_time = str(rows[0]["time"])
    last_time = str(rows[-1]["time"])
    model_note = (
        REAL_HISTORICAL_FORECAST_MODEL_NOTE
        if training_source == "real_open_meteo_historical_forecast"
        else REAL_PREVIOUS_RUNS_MODEL_NOTE
    )
    metadata: dict[str, Any] = {
        "training_source": training_source,
        "model_note": model_note,
        "training_rows": len(rows),
        "training_period": f"{first_time[:10]} to {last_time[:10]}",
        "uncertainty_basis": "rolling 5-week daylight historical absolute forecast error by week-of-year",
        "seasonal_uncertainty": build_seasonal_uncertainty_summary(rows),
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


def save_forecast_error_model_artifact(
    scenario: ForecastScenario,
    model: GradientBoostingRegressor,
    metadata: dict[str, Any],
    target: str = "runtime",
) -> tuple[Path, Path]:
    model_path, metadata_path = model_artifact_paths(scenario, target)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as file:
        pickle.dump(model, file)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return model_path, metadata_path


def load_forecast_error_model_artifact(
    scenario: ForecastScenario,
) -> tuple[GradientBoostingRegressor | None, dict[str, Any]]:
    load_errors: list[str] = []
    for model_path, metadata_path in model_artifact_read_paths(scenario):
        if not metadata_path.exists():
            continue

        metadata = json.loads(metadata_path.read_text())
        if model_path.exists():
            try:
                with model_path.open("rb") as file:
                    model = pickle.load(file)
                return model, metadata
            except Exception as exc:
                load_errors.append(f"{model_path.name}: {exc}")
                metadata["model_note"] = (
                    f"{metadata.get('model_note', REAL_PREVIOUS_RUNS_MODEL_NOTE)} "
                    "The deployed app is using the saved seasonal uncertainty metadata directly; "
                    f"model pickle load skipped because: {exc}."
                )
                metadata["training_source"] = f"{metadata.get('training_source', 'real_open_meteo_previous_runs')}_metadata"
                return None, metadata

        if metadata.get("seasonal_uncertainty"):
            metadata["model_note"] = (
                f"{metadata.get('model_note', REAL_PREVIOUS_RUNS_MODEL_NOTE)} "
                "The deployed app is using saved seasonal uncertainty metadata directly."
            )
            metadata["training_source"] = f"{metadata.get('training_source', 'real_open_meteo_previous_runs')}_metadata"
            return None, metadata

    raise RuntimeError(
        "No offline forecast-error model artifact found. Run "
        "python3 packages/ml/build_forecast_error_artifact.py first."
        + (f" Load errors: {'; '.join(load_errors)}" if load_errors else "")
    )


def build_real_training_rows(
    scenario: ForecastScenario,
    past_days: int | None = None,
    lead_days: int = PREVIOUS_RUNS_LEAD_DAYS,
) -> list[dict[str, float | str]]:
    past_days = past_days or previous_runs_past_days()
    cache_path = cache_path_for_training_data(
        scenario,
        source=f"previous_runs_day{lead_days}",
        window_label=previous_runs_window_label(),
    )
    unavailable_cache_path = cache_path.with_suffix(".unavailable.json")
    if cache_path.exists():
        rows = json.loads(cache_path.read_text())
        validate_real_training_rows(rows)
        return rows
    if unavailable_cache_path.exists():
        payload = json.loads(unavailable_cache_path.read_text())
        raise RuntimeError(payload.get("reason", "Forecast-error training data unavailable."))

    try:
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
    except Exception as exc:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        unavailable_cache_path.write_text(json.dumps({"reason": str(exc)}))
        raise

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(rows))
    if unavailable_cache_path.exists():
        unavailable_cache_path.unlink()
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
) -> tuple[GradientBoostingRegressor | None, dict[str, Any]]:
    scenario = scenario or ForecastScenario()
    return load_forecast_error_model_artifact(scenario)


def unavailable_forecast_error_result(
    forecast: dict[str, list[Any]],
    forecast_source: str,
    reason: Exception,
) -> dict[str, Any]:
    raw_forecast = np.array(forecast["raw_forecast_w_m2"], dtype=float)
    zeros = np.zeros_like(raw_forecast)
    return {
        "time": forecast["time"],
        "forecast_source": forecast_source,
        "model_note": f"{UNAVAILABLE_MODEL_NOTE} Reason: {reason}",
        "training_source": "unavailable",
        "training_rows": 0,
        "training_period": "unavailable",
        "mae_before_correction_w_m2": None,
        "mae_after_correction_w_m2": None,
        "mean_forecast_error_w_m2": None,
        "evaluation_basis": "unavailable",
        "raw_shortwave_radiation_w_m2": raw_forecast,
        "predicted_error_w_m2": zeros,
        "corrected_shortwave_radiation_w_m2": raw_forecast,
        "selected_shortwave_radiation_w_m2": raw_forecast,
        "forecast_uncertainty_w_m2": None,
        "forecast_uncertainty_mae_w_m2": None,
        "seasonal_uncertainty": [],
        "mean_forecast_uncertainty_w_m2": None,
        "max_forecast_uncertainty_w_m2": None,
        "ml_correction_applied": False,
        "ml_correction_reason": "Forecast uncertainty artifact unavailable; dispatch uses the raw Open-Meteo forecast.",
    }


def seasonal_uncertainty_arrays(
    metadata: dict[str, Any],
    times: list[str],
    raw_forecast_w_m2: np.ndarray,
) -> dict[str, np.ndarray]:
    summary = metadata.get("seasonal_uncertainty") or []
    by_week = {int(row["week_of_year"]): row for row in summary}
    daylight_rows = [row for row in summary if float(row.get("p90_abs_error_w_m2", 0.0)) > 0]
    fallback_p90 = float(np.median([float(row["p90_abs_error_w_m2"]) for row in daylight_rows])) if daylight_rows else 0.0
    fallback_mae = float(np.median([float(row["mae_w_m2"]) for row in daylight_rows])) if daylight_rows else 0.0

    p90_values: list[float] = []
    mae_values: list[float] = []
    for index, timestamp in enumerate(times):
        if raw_forecast_w_m2[index] < MIN_IRRADIANCE_FOR_UNCERTAINTY_W_M2:
            p90_values.append(0.0)
            mae_values.append(0.0)
            continue

        parsed = datetime.fromisoformat(timestamp)
        row = by_week.get(int(parsed.isocalendar().week))
        p90_values.append(float(row["p90_abs_error_w_m2"]) if row else fallback_p90)
        mae_values.append(float(row["mae_w_m2"]) if row else fallback_mae)

    return {
        "p90_uncertainty_w_m2": np.array(p90_values, dtype=float),
        "mae_uncertainty_w_m2": np.array(mae_values, dtype=float),
    }


def get_forecast_uncertainty(
    scenario: ForecastScenario | None = None,
    hours: int = 24,
) -> dict[str, Any]:
    scenario = scenario or ForecastScenario()
    forecast = fetch_open_meteo_solar_forecast(scenario, hours)
    forecast_source = str(forecast.get("forecast_source", "Open-Meteo live forecast"))

    try:
        _model, metadata = train_forecast_error_model(scenario)
    except Exception as exc:
        return unavailable_forecast_error_result(forecast, forecast_source, exc)

    raw_forecast = np.array(forecast["raw_forecast_w_m2"], dtype=float)
    uncertainty = seasonal_uncertainty_arrays(metadata, forecast["time"], raw_forecast)
    forecast_uncertainty = uncertainty["p90_uncertainty_w_m2"]
    daylight_uncertainty = forecast_uncertainty[raw_forecast >= MIN_IRRADIANCE_FOR_UNCERTAINTY_W_M2]
    if daylight_uncertainty.size == 0:
        daylight_uncertainty = forecast_uncertainty
    mean_uncertainty = float(np.mean(daylight_uncertainty))
    max_uncertainty = float(np.max(daylight_uncertainty))
    correction_applied = False
    selected_shortwave = raw_forecast
    correction_reason = (
        "Forecast uncertainty is estimated from a rolling seasonal window of historical daylight "
        f"forecast errors. Dispatch uses the raw Open-Meteo forecast. Current daylight P90 "
        f"uncertainty averages {mean_uncertainty:.1f} W/m2."
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
        "uncertainty_basis": metadata.get("uncertainty_basis", "rolling seasonal daylight historical error"),
        "raw_shortwave_radiation_w_m2": raw_forecast,
        "predicted_error_w_m2": np.zeros_like(raw_forecast),
        "corrected_shortwave_radiation_w_m2": raw_forecast,
        "selected_shortwave_radiation_w_m2": selected_shortwave,
        "forecast_uncertainty_w_m2": forecast_uncertainty,
        "forecast_uncertainty_mae_w_m2": uncertainty["mae_uncertainty_w_m2"],
        "seasonal_uncertainty": metadata.get("seasonal_uncertainty", []),
        "mean_forecast_uncertainty_w_m2": mean_uncertainty,
        "max_forecast_uncertainty_w_m2": max_uncertainty,
        "ml_correction_applied": correction_applied,
        "ml_correction_reason": correction_reason,
    }


def get_corrected_shortwave_forecast(
    scenario: ForecastScenario | None = None,
    hours: int = 24,
) -> dict[str, Any]:
    """Backward-compatible wrapper for older engine/API code."""
    return get_forecast_uncertainty(scenario, hours)


def plot_raw_vs_corrected(
    times: list[str],
    raw_forecast_w_m2: np.ndarray,
    uncertainty_w_m2: np.ndarray,
    scenario: ForecastScenario,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "brasilia_forecast_uncertainty.png"
    x = np.arange(len(times))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, raw_forecast_w_m2, label="Raw Open-Meteo forecast", linewidth=2.4, color="#f59e0b")
    ax.plot(x, uncertainty_w_m2, label="P90 forecast uncertainty", linewidth=2.4, color="#7c3aed")
    ax.set_title(f"BESSAi Solar Forecast Uncertainty Demo - {scenario.location_name}")
    ax.set_xlabel("Local forecast hour")
    ax.set_ylabel("W/m2")
    ax.set_xticks(x[::2])
    ax.set_xticklabels([datetime.fromisoformat(times[index]).strftime("%H:%M") for index in x[::2]])
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return output_path


def print_forecast_table(times: list[str], corrected: dict[str, np.ndarray]) -> None:
    print("\nForecast uncertainty table")
    print("Hour  | Raw forecast | P90 uncertainty | Mean abs error")
    print("-------------------------------------------------------")
    for index, timestamp in enumerate(times):
        hour = datetime.fromisoformat(timestamp).strftime("%H:%M")
        raw = corrected["raw_forecast_w_m2"][index]
        p90 = corrected["forecast_uncertainty_w_m2"][index]
        mae = corrected["forecast_uncertainty_mae_w_m2"][index]
        print(f"{hour} | {raw:12.1f} | {p90:15.1f} | {mae:14.1f}")


def format_optional_metric(value: Any, unit: str) -> str:
    if value is None:
        return "Unavailable"
    return f"{float(value):.1f} {unit}"


def run_forecast_error_demo() -> tuple[dict[str, np.ndarray], Path]:
    scenario = ForecastScenario()
    forecast = get_corrected_shortwave_forecast(scenario)
    corrected = {
        "raw_forecast_w_m2": forecast["raw_shortwave_radiation_w_m2"],
        "forecast_uncertainty_w_m2": forecast["forecast_uncertainty_w_m2"],
        "forecast_uncertainty_mae_w_m2": forecast["forecast_uncertainty_mae_w_m2"],
    }
    chart_path = plot_raw_vs_corrected(
        list(forecast["time"]),
        corrected["raw_forecast_w_m2"],
        corrected["forecast_uncertainty_w_m2"],
        scenario,
    )

    print("BESSAi forecast uncertainty demo")
    print(f"Location: {scenario.location_name}")
    print(f"Forecast source: {forecast['forecast_source']}")
    print(f"Training source: {forecast['training_source']}")
    print(f"Training rows: {forecast['training_rows']}")
    print(f"Training period: {forecast['training_period']}")
    print(f"Raw forecast MAE: {format_optional_metric(forecast['mae_before_correction_w_m2'], 'W/m2')}")
    print(f"Model residual MAE: {format_optional_metric(forecast['mae_after_correction_w_m2'], 'W/m2')}")
    print(f"Mean forecast error: {format_optional_metric(forecast['mean_forecast_error_w_m2'], 'W/m2')}")
    print(
        "Mean forecast uncertainty: "
        f"{format_optional_metric(forecast['mean_forecast_uncertainty_w_m2'], 'W/m2')}"
    )
    print(f"Max forecast uncertainty: {format_optional_metric(forecast['max_forecast_uncertainty_w_m2'], 'W/m2')}")
    print(f"Evaluation basis: {forecast['evaluation_basis']}")
    print(f"Uncertainty basis: {forecast.get('uncertainty_basis', 'unavailable')}")
    print(f"Uncertainty note: {forecast['ml_correction_reason']}")
    print(f"Model note: {forecast['model_note']}")
    print_forecast_table(list(forecast["time"]), corrected)
    print(f"\nForecast chart: {chart_path}")

    return corrected, chart_path
