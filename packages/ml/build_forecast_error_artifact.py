"""Build offline Open-Meteo forecast-error model artifacts for BESSAi."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from packages.ml.forecast_error import (  # noqa: E402
    DEFAULT_ARTIFACT_END_DATE,
    DEFAULT_ARTIFACT_START_DATE,
    DEFAULT_PREVIOUS_RUNS_PAST_DAYS,
    ForecastScenario,
    build_historical_forecast_training_rows,
    build_previous_runs_training_rows,
    save_forecast_error_model_artifact,
    train_model_from_rows,
)


SUPPORTED_LOCATIONS = {
    "brasilia": ForecastScenario(
        location_name="Brasilia, Brazil",
        latitude=-15.826016,
        longitude=-47.812539,
        timezone="America/Sao_Paulo",
    ),
    "west_bahia": ForecastScenario(
        location_name="West of Bahia, Brazil",
        latitude=-13.792761,
        longitude=-46.104032,
        timezone="America/Sao_Paulo",
    ),
}


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build cached BESSAi forecast-error model artifacts from matched "
            "Open-Meteo historical forecast and historical weather rows."
        )
    )
    parser.add_argument(
        "--location",
        choices=("all", *SUPPORTED_LOCATIONS.keys()),
        default="all",
        help="Location artifact to build. Default: all.",
    )
    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=DEFAULT_ARTIFACT_START_DATE,
        help=(
            "First training date for --source historical-forecast. "
            f"Default: {DEFAULT_ARTIFACT_START_DATE.isoformat()}."
        ),
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        default=DEFAULT_ARTIFACT_END_DATE,
        help=(
            "Last training date for --source historical-forecast. "
            f"Default: {DEFAULT_ARTIFACT_END_DATE.isoformat()}."
        ),
    )
    parser.add_argument(
        "--source",
        choices=("previous-runs", "historical-forecast"),
        default="previous-runs",
        help=(
            "previous-runs uses true previous-day forecast archive rows; "
            "historical-forecast uses date-range archived forecast rows and may be too close to reanalysis."
        ),
    )
    parser.add_argument(
        "--past-days",
        type=int,
        default=DEFAULT_PREVIOUS_RUNS_PAST_DAYS,
        help=f"Past-days window for --source previous-runs. Default: {DEFAULT_PREVIOUS_RUNS_PAST_DAYS}.",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=31,
        help="API date-range chunk size. Smaller chunks are slower but less fragile.",
    )
    parser.add_argument(
        "--target",
        choices=("runtime", "bundled"),
        default="runtime",
        help=(
            "runtime writes outputs/cache for local use; bundled writes "
            "packages/ml/data/cache for deployable demo assets."
        ),
    )
    return parser.parse_args()


def selected_locations(location: str) -> list[tuple[str, ForecastScenario]]:
    if location == "all":
        return list(SUPPORTED_LOCATIONS.items())
    return [(location, SUPPORTED_LOCATIONS[location])]


def build_artifacts(args: argparse.Namespace) -> int:
    if args.source == "historical-forecast" and args.end_date < args.start_date:
        raise ValueError("--end-date must be on or after --start-date.")
    if args.past_days < 1:
        raise ValueError("--past-days must be positive.")
    if args.chunk_days < 1:
        raise ValueError("--chunk-days must be positive.")

    failures: list[str] = []
    print("BESSAi forecast-error artifact builder")
    if args.source == "previous-runs":
        print("Source: Open-Meteo Previous Runs matched to Historical Weather")
        print(f"Past days: {args.past_days}")
    else:
        print("Source: Open-Meteo Historical Forecast matched to Historical Weather")
        print(f"Training period: {args.start_date.isoformat()} to {args.end_date.isoformat()}")
        print(f"Chunk size: {args.chunk_days} days")
    print(f"Target: {args.target}")

    for location_key, scenario in selected_locations(args.location):
        print(f"\n{scenario.location_name} [{location_key}]")
        try:
            if args.source == "previous-runs":
                rows = build_previous_runs_training_rows(
                    scenario,
                    past_days=args.past_days,
                )
                training_source = "real_open_meteo_previous_runs"
            else:
                rows = build_historical_forecast_training_rows(
                    scenario,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    chunk_days=args.chunk_days,
                )
                training_source = "real_open_meteo_historical_forecast"
            print(f"  matched rows: {len(rows):,}")
            model, metadata = train_model_from_rows(rows, training_source=training_source)
            model_path, metadata_path = save_forecast_error_model_artifact(
                scenario,
                model,
                metadata,
                target=args.target,
            )
            print(f"  raw MAE: {metadata['mae_before_correction_w_m2']:.1f} W/m2")
            print(f"  residual MAE: {metadata['mae_after_correction_w_m2']:.1f} W/m2")
            print(f"  wrote model: {model_path}")
            print(f"  wrote metadata: {metadata_path}")
        except Exception as exc:
            failures.append(f"{scenario.location_name}: {exc}")
            print(f"  failed: {exc}")

    if failures:
        print("\nArtifact build completed with failures:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nArtifact build completed successfully.")
    return 0


def main() -> None:
    raise SystemExit(build_artifacts(parse_args()))


if __name__ == "__main__":
    main()
