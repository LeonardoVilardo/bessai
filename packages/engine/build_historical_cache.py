"""Build NASA POWER historical irradiance cache for supported BESSAi demo sites."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from packages.engine.core import (  # noqa: E402
    BUNDLED_CACHE_DIR,
    CACHE_DIR,
    DEFAULT_HISTORICAL_YEARS,
    Scenario,
    fetch_nasa_power_hourly_irradiance,
    nasa_power_bundled_cache_path,
    nasa_power_cache_path,
)


SUPPORTED_LOCATIONS = {
    "brasilia": Scenario(
        location_name="Brasilia, Brazil",
        latitude=-15.826016,
        longitude=-47.812539,
        timezone="America/Sao_Paulo",
    ),
    "west_bahia": Scenario(
        location_name="West of Bahia, Brazil",
        latitude=-13.792761,
        longitude=-46.104032,
        timezone="America/Sao_Paulo",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download or copy cached NASA POWER hourly irradiance for the fixed "
            "BESSAi demo locations."
        )
    )
    parser.add_argument(
        "--location",
        choices=("all", *SUPPORTED_LOCATIONS.keys()),
        default="all",
        help="Location cache to build. Default: all.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=min(DEFAULT_HISTORICAL_YEARS),
        help="First historical year to cache.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=max(DEFAULT_HISTORICAL_YEARS),
        help="Last historical year to cache.",
    )
    parser.add_argument(
        "--target",
        choices=("runtime", "bundled"),
        default="runtime",
        help=(
            "runtime writes outputs/cache for local use; bundled writes "
            "packages/engine/data/cache for deployable demo assets."
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Rewrite the selected target cache file even if it already exists.",
    )
    return parser.parse_args()


def selected_locations(location: str) -> list[tuple[str, Scenario]]:
    if location == "all":
        return list(SUPPORTED_LOCATIONS.items())
    return [(location, SUPPORTED_LOCATIONS[location])]


def target_cache_path(scenario: Scenario, year: int, target: str) -> Path:
    if target == "bundled":
        return nasa_power_bundled_cache_path(scenario, year)
    return nasa_power_cache_path(scenario, year)


def build_cache(args: argparse.Namespace) -> int:
    years = range(args.start_year, args.end_year + 1)
    failures: list[str] = []
    target_root = BUNDLED_CACHE_DIR if args.target == "bundled" else CACHE_DIR

    print("BESSAi historical irradiance cache builder")
    print(f"Target: {args.target} ({target_root})")
    print(f"Years: {args.start_year}-{args.end_year}")

    for location_key, scenario in selected_locations(args.location):
        print(f"\n{scenario.location_name} [{location_key}]")
        for year in years:
            destination = target_cache_path(scenario, year, args.target)
            if destination.exists() and not args.refresh:
                print(f"  {year}: cached")
                continue

            try:
                data = fetch_nasa_power_hourly_irradiance(scenario, year)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(data))
                print(f"  {year}: wrote {destination.name}")
            except Exception as exc:
                failures.append(f"{scenario.location_name} {year}: {exc}")
                print(f"  {year}: failed ({exc})")

    if failures:
        print("\nCache build completed with failures:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nCache build completed successfully.")
    return 0


def main() -> None:
    raise SystemExit(build_cache(parse_args()))


if __name__ == "__main__":
    main()
