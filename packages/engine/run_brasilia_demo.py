"""Run the BESSAi Brasilia dispatch demo with forecast uncertainty diagnostics."""

from pathlib import Path
import sys

ML_DIR = Path(__file__).resolve().parents[1] / "ml"
sys.path.insert(0, str(ML_DIR))

from core import run_brasilia_raw_and_corrected_demo
from forecast_error import get_forecast_uncertainty


if __name__ == "__main__":
    run_brasilia_raw_and_corrected_demo(get_forecast_uncertainty())
