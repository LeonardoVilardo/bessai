"""Run the BESSAi Brasilia raw vs ML-corrected forecast demo."""

from pathlib import Path
import sys

ML_DIR = Path(__file__).resolve().parents[1] / "ml"
sys.path.insert(0, str(ML_DIR))

from core import run_brasilia_raw_and_corrected_demo
from forecast_error import get_corrected_shortwave_forecast


if __name__ == "__main__":
    run_brasilia_raw_and_corrected_demo(get_corrected_shortwave_forecast())
