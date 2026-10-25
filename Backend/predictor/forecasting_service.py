
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "b2b_monthly_model_demand_all_years.csv"
)

ARIMA_ORDER = (1, 1, 1)

MIN_HISTORY_MONTHS = 12
MAX_FORECAST_MONTHS = 24


class ForecastError(Exception):
    """Raised when a demand forecast cannot be produced."""


def load_monthly_data():
    """Load and validate the historical monthly order-frequency dataset."""
    if not DATA_FILE.exists():
        raise ForecastError(
            f"Forecast dataset not found: {DATA_FILE}"
        )

    try:
        data = pd.read_csv(
            DATA_FILE,
            dtype={"PRODUCT_MODEL_ID": "string"},
        )
    except (OSError, pd.errors.ParserError) as exc:
        raise ForecastError(
            "Unable to read the forecasting dataset."
        ) from exc

    required_columns = {
        "month",
        "PRODUCT_MODEL_ID",
        "order_frequency",
    }

    missing = required_columns - set(data.columns)
    if missing:
        raise ForecastError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    data["month"] = pd.to_datetime(
        data["month"], errors="coerce"
    )
    data["order_frequency"] = pd.to_numeric(
        data["order_frequency"], errors="coerce"
    )

    if (
        data["month"].isna().any()
        or data["PRODUCT_MODEL_ID"].isna().any()
        or data["order_frequency"].isna().any()
    ):
        raise ForecastError(
            "Dataset contains missing or invalid required values."
        )

    if (data["order_frequency"] < 0).any():
        raise ForecastError(
            "Dataset contains negative order-frequency values."
        )

    if data.duplicated(["month", "PRODUCT_MODEL_ID"]).any():
        raise ForecastError(
            "Dataset contains duplicate month/product combinations."
        )

    return data


def forecast_product_model(product_model_id, horizon=12):
    """
    Forecast monthly order frequency for one known product model.

    Returns historical metadata and non-negative point forecasts.
    """
    if not isinstance(product_model_id, str):
        raise ForecastError(
            "product_model_id must be a string."
        )

    product_model_id = product_model_id.strip()
    if not product_model_id:
        raise ForecastError(
            "product_model_id cannot be empty."
        )

    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise ForecastError(
            "horizon must be an integer."
        )

    if not 1 <= horizon <= MAX_FORECAST_MONTHS:
        raise ForecastError(
            f"horizon must be between 1 and {MAX_FORECAST_MONTHS}."
        )

    data = load_monthly_data()

    product_data = data[
        data["PRODUCT_MODEL_ID"] == product_model_id
    ].copy()

    if product_data.empty:
        raise ForecastError(
            f"Unknown product model ID: {product_model_id}"
        )

    product_data = product_data.sort_values("month")

    # Use history only from the first observed month onward.
    first_month = product_data["month"].min()
    last_month = product_data["month"].max()

    history = (
        product_data
        .set_index("month")["order_frequency"]
        .sort_index()
        .asfreq("MS", fill_value=0)
    )

    history = history.astype(float)

    if len(history) < MIN_HISTORY_MONTHS:
        raise ForecastError(
            f"Insufficient history: {len(history)} months available; "
            f"at least {MIN_HISTORY_MONTHS} are required."
        )

    if not np.isfinite(history.to_numpy()).all():
        raise ForecastError(
            "Historical series contains non-finite values."
        )

    try:
        fitted_model = ARIMA(
            history,
            order=ARIMA_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()

        predicted = np.asarray(
            fitted_model.forecast(steps=horizon),
            dtype=float,
        )

    except Exception as exc:
        raise ForecastError(
            "ARIMA fitting or forecasting failed for this product model."
        ) from exc

    if (
        len(predicted) != horizon
        or not np.isfinite(predicted).all()
    ):
        raise ForecastError(
            "ARIMA returned invalid forecast values."
        )

    # Order frequency cannot be negative. Values remain continuous
    # expected frequencies rather than rounded integer counts.
    predicted = np.maximum(predicted, 0.0)

    forecast_dates = pd.date_range(
        start=last_month + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )

    return {
        "product_model_id": product_model_id,
        "model": f"ARIMA{ARIMA_ORDER}",
        "target": "monthly_order_frequency",
        "history_start": first_month.strftime("%Y-%m"),
        "history_end": last_month.strftime("%Y-%m"),
        "history_months": len(history),
        "forecast_horizon_months": horizon,
        "forecast": [
            {
                "month": month.strftime("%Y-%m"),
                "predicted_order_frequency": round(float(value), 3),
            }
            for month, value in zip(forecast_dates, predicted)
        ],
        "limitations": [
            "Forecasts estimate monthly order frequency, not physical unit demand.",
            "Historical first-observed month is not a verified product launch date.",
            "Point forecasts are estimates and are not guaranteed outcomes.",
        ],
    }
