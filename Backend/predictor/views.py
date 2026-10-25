
import json
import logging
import math
from pathlib import Path

import joblib
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from .forecasting_service import (
    ForecastError,
    forecast_product_model,
)

logger = logging.getLogger(__name__)

# Project root: SupplyChain IQ/
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "dataco_logistic_pipeline.joblib"
CONFIG_PATH = MODEL_DIR / "dataco_model_config.json"

model = joblib.load(MODEL_PATH)

with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
    config = json.load(config_file)

FEATURES = config["features"]
THRESHOLD = float(config["threshold"])

CATEGORICAL_FEATURES = [
    "Type",
    "Category Name",
    "Customer Segment",
    "Department Name",
    "Market",
    "Order Country",
    "Order Region",
    "Shipping Mode",
]

NUMERIC_FEATURES = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Order Item Product Price",
    "Order Item Discount Rate",
    "order_year",
    "order_month",
    "order_day_of_week",
]


def validate_payload(payload):
    """Return a list of input validation errors."""
    errors = []

    if not isinstance(payload, dict):
        return ["Request body must be a JSON object."]

    missing = [feature for feature in FEATURES if feature not in payload]
    if missing:
        errors.append({
            "error": "Missing required features.",
            "missing_features": missing,
        })

    if missing:
        return errors

    for feature in CATEGORICAL_FEATURES:
        value = payload[feature]
        if not isinstance(value, str) or not value.strip():
            errors.append({
                "field": feature,
                "error": "Must be a non-empty string.",
            })

    numeric_values = {}

    for feature in NUMERIC_FEATURES:
        value = payload[feature]

        # JSON booleans are technically integers in Python, but aren't valid here.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append({
                "field": feature,
                "error": "Must be a JSON number.",
            })
            continue

        value = float(value)

        if not math.isfinite(value):
            errors.append({
                "field": feature,
                "error": "Must be a finite number.",
            })
            continue

        numeric_values[feature] = value

    # Additional domain checks for numeric fields.
    checks = {
        "Days for shipment (scheduled)": lambda x: x >= 0,
        "Order Item Quantity": lambda x: x > 0,
        "Order Item Product Price": lambda x: x >= 0,
        "Order Item Discount Rate": lambda x: 0 <= x <= 1,
        "order_year": lambda x: x.is_integer(),
        "order_month": lambda x: x.is_integer() and 1 <= x <= 12,
        "order_day_of_week": lambda x: x.is_integer() and 0 <= x <= 6,
    }

    for feature, check in checks.items():
        if feature in numeric_values and not check(numeric_values[feature]):
            errors.append({
                "field": feature,
                "error": "Value is outside the allowed range.",
            })

    return errors


@csrf_exempt  # Development testing only; configure protection before deployment.
@require_POST
def predict_late_delivery(request):
    try:
        payload = json.loads(request.body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse(
            {"error": "Invalid JSON request body."},
            status=400,
        )

    errors = validate_payload(payload)

    if errors:
        if len(errors) == 1 and isinstance(errors[0], str):
            return JsonResponse({"error": errors[0]}, status=400)

        if len(errors) == 1 and "missing_features" in errors[0]:
            return JsonResponse(errors[0], status=400)

        return JsonResponse({"errors": errors}, status=400)

    try:
        input_df = pd.DataFrame(
            [{feature: payload[feature] for feature in FEATURES}],
            columns=FEATURES,
        )

        probability = float(model.predict_proba(input_df)[0, 1])
        predicted_risk = int(probability >= THRESHOLD)

        return JsonResponse({
            "model": config["model_name"],
            "late_delivery_risk_probability": round(probability, 4),
            "predicted_late_delivery_risk": predicted_risk,
            "threshold": THRESHOLD,
            "interpretation": (
                "High predicted late-delivery risk"
                if predicted_risk == 1
                else "Low predicted late-delivery risk"
            ),
        })

    except Exception:
        logger.exception("Late-delivery prediction failed.")
        return JsonResponse(
            {"error": "Prediction failed due to an internal server error."},
            status=500,
        )


@csrf_exempt  # Development testing only; secure appropriately before deployment.
@require_POST
def forecast_demand(request):
    """Return an ARIMA forecast for a product model."""
    try:
        payload = json.loads(request.body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse(
            {"error": "Invalid JSON request body."},
            status=400,
        )

    if not isinstance(payload, dict):
        return JsonResponse(
            {"error": "Request body must be a JSON object."},
            status=400,
        )

    product_model_id = payload.get("product_model_id")
    horizon = payload.get("horizon", 12)

    if not isinstance(product_model_id, str) or not product_model_id.strip():
        return JsonResponse(
            {"error": "product_model_id must be a non-empty string."},
            status=400,
        )

    if isinstance(horizon, bool) or not isinstance(horizon, int):
        return JsonResponse(
            {"error": "horizon must be an integer."},
            status=400,
        )

    try:
        result = forecast_product_model(
            product_model_id=product_model_id,
            horizon=horizon,
        )
        return JsonResponse(result, status=200)

    except ForecastError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    except Exception:
        logger.exception("Demand forecasting failed.")
        return JsonResponse(
            {"error": "Forecasting failed due to an internal server error."},
            status=500,
        )

@csrf_exempt
@require_POST
def list_product_models(request):
    """Return available product model IDs for the forecasting UI."""
    try:
        from .forecasting_service import load_monthly_data

        data = load_monthly_data()
        product_ids = sorted(
            data["PRODUCT_MODEL_ID"].astype(str).unique().tolist()
        )

        return JsonResponse({
            "product_models": product_ids,
            "count": len(product_ids),
        })

    except ForecastError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    except Exception:
        logger.exception("Unable to list product models.")
        return JsonResponse(
            {"error": "Unable to load product models."},
            status=500,
        )
