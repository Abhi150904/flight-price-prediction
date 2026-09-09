from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, computed_field


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.price_context import apply_price_context
from src.uncertainty import apply_residual_intervals


ARTIFACT_PATH = ROOT / "models/final_price_artifacts.joblib"


class FarePredictionRequest(BaseModel):
    route: Literal["ATL-BOS", "LAX-JFK"]
    lead_time_days: int = Field(ge=1, le=60)
    seatsRemaining: int = Field(ge=0, le=10)
    segment_count: int = Field(ge=1, le=3)
    travel_duration_minutes: int = Field(ge=1)
    elapsedDays: int = Field(ge=0, le=2)
    distance_imputed_by_route_stop: float = Field(gt=0)
    isBasicEconomy: bool
    isNonStop: bool
    missing_total_travel_distance: bool = False
    first_airline_code: str = Field(min_length=2, max_length=2)
    first_cabin_code: Literal["coach", "premium coach", "business", "first"]
    departure_period: Literal["morning", "afternoon", "evening", "night"]
    arrival_period: Literal["morning", "afternoon", "evening", "night"]

    @computed_field
    @property
    def startingAirport(self) -> str:
        return self.route.split("-")[0]

    @computed_field
    @property
    def destinationAirport(self) -> str:
        return self.route.split("-")[1]


class FarePredictionResponse(BaseModel):
    estimated_fare: float
    likely_range: dict[str, float]
    typical_fare: float
    low_threshold: float
    high_threshold: float
    price_category: Literal["LOW", "TYPICAL", "HIGH"]
    comparable_count: int
    notes: list[str]


app = FastAPI(
    title="Airfare Intelligence API",
    version="0.1.0",
    description=(
        "Predicts Expedia-observed airfare for selected routes and provides "
        "training-data price context. This API does not guarantee future fares."
    ),
)

artifacts = joblib.load(ARTIFACT_PATH)
model = artifacts["model"]
features = artifacts["features"]
price_context = artifacts["price_context"]
price_context_global_quantiles = artifacts["price_context_global_quantiles"]
residual_intervals = artifacts["residual_intervals"]
residual_global_interval = artifacts["residual_global_interval"]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=FarePredictionResponse)
def predict(request: FarePredictionRequest) -> FarePredictionResponse:
    row = pd.DataFrame([request.model_dump()])
    prediction = float(model.predict(row[features])[0])
    row["totalFare"] = prediction

    context = apply_price_context(
        row,
        price_context,
        price_context_global_quantiles,
    ).iloc[0]

    interval_input = row.copy()
    interval_input["actual"] = prediction
    interval_input["prediction"] = prediction
    interval = apply_residual_intervals(
        interval_input,
        residual_intervals,
        residual_global_interval,
    ).iloc[0]

    notes = [
        "Price category compares the estimate with similar fares observed in training data.",
        "Likely range is based on empirical residual intervals and is less reliable for rare premium-cabin cases.",
        "This endpoint estimates observed fare context; it does not guarantee future price movement.",
    ]

    return FarePredictionResponse(
        estimated_fare=round(prediction, 2),
        likely_range={
            "lower": round(float(interval["predicted_lower"]), 2),
            "upper": round(float(interval["predicted_upper"]), 2),
        },
        typical_fare=round(float(context["typical_fare"]), 2),
        low_threshold=round(float(context["low_threshold"]), 2),
        high_threshold=round(float(context["high_threshold"]), 2),
        price_category=context["price_category"],
        comparable_count=int(context["comparable_count"]),
        notes=notes,
    )
