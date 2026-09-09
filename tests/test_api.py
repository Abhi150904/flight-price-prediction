from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


MODEL_ARTIFACT = Path("models/final_price_artifacts.joblib")


@pytest.mark.skipif(not MODEL_ARTIFACT.exists(), reason="model artifact has not been generated")
def test_api_health_and_predict() -> None:
    from api.main import app

    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}

    payload = {
        "route": "ATL-BOS",
        "lead_time_days": 14,
        "seatsRemaining": 4,
        "segment_count": 1,
        "travel_duration_minutes": 150,
        "elapsedDays": 0,
        "distance_imputed_by_route_stop": 947,
        "isBasicEconomy": True,
        "isNonStop": True,
        "missing_total_travel_distance": False,
        "first_airline_code": "DL",
        "first_cabin_code": "coach",
        "departure_period": "morning",
        "arrival_period": "morning",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["estimated_fare"] > 0
    assert body["likely_range"]["lower"] <= body["estimated_fare"]
    assert body["likely_range"]["upper"] >= body["estimated_fare"]
    assert body["price_category"] in {"LOW", "TYPICAL", "HIGH"}
