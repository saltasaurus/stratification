import uuid

import pytest
from convection.gateway.app import create_app
from fastapi.testclient import TestClient


def make_client(score=0.9, n_features=2, threshold=0.5, calls=None):
    def fake_predict(features):
        if calls is not None:
            calls.append(features)
        return score

    return TestClient(create_app(fake_predict, "test-v1", n_features=n_features, threshold=threshold))


def test_returns_201_and_full_response():
    r = make_client(score=0.9).post("/predictions", json={"features": [0.0, 1.0]})
    assert r.status_code == 201
    body = r.json()
    assert set(body) == {"id", "defect", "score", "model_version"}
    assert body["defect"] is True
    assert body["score"] == pytest.approx(0.9)
    assert body["model_version"] == "test-v1"


def test_id_is_uuid_and_unique_per_request():
    client = make_client()
    ids = {client.post("/predictions", json={"features": [0, 1]}).json()["id"] for _ in range(5)}
    assert len(ids) == 5
    for i in ids:
        uuid.UUID(i)


@pytest.mark.parametrize(("score", "expected"), [(0.49, False), (0.5, True), (0.51, True)])
def test_threshold_boundary_is_inclusive(score, expected):
    r = make_client(score=score).post("/predictions", json={"features": [0, 0]})
    assert r.json()["defect"] is expected


def test_custom_threshold():
    r = make_client(score=0.3, threshold=0.2).post("/predictions", json={"features": [0, 0]})
    assert r.json()["defect"] is True


def test_passes_features_through_to_predict():
    calls = []
    make_client(calls=calls).post("/predictions", json={"features": [1.5, -2.0]})
    assert calls == [[1.5, -2.0]]


@pytest.mark.parametrize("features", [[], [1.0], [1.0, 2.0, 3.0]])
def test_wrong_feature_count_is_422_and_skips_predict(features):
    calls = []
    r = make_client(calls=calls).post("/predictions", json={"features": features})
    assert r.status_code == 422
    assert f"Expected 2 features; got {len(features)}" in r.json()["detail"]
    assert calls == []


@pytest.mark.parametrize("payload", [{}, {"features": "abc"}, {"features": ["a", "b"]}, {"x": [0, 1]}])
def test_bad_payload_rejected_by_pydantic(payload):
    calls = []
    r = make_client(calls=calls).post("/predictions", json=payload)
    assert r.status_code == 422
    assert calls == []


def test_get_not_allowed_yet():
    assert make_client().get("/predictions").status_code == 405
