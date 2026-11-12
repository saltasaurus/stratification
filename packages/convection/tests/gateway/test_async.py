import threading
import time
import uuid

import pytest
from convection.gateway.app import create_app
from convection.gateway.store import InMemoryStore
from fastapi.testclient import TestClient


def make_client(predict=lambda f: 0.9, store=None, threshold=0.5):
    return TestClient(create_app(predict, "test-v1", n_features=2, threshold=threshold, store=store))


def poll(client, id, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/predictions/{id}").json()
        if body["status"] != "pending":
            return body
        time.sleep(0.01)
    pytest.fail(f"prediction {id} still pending after {timeout}s")


# --- POST hands off work ---

def test_post_returns_202_pending_without_score():
    gate = threading.Event()
    c = make_client(predict=lambda f: gate.wait() or 0.9)
    r = c.post("/predictions", json={"features": [0, 1]})
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "pending"
    assert body["score"] is None and body["defect"] is None
    assert body["model_version"] == "test-v1"
    uuid.UUID(body["id"])
    gate.set()


def test_post_does_not_block_on_slow_predict():
    gate = threading.Event()
    c = make_client(predict=lambda f: gate.wait() or 0.9)
    start = time.monotonic()
    c.post("/predictions", json={"features": [0, 1]})
    assert time.monotonic() - start < 0.5
    gate.set()


def test_get_shows_pending_while_worker_busy():
    gate = threading.Event()
    c = make_client(predict=lambda f: gate.wait() or 0.9)
    id = c.post("/predictions", json={"features": [0, 1]}).json()["id"]
    assert c.get(f"/predictions/{id}").json()["status"] == "pending"
    gate.set()
    assert poll(c, id)["status"] == "done"


# --- worker completes jobs under the same id ---

def test_poll_until_done_has_score_and_same_id():
    c = make_client(predict=lambda f: 0.9)
    id = c.post("/predictions", json={"features": [0, 1]}).json()["id"]
    body = poll(c, id)
    assert body == {"id": id, "status": "done", "defect": True, "score": pytest.approx(0.9), "model_version": "test-v1"}


@pytest.mark.parametrize(("score", "expected"), [(0.49, False), (0.5, True)])
def test_threshold_applied_by_worker(score, expected):
    c = make_client(predict=lambda f: score)
    id = c.post("/predictions", json={"features": [0, 0]}).json()["id"]
    assert poll(c, id)["defect"] is expected


def test_worker_receives_request_features():
    seen = []
    c = make_client(predict=lambda f: seen.append(f) or 0.1)
    id = c.post("/predictions", json={"features": [1.5, -2.0]}).json()["id"]
    poll(c, id)
    assert seen == [[1.5, -2.0]]


def test_worker_does_not_create_extra_records():
    s = InMemoryStore()
    c = make_client(store=s)
    id = c.post("/predictions", json={"features": [0, 1]}).json()["id"]
    poll(c, id)
    assert list(s._data) == [id]


def test_many_jobs_all_complete_in_order():
    c = make_client(predict=lambda f: f[0] / 10)
    ids = [c.post("/predictions", json={"features": [i, 0]}).json()["id"] for i in range(10)]
    scores = [poll(c, i)["score"] for i in ids]
    assert scores == pytest.approx([i / 10 for i in range(10)])


# --- failures ---

def test_predict_exception_marks_failed():
    def boom(f):
        raise RuntimeError("model crashed")

    c = make_client(predict=boom)
    id = c.post("/predictions", json={"features": [0, 1]}).json()["id"]
    body = poll(c, id)
    assert body["status"] == "failed"
    assert body["score"] is None and body["defect"] is None


def test_worker_survives_a_failed_job():
    def flaky(f):
        if f[0] == 1:
            raise RuntimeError("bad input")
        return 0.9

    c = make_client(predict=flaky)
    bad = c.post("/predictions", json={"features": [1, 0]}).json()["id"]
    good = c.post("/predictions", json={"features": [0, 0]}).json()["id"]
    assert poll(c, bad)["status"] == "failed"
    assert poll(c, good)["status"] == "done"


# --- unchanged behavior from steps 1 and 2 ---

@pytest.mark.parametrize("features", [[], [1.0], [1.0, 2.0, 3.0]])
def test_wrong_feature_count_is_422_and_nothing_queued(features):
    s = InMemoryStore()
    r = make_client(store=s).post("/predictions", json={"features": features})
    assert r.status_code == 422
    assert f"Expected 2 features; got {len(features)}" in r.json()["detail"]
    assert s._data == {}


@pytest.mark.parametrize("payload", [{}, {"features": "abc"}, {"x": [0, 1]}])
def test_bad_payload_is_422(payload):
    assert make_client().post("/predictions", json=payload).status_code == 422


def test_unknown_id_is_404():
    assert make_client().get("/predictions/nope").status_code == 404


def test_store_isolation():
    s1, s2 = InMemoryStore(), InMemoryStore()
    id = make_client(store=s1).post("/predictions", json={"features": [0, 1]}).json()["id"]
    assert s2.get(id) is None
