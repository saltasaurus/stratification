import threading
import time

import pytest
from convection.gateway.app import create_app
from convection.gateway.store import InMemoryStore
from fastapi.testclient import TestClient


class CountingPredict:
    def __init__(self, score=0.9):
        self.calls = 0
        self.score = score
        self._lock = threading.Lock()

    def __call__(self, features):
        with self._lock:
            self.calls += 1
        return self.score


class SlowKeyStore(InMemoryStore):
    """Widens the check-then-act window so a missing lock shows up reliably."""

    def get_id_for_key(self, key):
        result = super().get_id_for_key(key)
        time.sleep(0.05)
        return result


def make_client(predict=None, store=None):
    predict = predict or CountingPredict()
    return TestClient(create_app(predict, "test-v1", n_features=2, store=store))


def post(client, key=None, features=(0, 1)):
    headers = {"Idempotency-Key": key} if key is not None else {}
    return client.post("/predictions", json={"features": list(features)}, headers=headers)


def poll(client, id, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/predictions/{id}").json()
        if body["status"] != "pending":
            return body
        time.sleep(0.01)
    pytest.fail(f"prediction {id} still pending after {timeout}s")


def test_first_keyed_request_returns_202():
    r = post(make_client(), key="abc")
    assert r.status_code == 202
    assert r.json()["status"] == "pending"


def test_repeat_key_returns_200_and_same_id():
    c = make_client()
    first = post(c, key="abc").json()
    r = post(c, key="abc")
    assert r.status_code == 200
    assert r.json()["id"] == first["id"]


def test_repeat_after_done_returns_finished_record():
    c = make_client()
    id = post(c, key="abc").json()["id"]
    done = poll(c, id)
    r = post(c, key="abc")
    assert r.status_code == 200
    assert r.json() == done


def test_repeat_key_calls_predict_once():
    predict = CountingPredict()
    c = make_client(predict=predict)
    id = post(c, key="abc").json()["id"]
    poll(c, id)
    for _ in range(3):
        post(c, key="abc")
    time.sleep(0.1)
    assert predict.calls == 1


def test_no_key_gives_new_id_each_time():
    predict = CountingPredict()
    c = make_client(predict=predict)
    ids = {post(c).json()["id"] for _ in range(3)}
    assert len(ids) == 3
    for id in ids:
        poll(c, id)
    assert predict.calls == 3


def test_different_keys_get_different_ids():
    c = make_client()
    assert post(c, key="a").json()["id"] != post(c, key="b").json()["id"]


def test_key_saved_in_store():
    store = InMemoryStore()
    c = make_client(store=store)
    id = post(c, key="abc").json()["id"]
    assert store.get_id_for_key("abc") == id
    assert store.get_id_for_key("missing") is None


def test_invalid_request_does_not_reserve_key():
    store = InMemoryStore()
    c = make_client(store=store)
    assert post(c, key="abc", features=(1, 2, 3)).status_code == 422
    assert store.get_id_for_key("abc") is None
    assert post(c, key="abc").status_code == 202


def test_key_pointing_to_missing_record_returns_500():
    store = InMemoryStore()
    store.save_key("abc", "ghost-id")
    c = TestClient(create_app(CountingPredict(), "test-v1", n_features=2, store=store), raise_server_exceptions=False)
    assert post(c, key="abc").status_code == 500


def test_concurrent_same_key_makes_one_prediction():
    predict = CountingPredict()
    c = make_client(predict=predict, store=SlowKeyStore())
    results = []
    barrier = threading.Barrier(5)

    def send():
        barrier.wait()
        results.append(post(c, key="race"))

    threads = [threading.Thread(target=send) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ids = {r.json()["id"] for r in results}
    assert len(ids) == 1
    assert sorted(r.status_code for r in results) == [200, 200, 200, 200, 202]
    poll(c, ids.pop())
    time.sleep(0.1)
    assert predict.calls == 1
