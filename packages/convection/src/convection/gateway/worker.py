import queue

from .schemas import PredictFn, PredictionResponse
from .store import InMemoryStore


def run_worker(q: queue.Queue, store: InMemoryStore, predict: PredictFn, threshold: float, model_version):
    while True:
        id, features = q.get()
        try:
            score = predict(features)
            pr = PredictionResponse(
                id=id,
                defect=score >= threshold,
                score=score,
                model_version=model_version,
                status="done"
            )

        except Exception:
            pr = PredictionResponse(
                id=id,
                defect=None,
                score=None,
                model_version=model_version,
                status="failed"
            )
        finally:
            q.task_done()
        store.save(pr)
