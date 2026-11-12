from .schemas import PredictionResponse


class InMemoryStore:

    def __init__(self):
        self._data = {}

    def save(self, pred: PredictionResponse) -> None:
        self._data[pred.id] = pred

    def get(self, id: str) -> PredictionResponse | None:
        return self._data.get(id, None)