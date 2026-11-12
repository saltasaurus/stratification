from .schemas import PredictionResponse


class InMemoryStore:

    def __init__(self):
        self._data = {}
        self._key2id = {}
    def save_key(self, key, id):
        self._key2id[key] = id

    def get_id_for_key(self, key) -> str | None:
        return self._key2id.get(key, None)

    def save(self, pred: PredictionResponse) -> None:
        self._data[pred.id] = pred

    def get(self, id: str) -> PredictionResponse | None:
        return self._data.get(id, None)