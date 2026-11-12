from model import MODEL_VERSION, predict
from outcrop.gateway.app import create_app

app = create_app(predict, MODEL_VERSION, n_features=2)