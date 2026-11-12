from convection.gateway.app import create_app
from model import MODEL_VERSION, predict

app = create_app(predict, MODEL_VERSION, n_features=2)