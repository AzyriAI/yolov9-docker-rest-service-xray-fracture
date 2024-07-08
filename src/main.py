import logging
import torch
logger = logging.getLogger("uvicorn")

from src.utils import config
from src.api.server import start
from src.api.api_v1 import init_api_v1
from src.yolov9.model import load_model

# Load settings
parameters = config.load()

# Load AI model
model = load_model(parameters)

# Start API server
server = start()

# Init API endpoints
init_api_v1(server, model)
