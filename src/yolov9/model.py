import os
import sys
import torch
import numpy as np
import logging
logger = logging.getLogger("uvicorn")

from src.yolov9.models.common import DetectMultiBackend

MODEL_STORAGE = os.getenv('MODEL_STORAGE')

def load_model(parameters):
  device = torch.device('cpu')
  model = DetectMultiBackend(MODEL_STORAGE + parameters.model, device=device, fp16=parameters.half)

  return model