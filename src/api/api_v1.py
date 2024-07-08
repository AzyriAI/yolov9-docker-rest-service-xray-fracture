import logging
logger = logging.getLogger("uvicorn")

from pydantic import BaseModel
from src.utils import service
from src.yolov9 import predict

class DataImagePredict(BaseModel):
    imageUrl: str

def init_api_v1(server, model):
  logger.info("AI Server - init api v1")
  
  # Input: A remote url containing the x-ray image
  #        Example: https://github.com/AzyriAI/ai-ml-dl-samples/blob/main/data/fracture/fra1.png?raw=true
  #
  # Output: Returns a local url
  #        Example: http://localhost:3100/v1/fracture/prediction/ro44gpsizt2w0jxtcnvt.png
  @server.post("/v1/fracture/predict")
  async def api_fracture_predict(data : DataImagePredict):
    logger.info("/v1/fracture/predict")
    image = predict.inference(model=model,source=data.imageUrl)
    imageUrl = service.image_to_url(image)
    return { "status" : "success", "payload" : imageUrl}

  # Input: Local url
  #        Example: http://localhost:3100/v1/fracture/prediction/ro44gpsizt2w0jxtcnvt.png
  #
  # Output: Binary image
  @server.get("/v1/fracture/prediction/{filename}")
  async def prediction(filename: str):
    logger.info(f"/v1/fracture/prediction/{filename}")
    return service.prediction(filename=filename)