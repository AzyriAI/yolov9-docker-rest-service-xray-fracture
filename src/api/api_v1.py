import logging
logger = logging.getLogger("uvicorn")

from pydantic import BaseModel
from src.utils import service
from src.yolov9 import predict

class DataImagePredict(BaseModel):
    imageUrl: str
    user_id: str
    chat_id: str

def init_api_v1(server, model):
  logger.info("AI Server - init api v1")
  
  @server.post("/v1/fracture/predict")
  async def api_fracture_predict(data : DataImagePredict):
    logger.info("/v1/fracture/predict")
    image = predict.inference(model=model,source=data.imageUrl)
    imageUrl = service.image_to_url(image, data.user_id, data.chat_id)
    return { "status" : "success", "payload" : imageUrl}
