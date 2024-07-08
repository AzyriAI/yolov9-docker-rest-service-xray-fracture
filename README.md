# yolov9-rest-service-xray-fracture
Production ready YoloV9 REST Service for x-ray fracture detection

# Running locally

## Clone this repo

``` shell
git clone https://github.com/AzyriAI/yolov9-docker-rest-service-xray-fracture.git
cd yolov9-docker-rest-service-xray-fracture
```

## Install requirements 

``` shell
pip install -r requirements.txt
```

## Download pre-trained weights
In case you didn't train your own model you can download this train. Weights are from best model after 100 epochs.

It tooks me 12 hours to train yolov9 for frature detection (1 GPU)

https://www.kaggle.com/datasets/karelbecerra/yolov9-weights-x-ray-fracture-detection


``` shell
unzip best.pt.zip
mv best.pt ./weights 
```

## Set up environment variables
You can find enviroment variables in env.sample

``` shell
export MODEL_STORAGE=./weights/
export SERVER_HOST=http://localhost:3100
export OUT_PUT=./prediction/
export SERVICE=/v1/fracture/prediction/
```

# Run your server

``` shell
uvicorn src.main:server --host 0.0.0.0 --port 3100
```

You shoud see a similar output
``` shell
Fusing layers... 
yolov9-c summary: 962 layers, 51018070 parameters, 0 gradients, 239.0 GFLOPs
INFO:     Azyri AI Server - starting
INFO:     AI Server - init api v1
INFO:     Started server process [61457]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:3100 (Press CTRL+C to quit)
```

# Test your prediction
Open a new terminal. Server expects url as input and returns local url as output.

Example with fra1.png:
``` shell
curl -X POST http://localhost:3100/v1/fracture/predict \
-H "Content-Type: application/json" \
-d '{"imageUrl": "https://github.com/AzyriAI/ai-ml-dl-samples/blob/main/data/fracture/fra1.png?raw=true"}'
```

Example with fra2.png:
``` shell
curl -X POST http://localhost:3100/v1/fracture/predict \
-H "Content-Type: application/json" \
-d '{"imageUrl": "https://github.com/AzyriAI/ai-ml-dl-samples/blob/main/data/fracture/fra2.png?raw=true"}'
```

You can find some x-rays to test your own local server in https://github.com/AzyriAI/ai-ml-dl-samples/tree/main/data/fracture

# Check your result
After running your prediction and if everything went well you should get a response similar to this

``` shell
{"status":"success","payload":"http://localhost:3100/v1/fracture/prediction/edf66m4r0tj2rsjmknou.png"}
```

You have to options
1- Open your browser and paste the result url
2- Execute wget command

``` shell
wget http://localhost:3100/v1/fracture/prediction/edf66m4r0tj2rsjmknou.png
```

# References
Yolov9
https://github.com/WongKinYiu/yolov9
