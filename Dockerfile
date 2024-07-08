FROM python:3.9.12
RUN apt-get update
RUN apt-get install -y python3-opencv
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /code 
COPY ./requirements.txt /code/requirements.txt
RUN pip3 install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./app /code/app
EXPOSE {{PORT}}

CMD ["uvicorn", "app.main:server", "--host", "0.0.0.0", "--port", "3000"]