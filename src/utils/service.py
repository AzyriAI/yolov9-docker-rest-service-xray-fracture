import os
import cv2
import random, string
import mimetypes
from pathlib import Path
from fastapi import HTTPException, Response

SERVER_HOST = os.getenv('SERVER_HOST')
SERVICE = os.getenv('SERVICE')
OUT_PUT = os.getenv('OUT_PUT')

def random_string(length):
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(length))

def image_to_url(image):
  # Generate random strint
  image_name = random_string(20) + '.png'  
  try:
    
    # Write image into localfolder
    # Production: return raw image or store it in a secure fs
    cv2.imwrite(OUT_PUT + image_name, image)

    # Return result for future reference
    return SERVER_HOST + SERVICE + image_name

  except Exception as e:
    print('Error during image upload:', e)


def prediction(filename: str):
    # Define the path to the file
    file_path = Path(f"{OUT_PUT}{filename}")

    # Check if the file exists
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    # Determine the file's MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    
    # Read the file content
    try:
        with file_path.open("rb") as file:
            content = file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return Response(content=content, media_type=mime_type)
