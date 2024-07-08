import cv2
import requests
import numpy as np

from src.yolov9.utils.augmentations import letterbox

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
        files = []
        files.append(path)

        #images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        ni = len(files)

        self.img_size = img_size
        self.stride = stride
        self.files = files
        self.nf = ni # number of files
        self.mode = 'image'
        self.auto = auto
        self.transforms = transforms  # optional
        self.vid_stride = vid_stride  # video frame-rate stride
        self.cap = None
        assert self.nf > 0, f'No images or videos found in {path}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        # Read image
        self.count += 1
        #im0 = cv2.imread(path)  # BGR
        response = requests.get(path)
        image_data = np.frombuffer(response.content, np.uint8)
        im0 = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        assert im0 is not None, f'Image Not Found {path}'
        s = f'image {self.count}/{self.nf} {path}: '

        im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        return path, im, im0, self.cap, s

    def __len__(self):
        return self.nf  # number of files
