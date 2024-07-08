import torch
from pathlib import Path
from PIL import Image

from src.yolov9.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages
from src.yolov9.utils.general import (Profile, increment_path, non_max_suppression, scale_boxes, strip_optimizer, xyxy2xywh, cv2)
from src.yolov9.utils.plots import Annotator, colors

import logging
LOGGER = logging.getLogger("uvicorn")

def inference(model,
        source='',
        imgsz=(640, 640),  # inference size (height, width)
        max_det=1000,  # maximum detections per image
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        iou_thres=0.45,  # NMS IOU threshold
        agnostic_nms=False,  # class-agnostic NMS
        line_thickness=2,  # bounding box thickness (pixels)
        augment=False,  # augmented inference
        conf_thres=0.25,  # confidence threshold
        view_img=False,  # show results
        save_dir='runs/detect',
        save_txt=False,
        webcam=False,
        save_conf=False,  # save confidences in --save-txt labels
        save_img=True,
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        vid_stride=1,  # video frame-rate stride
    ):
  
    device = torch.device('cpu')
    result = None
    stride, names, pt = model.stride, model.names, model.pt
  
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen = 0
    windows = []
    dt = (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im, augment=augment, visualize=False)
            pred = pred[0][1]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            #save_path = str(save_dir / p.name)  # im.jpg
            #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_img or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()
            # Save results (image with detections)
            
            cv2.imwrite('./out.png', im0)
            print('im0 type', type(im0))
            return Image.fromarray(im0)


        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    #if save_txt or save_img:
    #    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    #if update:
    #    strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

    if result:
        print('some result')
    else:
        print('NONE')
    return result