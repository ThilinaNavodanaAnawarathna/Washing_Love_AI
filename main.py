import io
import uvicorn
import numpy as np
import nest_asyncio
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import cv2
from typing import List
from numpy import ndarray
from typing import Tuple
from PIL import Image
import base64
from fastapi import Response
import os
from ultralytics import YOLO
import unicodedata
import re


class Detection:
    def __init__(self,
                 model_path: str,
                 classes: List[str]
                 ):
        self.model_path = model_path
        self.classes = classes
        self.model = self.__load_model()

    def __load_model(self) -> cv2.dnn_Net:
        net = cv2.dnn.readNet(self.model_path)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    def __extract_ouput(self,
                        preds: ndarray,
                        image_shape: Tuple[int, int],
                        input_shape: Tuple[int, int],
                        score: float = 0.1,
                        nms: float = 0.0,
                        confidence: float = 0.0
                        ) -> dict[list, list, list]:
        class_ids, confs, boxes = list(), list(), list()

        image_height, image_width = image_shape
        input_height, input_width = input_shape
        x_factor = image_width / input_width
        y_factor = image_height / input_height

        rows = preds[0].shape[0]
        for i in range(rows):
            row = preds[0][i]
            conf = row[4]

            classes_score = row[4:]
            _, _, _, max_idx = cv2.minMaxLoc(classes_score)
            class_id = max_idx[1]
            # print(classes_score[class_id])
            if (classes_score[class_id] > score):
                confs.append(conf)
                label = self.classes[int(class_id)]
                class_ids.append(label)

                # extract boxes
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

        r_class_ids, r_confs, r_boxes = list(), list(), list()
        indexes = cv2.dnn.NMSBoxes(boxes, confs, confidence, nms)
        for i in indexes:
            r_class_ids.append(class_ids[i])
            r_confs.append(confs[i] * 100)
            r_boxes.append(boxes[i].tolist())

        return r_class_ids

    def __call__(self,
                 image: ndarray,
                 width: int = 640,
                 height: int = 640,
                 score: float = 0.1,
                 nms: float = 0.0,
                 confidence: float = 0.0
                 ) -> dict[list, list, list]:

        blob = cv2.dnn.blobFromImage(
            image, 1 / 255.0, (width, height),
            swapRB=True, crop=False
        )
        self.model.setInput(blob)
        preds = self.model.forward()
        preds = preds.transpose((0, 2, 1))

        # extract output
        results = self.__extract_ouput(
            preds=preds,
            image_shape=image.shape[:2],
            input_shape=(height, width),
            score=score,
            nms=nms,
            confidence=confidence
        )
        return results


detection = Detection(
    model_path='best.onnx',
    classes=['damaged door', 'damaged window', 'damaged headlight', 'damaged mirror', 'dent', 'damaged hood',
             'damaged bumper', 'damaged wind shield']
)

app = FastAPI()

_entity_re = re.compile(r"&([^;]+);")
_filename_ascii_strip_re = re.compile(r"[^A-Za-z0-9_.-]")
_windows_device_files = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(10)),
    *(f"LPT{i}" for i in range(10)),
}


def secure_filename(filename: str) -> str:
    r"""Pass it a filename and it will return a secure version of it.  This
    filename can then safely be stored on a regular file system and passed
    to :func:`os.path.join`.  The filename returned is an ASCII only string
    for maximum portability.

    On windows systems the function also makes sure that the file is not
    named after one of the special device files.

    >>> secure_filename("My cool movie.mov")
    'My_cool_movie.mov'
    >>> secure_filename("../../../etc/passwd")
    'etc_passwd'
    >>> secure_filename('i contain cool \xfcml\xe4uts.txt')
    'i_contain_cool_umlauts.txt'

    The function might return an empty filename.  It's your responsibility
    to ensure that the filename is unique and that you abort or
    generate a random filename if the function returned an empty one.

    .. versionadded:: 0.5

    :param filename: the filename to secure
    """
    filename = unicodedata.normalize("NFKD", filename)
    filename = filename.encode("ascii", "ignore").decode("ascii")

    for sep in os.sep, os.path.altsep:
        if sep:
            filename = filename.replace(sep, " ")
    filename = str(_filename_ascii_strip_re.sub("", "_".join(filename.split()))).strip(
        "._"
    )

    # on nt a couple of special files are present in each folder.  We
    # have to ensure that the target file is not such a filename.  In
    # this case we prepend an underline
    if (
            os.name == "nt"
            and filename
            and filename.split(".")[0].upper() in _windows_device_files
    ):
        filename = f"_{filename}"

    return filename


@app.post('/detection')
def post_detection(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    image_array = np.array(image)
    image_array = image_array[:, :, ::-1].copy()

    input_image_name = secure_filename(filename=file.filename)
    input_image_path = f"images/{input_image_name}"
    image.save(input_image_path)

    model = YOLO("best.pt")
    results = model(source=input_image_path)
    res_plotted = results[0].plot()

    # Convert the OpenCV image to a base64 string
    _, buffer = cv2.imencode('.png', res_plotted)
    img_base64 = base64.b64encode(buffer).decode()

    results = detection(image_array)
    result = {
        "damageParts": results,
        "imageBase64": img_base64
    }
    return result


@app.post('/detect-damage')
async def detect_car_damage():
    img_pth = "1.jpeg"
    model = YOLO("best.pt")
    results = model(source=img_pth)
    res_plotted = results[0].plot()

    # Convert the OpenCV image to a base64 string
    _, buffer = cv2.imencode('.png', res_plotted)
    img_base64 = base64.b64encode(buffer).decode()

    result = {
        "imageBase64": img_base64
    }
    return result


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
