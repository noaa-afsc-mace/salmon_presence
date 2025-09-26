"""
Handles detectors
"""
from abc import ABC, abstractmethod
from ultralytics import YOLO
import numpy as np
import tensorflow as tf
import json
import re

MODEL_JSON = "detection/model_info.json"

# Load dict of available detectors
with open(MODEL_JSON) as json_file:
    MODEL_DICT = json.load(json_file)


class Detector(ABC):
    """
    A class that represents an object detector. Allows for maximum code
    flexibility.
    """

    def __init__(self, detector_name, confidence_threshold=None):
        self.detector_name = detector_name
        self.label_map_path = MODEL_DICT[detector_name]["label_map_path"]
        self.saved_model_path = MODEL_DICT[detector_name]["saved_model_path"]
        self.color_mode = MODEL_DICT[detector_name]["color_mode"]
        self.model_family = MODEL_DICT[detector_name]["model_family"]
        self.label_dict = None
        if confidence_threshold:
            self.confidence_threshold = confidence_threshold
        else:
            self.confidence_threshold = MODEL_DICT[detector_name]["confidence_threshold"]

    def get_label_dict(self):
        """
        Returns a dictionary of the form:
        {
            id: class_name,
            ...
        }
        """
        return self.label_dict

    @abstractmethod
    def load_model(self):
        """
        Populates the self.detector attribute.
        """
        pass

    @abstractmethod
    def load_label_dict(self):
        """
        Loads label map dict of form: {id_num: 'Name', ...}
        """
        pass

    @abstractmethod
    def detect(self, batch):
        """
        Gets detections for an image or batch of images.

        Args:
        batch: Array of arrays of images in cv2 form.
        [image_array, image_array]. If detector does not support batching,
        should be array of length 1.

        Returns:
        boxes: [[image 1 boxes], [image 2 boxes]] Boxes should be scaled and of
            form [ymin, xmin, ymax, xmax]
        classes: [[image 1 classes], [image 2 classes]] Classes should be
            strings of class name
        confidences: [[image 1 confidences], [image 2 confidences]] Confidences
            should be floats
        """
        pass


class TFDetector(Detector):
    def __init__(self, detector_name, confidence_threshold=None):
        super().__init__(detector_name, confidence_threshold)

    def load_model(self):
        model = tf.saved_model.load(self.saved_model_path)
        self.detector = model

    def load_label_dict(self):
        label_dict = {}
        with tf.io.gfile.GFile(self.label_map_path, "r") as fid:
            label_map_string = fid.read()

        text = re.sub("[^A-Za-z0-9]+", " ", label_map_string)
        items = text.split(" ")
        for i in range(len(items)):
            if items[i] == "id":
                label_dict[int(items[i + 1])] = items[i + 3]
                i += 4

        self.label_dict = label_dict

    def detect(self, batch):
        if self.color_mode == "rgb":
            input = np.array(batch)
            input_tensor = tf.convert_to_tensor(input)
            detector_fn = self.detector.signatures["serving_default"]
            detections = detector_fn(input_tensor)
        elif self.color_mode == "gray":
            batch_gray = []
            for img in batch:
                info = np.iinfo(img.dtype)
                img = img.astype(np.float64) / info.max  # normalize the data to 0 - 1
                img = 255 * img  # Now scale by 255
                img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
                img = np.stack((img,) * 3, axis=-1)
                batch_gray.append(img)
            input = np.array(batch_gray)
            input_tensor = tf.convert_to_tensor(input)
            detector_fn = self.detector.signatures["serving_default"]
            detections = detector_fn(input_tensor)
        else:
            raise ValueError(self.color_mode, "is not a valid color mode.")

        scores = list(detections["detection_scores"])
        classes = list(detections["detection_classes"])
        boxes = list(detections["detection_boxes"])

        # Remove detections below confidence threshold
        # Assumes all arrays are ordered by confidence scores
        for i in range(len(scores)):
            slice_index = np.argmax(scores[i] < self.confidence_threshold)
            scores[i] = np.array(scores[i][:slice_index])
            classes[i] = np.array(classes[i][:slice_index])
            classes[i] = list(map(lambda x: self.label_dict[x], classes[i]))
            boxes[i] = np.array(boxes[i][:slice_index])

        return boxes, classes, scores

class YOLODetector(Detector):
    def __init__(self, detector_name, confidence_threshold=None):
        super().__init__(detector_name, confidence_threshold)

    def load_model(self):
        model = YOLO(self.saved_model_path)
        self.detector = model

    def load_label_dict(self):
        label_dict = {}
        for i, c in enumerate(MODEL_DICT[self.detector_name]['classes']):
            label_dict[i] = c

        self.label_dict = label_dict

    def detect(self, batch):
        """Boxes should be scaled and of
            form [ymin, xmin, ymax, xmax]"""
        if self.color_mode == "bgr":
            # incoming batch is in RGB, yolo expects BGR
            batch = [x[:, :, ::-1] for x in batch]
            results = self.detector(batch, conf=self.confidence_threshold, verbose=False, imgsz=768, iou=0.5, max_det=100)
            
        else:
            raise ValueError(self.color_mode, "is not a valid color mode.")

        # create list of shape (num images, data)
        scores = []
        classes = []
        boxes = []
        for r in results:
            box_xyxyn = r.boxes.cpu().numpy().xyxyn
            # swap to get yxyx
            box_xyxyn[:,[0,1,2,3]] = box_xyxyn[:, [1,0,3,2]]
            boxes.append(box_xyxyn)
            classes.append([r.names[int(x)] for x in r.boxes.cpu().cls])
            scores_tensor = r.boxes.cpu().conf
            scores.append([float(conf) for conf in scores_tensor]) # convert tensors to floats

        return boxes, classes, scores

def create_detector(detector_name, confidence_threshold=None):
    """
    Creates detector object.

    Args:
    detector_name: Name of detector.
    """
    model_family = MODEL_DICT[detector_name]["model_family"]
    if model_family == "tensorflow":
        return TFDetector(detector_name, confidence_threshold)
    elif model_family == "yolo":
        return YOLODetector(detector_name, confidence_threshold)
    else:
        raise ValueError(model_family, "is not a valid model family.")
