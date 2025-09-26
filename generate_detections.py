""" 
Runs detections


Run detection model on all videos in VIDEO_PATH and save detections in DETECTION_DATA_PATH
The confidence threshold used will be the one set in the model config file. All other flags will be ignored.
All classes of interest defined in the model config file will be used.

usage: 
generate_detections.py [-h] [MODEL_NAME] [VIDEO_PATH] [SAVE_PATH]

arguments:
-h, --help            
                Show this help message and exit.
MODEL_NAME
                Object detection model name
VIDEO_PATH
                Path to directory containing videos
SAVE_PATH
                Path to save detections
"""

import cv2
import os
import glob
import argparse

from detection.Detector import create_detector, MODEL_DICT
from utils.data_writer import append_detections


OBJECT_DETECTION_MODELS = list(MODEL_DICT.keys())

# Initiate argument parser
parser = argparse.ArgumentParser(description="Run tracking experiment")
parser.add_argument(
    "model_name",
    choices=OBJECT_DETECTION_MODELS,
    help="Object detection model name",
    default=None,
    type=str,
)
parser.add_argument(
    "video_path",
    help="Path to directory containing videos",
    default=None,
    type=str,
)
parser.add_argument(
    "save_path",
    help="Path to directory to save detections",
    default=None,
    type=str,
)


def generate_detections(model, model_name, video_path, save_path):
    """
    Generates detections for every frame in videos and for every class supported by the model
    """
    # set up data dirs
    detection_data_path = os.path.join(save_path, model_name)
    if os.path.exists(detection_data_path):
        raise ValueError(f"Detection data path {detection_data_path} already exists. Please delete or rename.")
    else:
        os.makedirs(detection_data_path)
    
    batch_size = 8
    # Load model
    model.load_model()
    # Loop through videos
    file_extensions = ["*.MOV", "*.mov", "*.avi"]
    video_paths = []
    for ext in file_extensions:
        video_paths += glob.glob(os.path.join(video_path, ext))
    for video in video_paths:
        print(f"Running detection on {video} ...")
        frame_num = 0
        detection_csv_path = os.path.join(detection_data_path, os.path.basename(video).split(".")[0] + ".csv")
        # Creating csv file, this ensures that any existing detections from a previous run will be deleted and a fresh file created
        with open(detection_csv_path, 'w', newline='') as file:
            pass
        vid = cv2.VideoCapture(video)
        vid_fps = vid.get(cv2.CAP_PROP_FPS)
        success = True
        while success:
            # Make batch of images for prediction
            count = 0
            batch = []
            times = []
            while count < batch_size:
                success, frame = vid.read()
                if not success:
                    break
                batch.append(
                    frame[:, :, ::-1]
                )  # Convert from BGR to RGB and append
                times.append(round(float(frame_num) / vid_fps, 2))
                frame_num += 1
                count += 1
            if batch:  # Handle video with num frames divisible by batch size
                boxes, classes, confidences = model.detect(batch)
                append_detections(
                    boxes,
                    classes,
                    confidences,
                    frame_num,
                    times,
                    detection_csv_path,
                )
        vid.release()
    print("Detection complete")

if __name__ == "__main__":
    args = parser.parse_args()
    object_detector = args.model_name
    video_path = args.video_path
    save_path = args.save_path

    print(f"Generating detections using {object_detector} model")
    # create model
    model = create_detector(object_detector)
    model.load_label_dict()
    
    generate_detections(model, object_detector, video_path, save_path)