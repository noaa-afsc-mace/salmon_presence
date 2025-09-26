"""
This file is used to find the optimal hyperparameters for a quantifier
by maximizing the weighted score of precision and frames saved
"""

import os
import cv2
import time
import pickle
from tqdm import tqdm
from lipo import GlobalOptimizer
from tabulate import tabulate

from quantifying.DetectNum import DetectNum
from salmost import PICKLED_DETECTION_DATA_PATH, VIDEO_PATH
from statistics import mean

# ---- hey, these are the things you might want to change ----
NUM_ITERATIONS = 10
lower_bound_confidence_threshold = 0.0
upper_bound_confidence_threshold = 1.0
lower_bound_min_presence_length= 1
upper_bound_min_presence_length = 3
lower_bound_min_dets_per_frame = 1
upper_bound_min_dets_per_frame = 5
DETECTION_SOURCE = "yolo_11_salmon-only_best_clips"
TUNING_DATASET = "tune_0.2.txt"
# ----

OPTIMIZING_DATA_SAVE_PATH = "quantifying_data/optimizing/"
TUNING_DATA_DIR = "quantifying_data/presence_data/tune"
INDIVIDUAL_RECALL_WEIGHT = 0.7
PROPORTION_OF_FRAMES_SAVED_WEIGHT = 0.3

assert INDIVIDUAL_RECALL_WEIGHT + PROPORTION_OF_FRAMES_SAVED_WEIGHT == 1

def load_data():
    """
    Loads all detection and vid data
    """
    detections_dict = {} # dict of form {"vid_name": detection_dict}
    vid_dict = {} # dict of form {"vid_name": total_vid_frames}

    files_to_eval = [line.strip() for line in open(os.path.join(TUNING_DATA_DIR, TUNING_DATASET), 'r')]
    for i in tqdm(range(len(files_to_eval))):
        vid_name = os.path.basename(files_to_eval[i]).split(".")[0]
        vid_path = os.path.join(VIDEO_PATH, vid_name + ".MOV")
        detection_file = os.path.join(PICKLED_DETECTION_DATA_PATH, DETECTION_SOURCE, f"{vid_name}.pickle")
        # get num frames in video
        vid = cv2.VideoCapture(vid_path)
        total_vid_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        vid.release()
        # load detections
        with open(detection_file, 'rb') as handle:
            salmon_detections_dict = pickle.load(handle)
        # add to dict
        detections_dict[vid_name] = salmon_detections_dict
        vid_dict[vid_name] = total_vid_frames
    
    return detections_dict, vid_dict

def calculate_score(individual_recall, proportion_saved_frames):
    """
    Calculates score for a video
    """

    return (INDIVIDUAL_RECALL_WEIGHT*individual_recall) + (PROPORTION_OF_FRAMES_SAVED_WEIGHT*proportion_saved_frames)


def write_results_lipo_scale(score_list, conf_list, presence_list, dets_per_frame_list, headers, save_path):
    """
    Saves txt file of top 100 results to save_path

    Args:
    score_list: list of scores
    individual_recall_list: list of recall
    proportion_saved_frames_list: list of proportions
    headers: headers for table to save
    save_path: path to save file
    """
    # generate rows for tabulate
    tabulate_rows = []
    i = 0
    while i < min(100, len(score_list)):
        # generate rows for tabulate
        rank = i+1
        score = round(score_list[i],4)
        conf_thresh = round(conf_list[i],4)
        presence_thresh = presence_list[i]
        dets_per_frame_thresh = dets_per_frame_list[i]
        row = [rank, score, conf_thresh, presence_thresh, dets_per_frame_thresh]
        tabulate_rows.append(row)
        i += 1
    with open(save_path, 'w') as f:
        f.write(tabulate(tabulate_rows, headers=headers))

def quant_runner_for_lipo(confidence_threshold, min_presence_length, min_dets_per_frame):
    """
    Runs the detect_num quantifier for all vids in tune and returns a weighted score of precision and saved frames.
    All non-optimized inputs are hardcoded so that the LIPO optimizer can run properly

    Args:
    conf_thresh: float, lower bound of detection confidence, inclusive
    min_presence_len: int, min required frame length of presence, inclusive
    min_dets_per_frame: min number of detections per frame, inclusive

    Returns:
    score: float, weighted score for given params
    """

    # run on all tune vids
    scores = []
    # loop through files in presence source
    files_to_eval = [line.strip() for line in open(os.path.join(TUNING_DATA_DIR, TUNING_DATASET), 'r')]
    for i in range(len(files_to_eval)):
        vid_name = os.path.basename(files_to_eval[i]).split(".")[0]
        total_vid_frames = VID_DICT[vid_name]
        # create quantifier
        quantifier = DetectNum("detect_num", confidence_threshold, min_presence_length, min_dets_per_frame, total_vid_frames)
        # run quantification
        quantifier.quantify(DETECTION_DICT[vid_name], "")
        # load gt data
        gt_data = quantifier.load_gt(files_to_eval[i])
        # evaluate
        individual_recall, proportion_saved_frames, num_frames_reviewed, saved_frames, num_instances, num_detected = quantifier.individual_recall(gt_data, total_vid_frames)
        # save score
        scores.append(calculate_score(individual_recall, proportion_saved_frames))
    
    # calculate average score
    return mean(scores)

if __name__ == "__main__":

    # load all detections and video frame data into memory
    print("loading all detection data")
    DETECTION_DICT, VID_DICT = load_data()

    search = GlobalOptimizer(
        quant_runner_for_lipo,
        lower_bounds={"confidence_threshold": lower_bound_confidence_threshold, "min_presence_length": lower_bound_min_presence_length, "min_dets_per_frame": lower_bound_min_dets_per_frame},
        upper_bounds={"confidence_threshold": upper_bound_confidence_threshold, "min_presence_length": upper_bound_min_presence_length, "min_dets_per_frame": upper_bound_min_dets_per_frame},
        maximize=True,
    )

    print('Running LIPO...')
    search.run(NUM_ITERATIONS)
    top_sorted = sorted(search.evaluations, key=lambda x: x[1], reverse=True)
    conf_list = list(map(lambda x: x[0]["confidence_threshold"], top_sorted))
    presence_list = list(map(lambda x: x[0]["min_presence_length"], top_sorted))
    min_dets_list = list(map(lambda x: x[0]["min_dets_per_frame"], top_sorted))
    scores = list(map(lambda x: x[1], top_sorted))

    print("Top 10 hyper parameter combinations:")
    tabulate_headers = ['Rank', 'Score', 'Confidence threshold', 'Min presence threshold', 'Min detections per frame']
    tabulate_rows = []
    i = 0
    while i < min(10, len(scores)):
        # generate rows for tabulate
        tabulate_rows.append([i+1, round(scores[i], 4), round(conf_list[i],4), presence_list[i], min_dets_list[i]])
        i += 1
    print(tabulate(tabulate_rows, headers=tabulate_headers))
    time_str = time.strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(OPTIMIZING_DATA_SAVE_PATH, f"detect_num_{DETECTION_SOURCE}_{TUNING_DATASET[:-4]}_{time_str}_results.txt")
    write_results_lipo_scale(scores, conf_list, presence_list, min_dets_list, tabulate_headers, save_path)