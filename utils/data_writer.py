"""
Handles all things data writing and saving
"""

import os
import csv
from tabulate import tabulate
import matplotlib.pyplot as plt

def append_detections(boxes, classes, confidences, frame_num, times, csv_path):
    """
    Appends detection data to csv given by csv_path

    Args:
    boxes: List of bounding boxes [[frame n boxes], [frame n+1 boxes]] box is scaled and of form: [ymin, xmin, ymax, xmax]
    classes: List of classes.
    confidences: List of confidences.
    frame_num: Last frame num of batch.
    times: List of times corresponding to images in batch.
    csv_path: Path to csv file.
    """
    rows = []
    frame_offset = len(classes)

    # For each image
    for img_boxes, img_classes, img_confs, cur_time in zip(
        boxes, classes, confidences, times
    ):
        cur_frame = frame_num - frame_offset
        frame_offset -= 1
        # For each object
        for box, obj_class, conf in zip(img_boxes, img_classes, img_confs):

            row = [cur_frame, cur_time, obj_class, conf, box[0], box[1], box[2], box[3]]
            rows.append(row)
    # This is slower than writing all data at the end of the video but
    # worried about storing that much data in an array until end of video
    with open(csv_path, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(rows)

def write_overall_data(results_df, overall_results_save_path):
    """
    Calculates and writes overall results

    Args:
    results_df: dataframe with results for each video
    overall_results_save_path: path to save file
    """
    # calculate overall precision, recall, mse
    overall_precision = round(results_df["true_positives"].sum() / (results_df["true_positives"].sum() + results_df["false_positives"].sum()), 2) if (results_df["true_positives"].sum() + results_df["false_positives"].sum()) else 0
    overall_recall = round(results_df["true_positives"].sum() / (results_df["true_positives"].sum() + results_df["false_negatives"].sum()), 2) if (results_df["true_positives"].sum() + results_df["false_negatives"].sum()) else 0
    overall_mse = (results_df["vid_num_predicted"].sum() - results_df["vid_num_gt"].sum())**2
    average_video_individual_recall = round(results_df["individual_recall"].sum()/len(results_df.index),2) if len(results_df.index) else 0
    overall_detected_presences = results_df["num_detected"].sum()
    overall_missed_presences = results_df["num_instances"].sum() - overall_detected_presences
    overall_individual_recall = round(overall_detected_presences/results_df["num_instances"].sum(),2)
    proportion_tp_frames = round(results_df["true_positives"].sum()/results_df["num_included_frames"].sum(),2)
    proportion_fp_frames = round(results_df["false_positives"].sum()/results_df["num_included_frames"].sum(),2)
    proportion_gt_frames = round(results_df["num_gt_frames"].sum()/results_df["vid_frames"].sum(),2)

    # save overall results
    overall_headers = ["Overall Precision", "Overall Recall", "Total Predicted", "Total GT", "Overall MSE", \
                        "Average Video Individual Recall", "Overall Individual Recall", "Total Frames", "Total Saved Frames", "Proportion Included Frames", \
                        "Overall Detected Presences", "Overall Missed Presences", "Proportion TP frames in predicted", "Proportion FP frames in predicted", "Proportion GT frames in all frames"]
    overall_data = [[overall_precision, overall_recall, results_df["vid_num_predicted"].sum(), \
                    results_df["vid_num_gt"].sum(), overall_mse, average_video_individual_recall, overall_individual_recall,\
                    results_df["vid_frames"].sum(), results_df["saved_frames"].sum(), \
                    round(results_df["num_included_frames"].sum()/results_df["vid_frames"].sum(), 2), \
                    overall_detected_presences, overall_missed_presences, proportion_tp_frames,\
                    proportion_fp_frames, proportion_gt_frames]]
    with open(overall_results_save_path, "w") as f:
        f.write(tabulate(overall_data, headers=overall_headers))
    f.close()

def write_tow_data(results_df, tow_results_save_path):
    """
    Calculates and writes results for each tow

    Args:
        results_df: dataframe with results for each video
        tow_results_save_path: path to save file
    """
    # save per tow results
    tow_headers = ["Tow", "Precision", "Recall", "Total Predicted", "Total GT", "MSE", "Average Individual Recall", \
                   "Total Frames", "Total Saved Frames", "Proportion Included Frames", "Detected Presences", "Missed Presences", \
                    "Proportion TP frames in predicted", "Proportion FP frames in predicted", "Proportion GT frames in all frames"]
    tow_data = []
    for tow in results_df["tow"].unique():
        tow_df = results_df.loc[results_df['tow'] == tow]
        
        tow_precision = round(tow_df["true_positives"].sum() / (tow_df["true_positives"].sum() + tow_df["false_positives"].sum()), 2) if (tow_df["true_positives"].sum() + tow_df["false_positives"].sum()) else 0
        tow_recall = round(tow_df["true_positives"].sum() / (tow_df["true_positives"].sum() + tow_df["false_negatives"].sum()), 2) if (tow_df["true_positives"].sum() + tow_df["false_negatives"].sum()) else 0
        tow_mse = (tow_df["vid_num_predicted"].sum() - tow_df["vid_num_gt"].sum())**2
        tow_average_individual_recall = round(tow_df["individual_recall"].sum()/len(tow_df.index),2) if len(tow_df.index) else 0
        tow_detected_presences = tow_df["num_detected"].sum()
        tow_missed_presences = tow_df["num_instances"].sum() - tow_detected_presences
        prop_included_frames = round(tow_df["num_included_frames"].sum()/ tow_df["vid_frames"].sum(), 2)
        proportion_tp_frames = round(tow_df["true_positives"].sum()/tow_df["num_included_frames"].sum(),2)
        proportion_fp_frames = round(tow_df["false_positives"].sum()/tow_df["num_included_frames"].sum(),2)
        proportion_gt_frames = round(tow_df["num_gt_frames"].sum()/tow_df["vid_frames"].sum(),2)

        tow_data.append([tow, tow_precision, tow_recall, tow_df["vid_num_predicted"].sum(), tow_df["vid_num_gt"].sum(), tow_mse, tow_average_individual_recall, \
                        tow_df["vid_frames"].sum(), tow_df["saved_frames"].sum(), prop_included_frames, \
                        tow_detected_presences, tow_missed_presences, proportion_tp_frames, proportion_fp_frames, proportion_gt_frames])
    with open(tow_results_save_path, "w") as f:
        # sort results by vid name
        f.write(tabulate(tow_data, headers=tow_headers))
    f.close()

def write_vid_data(results_df, video_results_save_path):
    """
    Write vid data

    Args:
    results_df: dataframe with results for each video
    video_results_save_path: path to save file
    """

    # save per-video results
    with open(video_results_save_path, "w") as f:
        # sort results by vid name
        f.write(tabulate(results_df.sort_values("vid_name"), headers='keys', tablefmt='psql'))
    f.close()

# "Any road followed precisely to its end leads precisely nowhere."