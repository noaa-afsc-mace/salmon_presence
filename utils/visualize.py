"""
Various plotting functions
"""

import math
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def visualize_tows(presence_prediction_data_dict, gt_presence_data_dict, experiment_results_path):
    """
    Plots results for all tows

    There is 30 frame overlap between videos

    Args:
    presence_prediction_data_dict: dict of form {vid_name: [num_vid_frames, set(all frames where salmon were predicted present)]}
    gt_presence_data_dict: dict of form {vid_name: [num_vid_frames, [[start frame, end frame], ...]]}
    experiment_results_path: path to save plots
    """

    # group all of the vids for a tow

    pred_dict = {} # dict of {tow: [(vid_name, num_vid_frames, set_of_frames), ...]}
    gt_dict = {} # {tow: [(vid_name, num_vid_frames, [[start frame, end frame], ...]), ...]}

    for vid_name, vid_info in presence_prediction_data_dict.items():
        tow = vid_name[:3]
        if tow in pred_dict:
            pred_dict[tow].append((vid_name, vid_info[0], vid_info[1]))
        else:
            pred_dict[tow] = [(vid_name, vid_info[0], vid_info[1])]
    
    for vid_name, vid_info in gt_presence_data_dict.items():
        tow = vid_name[:3]
        if tow in gt_dict:
            gt_dict[tow].append((vid_name, vid_info[0], vid_info[1]))
        else:
            gt_dict[tow] = [(vid_name, vid_info[0], vid_info[1])]

    # Iterate through each tow, do all kinds of stuff to it
    for tow, pred_tow_data in pred_dict.items():
        gt_tow_data = gt_dict[tow]
        # sort gt and pred data
        gt_tow_data.sort(key=lambda x: x[0][-4:])
        pred_tow_data.sort(key=lambda x: x[0][-4:])
        # now for the fun stuff, assembling a single set of frames for the whole graph
        merged_gt_data = [] # all start and end frames [[start, end], ...] 
        merged_frames_pred = [] # all presence frames
        total_len = 0 # total len not taking overlap into account
        overlap = 30
        for i in range(len(gt_tow_data)):
            # check frame num and vid name is the same for each
            assert gt_tow_data[i][1] == pred_tow_data[i][1]
            assert gt_tow_data[i][0] == pred_tow_data[i][0]
            # correct all frame nums
            frame_correction = (total_len - (i*overlap))
            merged_gt_data.extend([[gt[0]+ frame_correction, gt[1]+frame_correction] for gt in gt_tow_data[i][2]])
            merged_frames_pred.extend([f + frame_correction for f in pred_tow_data[i][2]])
            total_len += gt_tow_data[i][1]
        
        # make into set to account for overlaps
        # can't do any kind of merging for gt because I have no way of telling if they are different salmon or vid overlaps
        pred_presence_frames_set = set(merged_frames_pred)
        # graph 'em
        graph_tow(merged_gt_data, pred_presence_frames_set, tow, experiment_results_path)

def graph_tow(merged_gt_data, pred_presence_frames_set, tow, experiment_results_path):
    """
    Graphs entire tow of data

    Args:
    merged_gt_data: all start and end frames [[start, end], ...] 
    pred_presence_frames_set: set of all frames where salmon presence is predicted
    tow: tow name
    experiment_results_path: path to save graph
    """

    # determine detected presences and show as filled in point, show missed as empty point

    # find caught salmon (code copied from Quantifier method), probably a better way to do this
    caught = [] # middle frame (rounded up) of caught salmon
    missed = [] # same but for missed salmon
    for gt in merged_gt_data:
        start = gt[0]
        end = gt[1]
        middle = math.ceil((start + end) / 2)
        gt_frames = list(range(start, end+1))
        # check if any gt frames for this individual are in quantifier frames
        if len(set(gt_frames).intersection(pred_presence_frames_set)) > 0:
            caught.append(middle)
        else:
            missed.append(middle)
    
    # # get all false positive detections
    # gt_frames = []
    # for gt in merged_gt_data:
    #     gt_frames.extend(list(range(gt[0], gt[1]+1)))
    # gt_frames = list(set(gt_frames))
    # false_positives = list(pred_presence_frames_set - set(gt_frames))
            
    # colorblind friendly colors, source: https://www.nceas.ucsb.edu/sites/default/files/2022-06/Colorblind%20Safe%20Color%20Schemes.pdf
    red = (191/255,44/255,35/255)
    blue = (47/255,103/255,177/255)

    fig, ax = plt.subplots(figsize=(20,5))
    
    # graph caught and missed salmon
    ax.scatter(caught, [1]*len(caught), color=blue, marker='o')
    ax.scatter(missed, [2]*len(missed), facecolors='none', color=red, marker='o')

    def timeformat(x,pos=None):
        total_seconds = x / 30
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        return f"{hours:02d}:{minutes:02d}"

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(timeformat))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(base=30 * 900))
    # # graph false positives
    # plt.scatter(false_positives, [3]*len(false_positives), color=red, marker=',')

    plt.ylim(0, 3)
    plt.xlabel('Video Time [hh:mm]')
    plt.grid(axis='y')
    plt.yticks([1, 2], labels=["Detected salmon", "Missed salmon"])
    plt.title(f"Salmon presence detection in {tow}")
    plt.savefig(os.path.join(experiment_results_path, f"{tow}_presence_results.png"))
