"""
Graphs detection confidence threshold vs saved frames and individual recall
"""
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

from quantifying.DetectNum import DetectNum
from salmost import PICKLED_DETECTION_DATA_PATH, VIDEO_PATH
from detect_num_optimizer import TUNING_DATA_DIR, TUNING_DATASET
from statistics import mean

OPTIMIZING_DATA_SAVE_PATH = "quantifying_data/optimizing/"

# ----
# hey, these are the things you might want to change:
STEP_SIZE = 0.01 # confidence level increments
MIN_PRESENCE_LEN = 1
MIN_DETS_PER_FRAME = 1
lower_bound_confidence_threshold = 0.0
upper_bound_confidence_threshold = 1.0
DETECTION_SOURCE = "yolo_11_salmon-only_best_clips"
# ----

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


if __name__ == "__main__":

    # load all detections and video frame data into memory
    print("loading all detection data")
    DETECTION_DICT, VID_DICT = load_data()

    ind_recall = []
    prop_saved_frames = []
    conf_list = []
    precision_all = []
    recall_all = []

    print("Generating results...")
    # run for every confidence level
    for i in tqdm(range(int(lower_bound_confidence_threshold / STEP_SIZE), int(upper_bound_confidence_threshold / STEP_SIZE) + 1)):
        current_conf = i * STEP_SIZE
        # run on all tuning vids
        run_ind_recall = []
        run_prop_saved_frames = []
        run_precision = []
        run_recall =  []
        # loop through files in presence source
        files_to_eval = [line.strip() for line in open(os.path.join(TUNING_DATA_DIR, TUNING_DATASET), 'r')]
        for i in range(len(files_to_eval)):
            vid_name = os.path.basename(files_to_eval[i]).split(".")[0]
            total_vid_frames = VID_DICT[vid_name]
            # create quantifier
            quantifier = DetectNum("detect_num", current_conf, MIN_PRESENCE_LEN, MIN_DETS_PER_FRAME, total_vid_frames)
            # run quantification
            quantifier.quantify(DETECTION_DICT[vid_name], "")
            # load gt data
            gt_data = quantifier.load_gt(files_to_eval[i])
            # evaluate individual 
            individual_recall, proportion_saved_frames, num_frames_reviewed, saved_frames, num_instances, num_detected = quantifier.individual_recall(gt_data, total_vid_frames)
            # evaluate overall
            presence_precision, presence_recall, *_ = quantifier.evaluate(gt_data)
            # save score
            run_ind_recall.append(individual_recall)
            run_prop_saved_frames.append(1-proportion_saved_frames)
            run_precision.append(presence_precision)
            run_recall.append(presence_recall)

        ind_recall.append(mean(run_ind_recall))
        prop_saved_frames.append(mean(run_prop_saved_frames))
        conf_list.append(current_conf)
        precision_all.append(mean(run_precision))
        recall_all.append(mean(run_recall))

    # colorblind friendly colors, source: https://www.nceas.ucsb.edu/sites/default/files/2022-06/Colorblind%20Safe%20Color%20Schemes.pdf
    red = (191/255,44/255,35/255)
    blue = (47/255,103/255,177/255)

    # ----- first graph -------
    # Plotting the first stuff
    plt.plot(conf_list, ind_recall, color=blue, label='Individual presence instance recall')
    plt.plot(conf_list, prop_saved_frames, color=red, label='Proportion of included frames')

    # Adding labels and title
    plt.xlabel('Confidence threshold')
    plt.ylabel('Value')
    # plt.title(f"Confidence threshold vs presence recall and eliminated frames")
    plt.legend()  # Show legend with labels

    plt.savefig(os.path.join(OPTIMIZING_DATA_SAVE_PATH, f"conf_vs_recall_proportion_included_frames_{DETECTION_SOURCE}.png"), dpi=600)
    plt.close()

    # ----- second plot ------

    plt.plot(conf_list, precision_all, color=blue, label='Overall presence precision')
    plt.plot(conf_list, recall_all, color=red, label='Overall presence recall')

    # Adding labels and title
    plt.xlabel('Confidence threshold')
    plt.ylabel('Value')
    # plt.title(f"Confidence threshold vs presence recall and eliminated frames")
    plt.legend()  # Show legend with labels

    plt.savefig(os.path.join(OPTIMIZING_DATA_SAVE_PATH, f"conf_vs_overall_precision_recall_{DETECTION_SOURCE}.png"), dpi=600)
    plt.close()