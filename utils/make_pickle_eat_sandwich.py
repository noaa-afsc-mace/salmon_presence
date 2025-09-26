"""
Loads and then pickles all detections to specified location

This is done to speed up analysis for large detection files
"""
import os
import sys
import glob
from tqdm import tqdm
import pickle

# Add parent directory to path to import salmost
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from salmost import DETECTION_DATA_PATH, PRESENCE_DATA_PATH, PICKLED_DETECTION_DATA_PATH
from loaders import load_detections

# --- thing you might want to change ---

DETECTION_SOURCE = "yolo_11_salmon-only_best_clips"

# --------------------------------------


PICKLE_DESTINATION = os.path.join(PICKLED_DETECTION_DATA_PATH, DETECTION_SOURCE)
if os.path.exists(PICKLE_DESTINATION):
    raise ValueError(f"Pickled detections for {PICKLE_DESTINATION} already exists. Aborting")
else:
    os.mkdir(PICKLE_DESTINATION)

# pickle all detections
presence_files = glob.glob(os.path.join(PRESENCE_DATA_PATH, "*.csv"))
for i in tqdm(range(len(presence_files))):
    vid_name = os.path.basename(presence_files[i]).split(".")[0]
    detection_file = os.path.join(DETECTION_DATA_PATH, DETECTION_SOURCE, f"{vid_name}.csv")
    salmon_detections_dict = load_detections(detection_file, ["Salmon"])
    # make pickle
    pickle_name = os.path.join(PICKLE_DESTINATION, f'{vid_name}.pickle')
    with open(pickle_name, 'wb') as handle:
        pickle.dump(salmon_detections_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Eat sandwich")
