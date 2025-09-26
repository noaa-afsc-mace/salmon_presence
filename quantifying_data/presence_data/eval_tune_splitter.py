"""
NOTE: DON'T RUN THIS UNLESS YOU REALLY NEED TO - IT WILL MAKE ALL OTHER EXPERIMENTS INCOMPARABLE

Splits data in all_presence_data into tune and eval and creates .txt files for each

Files are grouped by tow, TUNE_SPLIT portion of videos in each tow are randomly assigned to tune, the rest are assigned to eval

All videos in a tow are randomized and the first n are selected for tuning for each tuning size

This means that all tuning sets are a subset of the largest tuning set and all eval sets contain the smallest eval set
"""

import glob
import os
import random
import math

PRESENCE_DATA_DIR = "quantifying_data/presence_data/all_presence_data/"
EVAL_DIR = "quantifying_data/presence_data/eval/"
TUNE_DIR = "quantifying_data/presence_data/tune/"
TUNE_SPLITS = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]

files = glob.glob(os.path.join(PRESENCE_DATA_DIR, "*.csv"))

tow_map = {}

for file in files:
    tow = os.path.basename(file).split(".")[0][1:3]
    if tow in tow_map:
        tow_map[tow].append(file)
    else:
        tow_map[tow] = [file]

for tune in TUNE_SPLITS:
    tune_file = os.path.join(TUNE_DIR, f"tune_{str(tune)}.txt")
    eval_file = os.path.join(EVAL_DIR, f"eval_{str(1-tune)}.txt")
    tune_files = []
    eval_files = []
    # iterate through each tow
    for t in tow_map.keys():
        tow_files = tow_map[t]
        num_tune = math.ceil(len(tow_files)*tune)
        # randomize files
        random.shuffle(tow_files)
        tune_files.extend(tow_files[:num_tune])
        eval_files.extend(tow_files[num_tune:])

    with open(tune_file, 'w') as file:
        # Write each item to a new line
        for v in tune_files:
            file.write(v + '\n')
    with open(eval_file, 'w') as file:
        # Write each item to a new line
        for v in eval_files:
            file.write(v + '\n')

print("Done splitting")