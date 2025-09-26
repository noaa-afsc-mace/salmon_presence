"""
Splits the csv downloaded from google sheets into individual csv files for each video

The input csv is of the form:
|       <vid_name>       |       <vid_name>       |
| start frame, end frame | start frame, end frame |
| frame num  , frame num | frame num  , frame num |

So it's a pretty hideous csv due to the merged cells

The output of this script is a bunch of new files with the naming format `T<tow number>_<vid_name>.csv`
with the format: 
start frame, end frame
frame num  , frame num
"""

import csv
import os

GOOGLE_SHEET = "quantifying_data/presence_data/google_sheet_conversion_stuff/sheets/2019_salmon_times_final.xlsx - Tow 16.csv"
SAVE_DIR = "quantifying_data/presence_data/all_presence_data"
TOW_NUM = 16
COLUMNS = ["start frame", "end frame"]

# load google sheet csv
sheet_data = []
with open(GOOGLE_SHEET, "r") as f:
    for line in f:
        line = line.strip().split(",")
        sheet_data.append(line)

# get number of videos in tow
vid_names = list(filter(None, sheet_data[0]))
num_vids = len(vid_names)

# iterate through data and get each vid's data
for i in range(num_vids):
    # get data for a vid
    single_vid_slice = [x[(i*2):(i*2)+2] for x in sheet_data]
    vid_name = single_vid_slice[0][0]
    frame_data = single_vid_slice[2:] # contains emptys
    new_file_name = f"T{TOW_NUM}_{vid_name}.csv"
    # write to new file
    with open(os.path.join(SAVE_DIR, new_file_name), "w+", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(COLUMNS)
        for r in frame_data:
            if r[0] and r[1]: # check that not empty
                csvwriter.writerow(r)
    csvfile.close()

print("Splitting complete")
