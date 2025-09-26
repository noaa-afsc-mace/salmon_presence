"""
All things data loading
"""

import csv

def load_detections(detections_csv_path, classes_of_interest):
    """
    Loads detections of classes of interest.

    Args:
    csv_path: Path to csv with detections of form:
        Frame, Video time, Class Name, Confidence, ymin, xmin, ymax, xmax
        where box coords are scaled values.
    classes_of_interest: List of classes of interest to be loaded.

    Returns dict:
    {'frame': {'vid_time': float, 'boxes': [], 'confs': [], 'classes': []}, ...}
    """
    detection_dict = {}
    with open(detections_csv_path, "r") as f:
        reader = csv.reader(f)
        
        for line in reader:
            frame = int(line[0])
            vid_time = float(line[1])
            class_name = line[2]
            # Only include classes of interest or all if no classes of interest
            if (class_name in classes_of_interest) or (classes_of_interest == []):
                conf = float(line[3])
                ymin = float(line[4])
                xmin = float(line[5])
                ymax = float(line[6])
                xmax = float(line[7])
                box = [ymin, xmin, ymax, xmax]

                if frame in detection_dict:
                    detection_dict[frame]["confs"].append(conf)
                    detection_dict[frame]["boxes"].append(box)
                    detection_dict[frame]["classes"].append(class_name)
                else:
                    detection_dict[frame] = {
                        "vid_time": vid_time,
                        "confs": [conf],
                        "boxes": [box],
                        "classes": [class_name],
                    }
            else:
                if frame in detection_dict:
                    continue
                else:
                    detection_dict[frame] = {
                        "vid_time": vid_time,
                        "confs": [],
                        "boxes": [],
                        "classes": [],
                    }

    return detection_dict
