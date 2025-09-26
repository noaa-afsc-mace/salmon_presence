"""
Quantifier class
"""

from abc import ABC, abstractmethod
import os

class Quantifier(ABC):
    """
    A class that represents a quantifier technique. Allows for maximum code
    flexibility.
    """

    def __init__(self, quantifier_name, confidence_threshold, min_presence_length, total_vid_frames):
        self.quantifier_name = quantifier_name
        self.confidence_threshold = confidence_threshold
        self.min_presence_length = min_presence_length
        self.total_vid_frames = total_vid_frames
        self.track_data = {} 
        # {
        # track_id: {
        #   "frames": [frame nums]
        #   "class_names": [class_names]
        #   "track_boxes": [boxes]
        #   }
        # }
        self.presence_frames = [] # list of frames where salmon are present

    def add_track_data(
        self, frame_num, track_class, track_id, track_box
    ):
        """
        Adds data to track_data dict.

        Args:
        frame_num: Int of frame number.
        track_class: String of track class.
        track_id: Int of track id.
        track_box: [ymin, xmin, ymax, xmax] in pixel values.
        """
        # Boxes not scaled
        if track_id not in self.track_data:
            self.track_data[track_id] = {
                "frames": [frame_num],
                "class_names": [track_class],
                "track_boxes": [track_box]
            }
        else:
            self.track_data[track_id]["frames"].append(frame_num)
            self.track_data[track_id]["class_names"].append(track_class)
            self.track_data[track_id]["track_boxes"].append(track_box)

    def load_gt(self, gt_path):
        """
        Load gt data

        Args:
        gt_path: Path to ground truth csv of form "start frame, end frame" for each salmon occurrence
        """
        gt_data = []
        gt_path = os.path.join("quantifying_data/presence_data/all_presence_data/" ,os.path.basename(gt_path))
        with open(gt_path, "r") as f:
            # Skip header
            next(f)
            for line in f:
                line = line.strip().split(",")
                start_frame = int(line[0])
                end_frame = int(line[1])
                assert start_frame <= end_frame
                gt_data.append([start_frame, end_frame])
        return gt_data

    def evaluate(self, gt_data):
        """
        Calculates video level metrics

        Args:
        gt_data: gt_data of form [[start frame, end frame],...]

        Returns:
        presence_precision: precision of presence
        presence_recall: recall of presence
        num_predicted: number of individuals predicted
        num_gt: number of individuals in gt
        mse: mean squared error of number of individuals predicted and number of individuals in gt
        num_true_positives: number of true positives
        num_false_positives: number of false positives
        num_false_negatives: number of false negatives
        num_gt_frames: number of gt frames
        """

        # get true positives
        # first, get all frames that quantifier detected salmon, as per correct method, so need to use results
        quantifier_frames = set(self.get_results_frames())
        # second, get all frames that gt has salmon
        gt_frames = []
        for gt in gt_data:
            gt_frames.extend(list(range(gt[0], gt[1]+1)))
        gt_frames = set(gt_frames)
        # third, get intersection of frames
        true_positives = list(quantifier_frames.intersection(gt_frames))
        # get false positives
        false_positives = list(quantifier_frames - gt_frames)
        # get false negatives
        false_negatives = list(gt_frames - quantifier_frames)
        # get precision and recall
        presence_precision = round(len(true_positives) / (len(true_positives) + len(false_positives)),2) if (len(true_positives) + len(false_positives)) else 0
        presence_recall = round(len(true_positives) / (len(true_positives) + len(false_negatives)),2) if (len(true_positives) + len(false_negatives)) else 0

        # get number of individuals predicted
        num_predicted = len(self.track_data.keys())
        # get number of individuals in gt
        num_gt = len(gt_data)
        # get mse
        mse = (num_predicted - num_gt)**2
        
        return presence_precision, presence_recall, num_predicted, num_gt, mse, len(true_positives), len(false_positives), len(false_negatives), len(gt_frames)

    def individual_recall(self, gt_data, total_vid_frames):
        """
        Calculates individual recall
        The idea is to determine the proportion of individual salmon that are detected at some point during their presence in the video.

        We iterate through each individual in the ground truth and determine if the quantifier detected a salmon in any of the frames that the individual is present in.

        Args:
        gt_data: gt_data of form [[start frame, end frame],...]
        total_vid_frames: total number of frames in video

        Returns:
        individual_recall: recall of individual salmon
        proportion_saved_frames: proportion of frames that would not need to be reviewed
        num_included_frames: number of frames included (TP+FP)
        saved_frames: number of frames that would not need to be reviewed. If salmon are missed then this is a pretty bad metric
        num_instances: number of instances of presence
        num_detected: number of instances detected
        """
        # get presence frames as per the current method (this is because of the min presence issue where some presence frames should get filtered out)
        quantifier_frames = set(self.get_results_frames())
        # get all frames that quantifier detected salmon
        num_detected = 0
        gt_frames = []
        for gt in gt_data:
            start = gt[0]
            end = gt[1]
            gt_sub_f = list(range(start, end+1))
            gt_frames.extend(gt_sub_f)
            # check if any gt frames for this individual are in quantifier frames
            caught = set(gt_sub_f).intersection(quantifier_frames)
            if len(caught) > 0:
                num_detected += 1
        gt_frames = set(gt_frames)
        num_instances = len(gt_data)
        individual_recall = round(num_detected / num_instances, 2) if num_instances else 1

        # calculate the number of frames that would need to be reviewed
        true_positives = list(quantifier_frames.intersection(gt_frames))
        false_positives = list(quantifier_frames - gt_frames)
        num_included_frames = len(true_positives)+len(false_positives)
        saved_frames = total_vid_frames - num_included_frames
        proportion_saved_frames = round(saved_frames / total_vid_frames, 2)

        return individual_recall, proportion_saved_frames, num_included_frames, saved_frames, num_instances, num_detected
    
    def presence_stats(self, gt_data, salmon_pollock_detections, vid_name, density_conf_threshold):
        """
        Calculates stats for each presence

        Args:
        gt_data: gt_data of form [[start frame, end frame],...]
        salmon_pollock_detections: detections of form: {'frame': {'vid_time': float, 'boxes': [], 'confs': [], 'classes': []}, ...}
        density_conf_threshold: confidence threshold for detections used in density calculation

        Returns:
        presence_data: list of form [[name, avg_detections, caught, length],...]]
        """
        # get presence frames as per the current method (this is because of the min presence issue where some presence frames should get filtered out)
        quantifier_frames = set(self.get_results_frames())
        # build lists of data
        presence_data = []
        for gt in gt_data:
            caught = False
            start = gt[0]
            end = gt[1]
            gt_frames = list(range(start, end+1))
            # check if any gt frames for this individual are in quantifier frames
            if len(set(gt_frames).intersection(quantifier_frames)) > 0:
                caught = True
            length = len(gt_frames)
            # get avg detection per frame
            num_detections = 0
            for frame in gt_frames:
                if frame in salmon_pollock_detections:
                    for conf in salmon_pollock_detections[frame]["confs"]:
                        if conf >= density_conf_threshold:
                            num_detections += 1
            avg_detections = round(num_detections / length, 2)

            presence_data.append([f"{vid_name}_{start}_{end}", avg_detections, caught, length])

        return presence_data

    def convert_presence_to_chunks(self, frame_list):
        """
        Converts list of frames to presence list of form [[start frame, end frame],...]
        Does NOT consider min_presence_length

        Args:
        frame_list: list frames

        Returns:
        presence_list: list of lists with start and end frames of each individual salmon: [[start frame, end frame],...] 
        """
        presence_list = []
        nums = sorted(set(frame_list))
        gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
        edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
        intermediate = list(zip(edges, edges))
        for i in intermediate:
            start_frame, end_frame = i
            presence_list.append([start_frame, end_frame])
        return presence_list

    @abstractmethod
    def get_results_start_end(self):
        """
        Returns results of quantification method

        Returns:
        presence_list: list of lists with start and end frames of each individual salmon: [[start frame, end frame],...] (subject to min presence threshold)
        """
        pass

    @abstractmethod
    def get_results_frames(self):
        """
        Returns results of quantification method

        Returns:
        frame_list: list of frames where salmon were present (subject to min presence threshold)
        """
        pass

    @abstractmethod
    def quantify(self, detections_dict, vid_path):
        """
        Quantifies salmon in video

        Args:
        detections_dict: dict of detections from model of form: {'frame': {'vid_time': float, 'boxes': [], 'confs': [], 'classes': []}, ...}
        vid_path: path to video
        """
        pass

    @abstractmethod
    def visualize(self):
        """
        Visualizes the track data, can vary between quantifiers
        """

    def barh_chunks(self, chunks):
        """
        Converts chunks to be used with barh. 
        If chunk is a single frame, width is 1, otherwise width is the difference between the start and end frame.

        Args:
        chunks: list of form [[start_frame, end_frame],...]
        """
        plot_chunks = []
        for c in chunks:
            if c[0] == c[1]:
                w = 1
            else:
                w = c[1] - c[0]
            plot_chunks.append([c[0], w])
        return plot_chunks

