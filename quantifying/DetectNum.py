import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import os

from quantifying.Quantifier import Quantifier

class DetectNum(Quantifier):
    def __init__(self, quantifier_name, confidence_threshold, min_presence_length, min_dets_per_frame, total_vid_frames):
        super().__init__(quantifier_name, confidence_threshold, min_presence_length, total_vid_frames)
        self.min_dets_per_frame = min_dets_per_frame
    
    def quantify(self, detections_dict, vid_path):
        """
        Quantifies salmon in video

        Gets all salmon detections above confidence threshold

        Only detects, does not attempt to distinguish between individuals
        """

        # get all detections and frames
        all_classes = []
        all_frames = []
        all_confs = []
        all_boxes = []
        for frame_num in detections_dict.keys():
            frame_detections = detections_dict[frame_num]
            boxes = np.array(frame_detections["boxes"])
            classes = frame_detections["classes"]
            confs = frame_detections["confs"]

            # filter out low confidence detections
            conf_mask = np.array(confs) >= self.confidence_threshold
            boxes = np.asarray(boxes)[conf_mask]
            classes = np.asarray(classes)[conf_mask]
            confs = np.asarray(confs)[conf_mask]
            
            # check if num above thresh for frame
            if len(classes) >= self.min_dets_per_frame:
                all_classes.extend(classes)
                all_frames.extend([frame_num]*len(classes))
                all_confs.extend(confs)
                all_boxes.extend(boxes)

        # add a single entry to track data, since we aren't trying to track individuals we just enter them all as one track, shouldn't affect presence calculation
        for f, c, b in zip(all_frames, all_classes, all_boxes):
            self.add_track_data(f, c, 0, b) # do I need to get rid of duplicates?
            self.presence_frames.append(f)

    def get_results_start_end(self):

        presence_list = []
        nums = sorted(set(self.presence_frames))
        gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
        edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
        intermediate = list(zip(edges, edges))
        for i in intermediate:
            start_frame, end_frame = i
            if ((end_frame - start_frame) + 1) < self.min_presence_length:
                continue
            else:
                presence_list.append([start_frame, end_frame])
        return presence_list
    
    def get_results_frames(self):

        frames_list = []
        for chunk in self.get_results_start_end():
            start_frame = chunk[0]
            end_frame = chunk[1]
            frames_list+=list(range(start_frame, end_frame+1))
        return frames_list
    
    def visualize(self, gt_data, save_path, vid_name):
        """
        It's all in the name
        """
        self.visualize_presence_barh(gt_data, save_path, vid_name)
        # self.visualize_presence_line(gt_data, save_path, vid_name)

    def visualize_presence_line(self, gt_data, save_path, vid_name):
        """
        It's all in the name
        """
        # gt has salmon
        gt_frames = []
        for gt in gt_data:
            gt_frames.extend(list(range(gt[0], gt[1]+1)))
        gt_frames = list(set(gt_frames))

        # plot
        fig, ax = plt.subplots()
        
        # plot track
        detect_y = 1
        gt_y = 3
        # plot detect
        for chunk in self.get_results_start_end():
            if chunk[0] == chunk[1]:
                ax.scatter(chunk[0], detect_y, c="black", marker = "|")
            else:
                ax.plot([chunk[0], chunk[1]], [detect_y, detect_y], c="black")
                ax.scatter(chunk[0], detect_y, c="black", marker = "|")
                ax.scatter(chunk[1], detect_y, c="black", marker = "|")
        # plot gt
        for chunk in gt_data:
            ax.plot([chunk[0], chunk[1]], [gt_y, gt_y], c="black")
            ax.scatter(chunk[0], gt_y, c="black", marker = "|")
            ax.scatter(chunk[1], gt_y, c="black", marker = "|")
        ax.set_ylim(0, 6)
        ax.set_xlabel('Frame number')
        ax.grid(axis='y')
        ax.set_yticks([1, 3], labels=["Predicted", "Actual"])
        ax.set_title(f"Salmon presence in {vid_name}")
        plt.tight_layout() 
        plt.savefig(os.path.join(save_path, f"{vid_name}_line_presence_graph.png"), dpi=300)
        plt.close()   

    def visualize_presence_barh(self, gt_data, save_path, vid_name):
        """
        It's all in the name
        """
        # want to get the overlapping frames between gt and detections
        # first, get all frames that quantifier detected salmon
        quantifier_frames = set(self.get_results_frames())
        # second, get all frames that gt has salmon
        gt_frames = []
        for gt in gt_data:
            gt_frames.extend(list(range(gt[0], gt[1]+1)))
        gt_frames = list(set(gt_frames))
        # third, get intersection of frames
        true_positives = list(quantifier_frames.intersection(set(gt_frames)))
        false_positives = list(quantifier_frames - set(gt_frames))
        # convert to chunks
        true_chunks = self.convert_presence_to_chunks(true_positives)
        false_chunks = self.convert_presence_to_chunks(false_positives)

        # plot
        # colorblind friendly colors, source: https://www.nceas.ucsb.edu/sites/default/files/2022-06/Colorblind%20Safe%20Color%20Schemes.pdf
        red = (191/255,44/255,35/255)
        blue = (47/255,103/255,177/255)
        lw = 0.5

        fig, ax = plt.subplots(figsize=(20,5))
        # plot true detections
        ax.broken_barh(self.barh_chunks(false_chunks), (2,4), linewidth=lw, color=red, facecolors=red) 
        ax.broken_barh(self.barh_chunks(true_chunks), (2,4), linewidth=lw, color=blue, facecolors=blue) 
        ax.broken_barh(self.barh_chunks(gt_data), (7,4), linewidth=lw, color=blue, facecolors=blue)

        def timeformat(x,pos=None):
            total_seconds = x / 30
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            return f"{minutes:02d}:{seconds:02d}"

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(timeformat))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(base=30 * 60))

        fs = 22
        plt.ylim(0, 13)
        plt.xlabel('Video Time [mm:ss]', fontsize=fs)
        plt.grid(axis='y')
        plt.yticks([4, 9], labels=["Predicted", "Actual"], fontsize=fs)
        
        # custom, manual legend
        red_patch = mpatches.Patch(color=red, label='False positive detections')
        blue_patch = mpatches.Patch(color=blue, label='True salmon presence')
        plt.legend(handles=[blue_patch, red_patch], fontsize=fs-2, loc="upper center", ncol=2)
        plt.tight_layout()
        
        # Save transparent graphs
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        plt.savefig(os.path.join(save_path, f"{vid_name}_barh_presence_graph.png"), dpi=600, transparent=True)
        plt.close()