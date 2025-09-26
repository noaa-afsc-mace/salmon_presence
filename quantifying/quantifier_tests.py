import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from DetectNum import DetectNum

class TestDetectNumQuantifier(unittest.TestCase):
    def test_evaluate(self):
        # setup quantifier
        quantifier = DetectNum("detect_num", 0.7, 1, 1, 1000)
        quantifier.presence_frames = [1, 2, 3, 4, 5]
        quantifier.track_data = {1:[], 2:[]} # making this kinda fake since only length is checked
        # precision and recall zero
        gt_data = [[10,15]]
        presence_precision, presence_recall, num_predicted, num_gt, mse, true_positives, false_positives, false_negatives, num_gt_frames = quantifier.evaluate(gt_data)
        self.assertEqual(presence_precision, 0, "incorrect presence precision")
        self.assertEqual(presence_recall, 0, "incorrect presence precision")
        self.assertEqual(num_predicted, 2, "incorrect num predicted")
        self.assertEqual(num_gt, 1, "incorrect num gt")
        self.assertEqual(mse, 1, "incorrect mse")
        self.assertEqual(true_positives, 0, "incorrect true positives")
        self.assertEqual(false_positives, 5, "incorrect false positives")
        self.assertEqual(false_negatives, 6, "incorrect false negatives")
        # precision and recall one
        gt_data = [[1,5]]
        presence_precision, presence_recall, num_predicted, num_gt, mse, true_positives, false_positives, false_negatives, num_gt_frames = quantifier.evaluate(gt_data)
        self.assertEqual(presence_precision, 1, "incorrect presence precision")
        self.assertEqual(presence_recall, 1, "incorrect presence precision")
        self.assertEqual(true_positives, 5, "incorrect true positives")
        self.assertEqual(false_positives, 0, "incorrect false positives")
        self.assertEqual(false_negatives, 0, "incorrect false negatives")
        # precision and recall one, multi fish overlap
        gt_data = [[1,3], [2,5]]
        presence_precision, presence_recall, num_predicted, num_gt, mse, true_positives, false_positives, false_negatives, num_gt_frames = quantifier.evaluate(gt_data)
        self.assertEqual(presence_precision, 1, "incorrect presence precision")
        self.assertEqual(presence_recall, 1, "incorrect presence precision")
        self.assertEqual(true_positives, 5, "incorrect true positives")
        self.assertEqual(false_positives, 0, "incorrect false positives")
        self.assertEqual(false_negatives, 0, "incorrect false negatives")

    def test_individual_recall(self):
        # setup quantifier
        quantifier = DetectNum("detect_num", 0.7, 1, 1, 1000)
        quantifier.presence_frames = [1, 2, 3, 4, 5]
        quantifier.track_data = {1:[], 2:[]}
        gt_data = [[1,5]]
        total_vid_frames = 10
        individual_recall, proportion_saved_frames, num_frames_reviewed, saved_frames, num_instances, num_detected = quantifier.individual_recall(gt_data, total_vid_frames)
        self.assertEqual(individual_recall, 1, "incorrect individual recall")
        self.assertEqual(proportion_saved_frames, .5, "incorrect proportion saved frames")
        self.assertEqual(num_frames_reviewed, 5, "incorrect num frames reviewed")
        self.assertEqual(saved_frames, 5, "incorrect saved frames")
        # missed fish
        gt_data = [[1,5],[6,10]]
        individual_recall, proportion_saved_frames, num_frames_reviewed, saved_frames, num_instances, num_detected = quantifier.individual_recall(gt_data, total_vid_frames)
        self.assertEqual(individual_recall, 0.5, "incorrect individual recall")
        self.assertEqual(proportion_saved_frames, .5, "incorrect proportion saved frames")
        self.assertEqual(num_frames_reviewed, 5, "incorrect num frames reviewed")
        self.assertEqual(saved_frames, 5, "incorrect saved frames")
        # missed fish with overlap
        gt_data = [[1,3],[2,5]]
        quantifier.presence_frames = [4,5]
        individual_recall, proportion_saved_frames, num_frames_reviewed, saved_frames, num_instances, num_detected = quantifier.individual_recall(gt_data, total_vid_frames)
        self.assertEqual(individual_recall, 0.5, "incorrect individual recall")
        self.assertEqual(proportion_saved_frames, .8, "incorrect proportion saved frames")
        self.assertEqual(num_frames_reviewed, 2, "incorrect num frames reviewed")
        self.assertEqual(saved_frames, 8, "incorrect saved frames")

    def test_get_results(self):
        # setup DetectNum quantifier
        quantifier = DetectNum("detect_num", 0.7, 1, 1, 1000)
        quantifier.presence_frames = [0,1,2,3,4,5]
        presence_list = quantifier.get_results_start_end()
        self.assertEqual(presence_list, [[0,5]])
        # test with multiple chunks
        quantifier.presence_frames = [0,1,2,4,5]
        presence_list = quantifier.get_results_start_end()
        self.assertEqual(presence_list, [[0,2],[4,5]])
        # test with single frame detections
        quantifier.presence_frames = [0,2,5]
        presence_list = quantifier.get_results_start_end()
        self.assertEqual(presence_list, [[0,0],[2,2], [5,5]])
        # test min presence length filtering
        quantifier.min_presence_length = 2
        quantifier.presence_frames = [0,2,5]
        presence_list = quantifier.get_results_start_end()
        self.assertEqual(presence_list, [])


if __name__ == '__main__':
    unittest.main()