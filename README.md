# Automated salmon presence prediction

![Salmon presence graph](final_results/experiments/detect_num-yolo_11_best_clips-0.2-2-1/T16_REC_0001_barh_presence_graph.png)


This repository contains the code for salmon presence prediction and evaluation methods as detailed in **Automated fish detection in videos to support commercial
fishing sustainability and innovation in the Alaska walleye pollock (_Gadus chalcogrammus_) trawl fishery**

[[ICES publication](https://academic.oup.com/icesjms/article/82/9/fsaf168/8262706)]

----

To replicate our [`final_results`](final_results):
1. Generate detections using our pre-trained models
2. Run parameter optimization [optional]
3. Run the salmon presence detection script [`salmost.py`](salmost.py)

## Setup
1. Install dependencies with `pip install requirements.txt`
2. Download the videos of tows 16, 18, and 20 from our [google cloud bucket](https://console.cloud.google.com/storage/browser/nmfs_odp_afsc/RACE/MACE/salmon_pollock_object_detection/tow_data/videos?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))).

## Methods

### Preparing the data

There are two important types of data you need: salmon detections and ground-truth presence annotations. Presence annotations are already saved in this repository.

#### Generate detection results
1. Run [`generate_detection.py`](generate_detections.py) to generate detections for the videos. See the file for documentation.
    - Our pre-trained models are in [`detectors/`](detection/detectors). You can select between our salmon-only or salmon and pollock model. Pre-trained model weights are also available in our [cloud bucket](https://console.cloud.google.com/storage/browser/nmfs_odp_afsc/RACE/MACE/salmon_pollock_object_detection/models/yolo11;tab=objects?prefix=&forceOnObjectsSortingFiltering=false).
    - The minimum detection confidence for detectors is set in [`model_info.json`](detection/model_info.json). For all models it is 0.0
2. Compress detection results using [`make_pickle_eat_sandwich.py`](utils/make_pickle_eat_sandwich.py). We compress results to speed up the optimization/evaluation process. 
    - Note that the save path for these files is specified in [`salmost.py`](salmost.py). You will need to update this.

#### Ground truth salmon presence data

All presence annotations are already processed and saved in [`quantifying_data/presence_data/all_presence_data`](quantifying_data/presence_data/all_presence_data). **You don't need to do anything unless you are using your own annotations or changing ours.**

Annotations of salmon presence are also stored in csv files (and spreadsheets for readability), with a different file for [tow 16](quantifying_data/presence_data/google_sheet_conversion_stuff/sheets/2019_tow16_salmon_times.xlsx), [tow 18](quantifying_data/presence_data/google_sheet_conversion_stuff/sheets/2019_tow18_salmon_times.xlsx), and [tow 20](quantifying_data/presence_data/google_sheet_conversion_stuff/sheets/2019_tow20_salmon_times.xlsx). See our [[ICES publication](https://academic.oup.com/icesjms/article/82/9/fsaf168/8262706)] for details about annotation methods.

If changes are made to the annotations in the spreadsheets, you need to do the following to convert those sheets to the data format necessary for analysis:

1. Convert the sheet for each tow to a CSV and copy them into [`quantifying_data/presence_data/google_sheet_conversion_stuff/sheets/`](quantifying_data/presence_data/google_sheet_conversion_stuff/sheets)
2. Run [`google_sheet_splitter.py`](quantifying_data/presence_data/google_sheet_conversion_stuff/google_sheet_splitter.py) for each file, making sure to change `GOOGLE_SHEET` and `TOW_NUM` each time. Data will be saved in [`all_presence_data/`](quantifying_data/presence_data/all_presence_data)

### Running the experiment

#### Optimization

*NOTE: for our paper, we pick the detection confidence score based on the graphs of conf vs precision and recall. We do use optimization to identify min presence threshold and num detections to use.*

##### 1. Generate eval and tuning dataset

*WARNING*: DON'T DO THIS UNLESS YOU REALLY NEED TO - IT WILL MAKE ALL OTHER EXPERIMENTS INCOMPARABLE

Run [`eval_tune_splitter.py`](quantifying_data/presence_data/eval_tune_splitter.py)

This file splits data in `all_presence_data` into tune and eval and creates .txt files for each in [`eval/`](quantifying_data/presence_data/eval) and [`tune/`](quantifying_data/presence_data/tune). Files are grouped by tow. `TUNE_SPLIT` portion of videos in each tow are randomly assigned to tune, the rest are assigned to eval. All videos in a tow are randomized and the first n are selected for tuning for each tuning size. This means that all tuning sets are a subset of the largest tuning set and all eval sets contain the smallest eval set.

##### 2. Run the optimizer

*NOTE*: Parameters are optimized based on two metrics: individual recall and proportion of saved frames. These are assigned weights during optimization, which influence optimization significantly. To adjust the weights, change the values for `INDIVIDUAL_RECALL_WEIGHT` and `PROPORTION_OF_FRAMES_SAVED_WEIGHT`

Run [`detect_num_optimizer.py`](detect_num_optimizer.py) to find optimal values for the `DetectNum` method. Results are saved in [`optimizing/`](quantifying_data/optimizing). See [`final_results/optimizing`](final_results/optimizing) for an example of the outputs.

Manually adjust parameters ranges if necessary. Note that the path to compressed detections is imported from [`salmost.py`](salmost.py).

##### 2. Graph detection confidence threshold vs saved frames and individual recall

From optimization, we determine the ideal minimum presence length and detections per frame. Update those in [`graph_confidence.py`](graph_confidence.py) and run the file to generate a graph of detection confidence threshold vs saved frames and individual recall. The graphs are saved in [`optimizing/`](quantifying_data/optimizing). You can use the graph to pick the ideal confidence threshold. See [`final_results/optimizing`](final_results/optimizing) for an example of the outputs.

Note that the path to the eval file is imported from [`detect_num_optimizer.py`](detect_num_optimizer.py).

##### 3. Update optimal parameters

Update the default values for parameters in [`salmost.py`](salmost.py), or just make sure to use them as arguments when running [`salmost.py`](salmost.py).

#### Getting salmon presence detection predictions

Run [`salmost.py`](salmost.py).

Make sure to update the paths to compressed detections and videos.

All results, including graphs, will be saved under [`quantifying_data\experiments`](quantifying_data/experiments). See [`final_results/experiments`](final_results/experiments) for an example of the outputs.

## Using new detection models

Tensorflow and Ultralytics YOLO models are currently supported. To add a new model, simply add the model to [`model_info.json`](detection/model_info.json).

## Testing

Always a good idea to write some tests! To run tests for the quantifier methods use:

```
python -m pytest quantifying/quantifier_tests.py -v
```

## License
This repository is released under the MIT license as found in the [LICENSE](LICENSE) file.

## Citation

If you find this repository useful, please consider giving a star ⭐ and citation 🦖:

```
@article{Wilson_Automated_fish_detection_2025,
author = {Wilson, Katherine and Lurbur, Moses and Yochum, Noëlle},
doi = {10.1093/icesjms/fsaf168},
journal = {ICES Journal of Marine Science},
title = {{Automated fish detection in videos to support commercial fishing sustainability and innovation in the Alaska walleye pollock (Gadus chalcogrammus) trawl fishery}},
year = {2025}
}
```

## Disclaimer
This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project content is provided on an "as is" basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.
