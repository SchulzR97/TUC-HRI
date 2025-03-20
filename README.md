# TUC-HRI
Author: [Robert Schulz](mailto:schulzr256@gmail.com?subject=TUC-HRI)

This repository contains the official sources for the **University of Technology Chemnitz - Human Robot Interaction Dataset (TUC-HRI)**. The dataset is hosted on HuggingFace. A library provides easy access to the dataset by providing an implementation as `torch.utils.data.Dataset`.

## 1 Dataset Usage
To use this dataset in your own project, follow this instructions:
1. Install `rsp-ml`
```bash
pip install rsp-ml
```
2. Implement the following python code:
```python
from rsp.ml.dataset import TUCHRI

ds_train = TUCHRI(
    phase='train',
    cache_dir=dataset_directory,
    load_depth_data=use_depth_channel,
    num_classes=num_actions,
    sequence_length=sequence_length
)
ds_val= TUCHRI(
    phase='val',
    cache_dir=dataset_directory,
    load_depth_data=use_depth_channel,
    num_classes=num_actions,
    sequence_length=sequence_length
)
```

> To implement your own custom version of the dataset, take inspirations by the official implementation at https://github.com/SchulzR97/rsp-ml/blob/main/rsp/ml/dataset/dataset.py

## 2 Create your own Dataset
### 2.1 Capture
The script `src/capture.py` can be used to capture sequences. All sequences are captured synchronous for the cameras provided in cam_names and rs_devices. For each cam_name, rs_devices should contain a corresponding serial number of the RealSense camera.

Recorded sequences are stored in `dataset_directory/sequences/cap_mode/new`


| Argument   | Type  | Description                              |
|------------|-------|------------------------------------------|
| cap_mode   | str   | currently just `realsense` is supported. |
| cam_names  | str   | list of camera names. Each camera name should corrspond to one rs_device |
| rs_devices | str   | list of RealSense serial numbers         |
| max_w_h    | int   | max side length of recorded frames |
| dataset_directory  | str | directory, in which the frames are stored |
| framerate  | float | framerate of the captured sequences |
| frame_buffer_size | int | Frames are stored in RAM buffer to improve performance. When `frame_buffer_size` is reached, frames are written to `dataset_directory` and buffer gets cleared to avoid memory overflow. |

### 2.2 Annotate
The script `src/annotate.py` can be used to annotate captured sequences. It loads data from `dataset_directory/sequences/cap_mode/new` and provides an OpenCV based UI. Frames are moved to sub directories named in the following way:
AaaaCcccSsssSEQseqseqseq

example: A001C002S003SEQ099

> Please provide a `SUBJECTS.txt` and a `LABELS.txt` in `dataset_directory/sequences/cap_mode`. `SUBJECTS.txt` should contain a line for each subject. `LABELS.txt` should contain a line for each action label.

| Argument           | Type  | Description                              |
|--------------------|-------|------------------------------------------|
| cap_mode           | str   | currently just `realsense` is supported. |
| dataset_directory  | str   | directory, in which the frames are stored  |

### 2.3 Preprocess
The script `src/preprocess.py` can be used to remove the backgrounds of the labeled sequences. First, a hsv filter is applied to remove green screen backgrounds. Second, YOLO is used to segment persons in the scene to remove all outer area backgrounds, that are not covered by the green screen.

| Argument           | Type  | Description                              |
|--------------------|-------|------------------------------------------|
| cap_mode           | str   | currently just `realsense` is supported. |
| dataset_directory  | str   | directory, in which the frames are stored  |
| out_dir            | str   | directory, where the preprocessed dataset is stored. |

### 2.4 Publish
The script `src/publish.py` can be used to upload the dataset to HuggingFace.

| Argument          | Type | Description                                      |
|-------------------|------|--------------------------------------------------|
| dataset_directory | str  | directory, in which the frames are stored        |
| cache_directory   | str  | cache directory -> temporary use                 |
| huggingface_token | str  | your hugging face token                          |
| huggingface_repo_id | str | your hugging face repository id |
| upload_sequences | int | upload sequences [0, 1] |
| upload_metadata   | int  | upload metadate [0, 1] |
