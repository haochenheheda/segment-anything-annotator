# Fork to fix the dependecies conflicts

I suggest using [uv](https://github.com/astral-sh/uv) as python package manager. The requirements.txt to avoid conflicts is the following, using python 3.10.12:

| Package                 | Version     |
|-------------------------|-------------|
| appdirs                 | 1.4.4       |
| beautifulsoup4          | 4.12.3      |
| black                   | 21.7b0      |
| certifi                 | 2024.12.14  |
| charset-normalizer      | 3.4.1       |
| click                   | 8.0.4       |
| colorama                | 0.4.6       |
| coloredlogs             | 15.0.1      |
| contourpy               | 1.3.1       |
| cycler                  | 0.12.1      |
| filelock                | 3.11.0      |
| flake8                  | 3.9.2       |
| flatbuffers             | 23.3.3      |
| fonttools               | 4.55.7      |
| fsspec                  | 2024.12.0   |
| gdown                   | 5.2.0       |
| humanfriendly           | 10.0        |
| idna                    | 3.10        |
| imageio                 | 2.37.0      |
| imgviz                  | 1.7.5       |
| importlib-resources     | 5.12.0      |
| isort                   | 5.9.2       |
| jinja2                  | 3.1.2       |
| kiwisolver              | 1.4.8       |
| labelme                 | 5.2.0       |
| lazy-loader             | 0.4         |
| loguru                  | 0.7.3       |
| markupsafe              | 3.0.2       |
| matplotlib              | 3.7.1       |
| mccabe                  | 0.6.1       |
| metaseg                 | 0.4.4       |
| mpmath                  | 1.3.0       |
| mypy-extensions         | 1.0.0       |
| natsort                 | 8.3.1       |
| networkx                | 3.1         |
| numpy                   | 1.23.5      |
| onnxruntime             | 1.14.1      |
| opencv-python           | 4.7.0.72    |
| osam                    | 0.2.2       |
| packaging               | 24.2        |
| pathspec                | 0.11.1      |
| pillow                  | 9.5.0       |
| protobuf                | 4.22.1      |
| pycocotools             | 2.0.6       |
| pycodestyle             | 2.7.0       |
| pydantic                | 1.10.21     |
| pyflakes                | 2.3.1       |
| pyparsing               | 3.2.1       |
| pyqt5                   | 5.15.9      |
| pyqt5-qt5               | 5.15.2      |
| pyqt5-sip               | 12.12.0     |
| pyreadline3             | 3.5.4       |
| pysocks                 | 1.7.1       |
| python-dateutil         | 2.8.2       |
| pyyaml                  | 6.0         |
| qtpy                    | 2.3.1       |
| regex                   | 2023.3.23   |
| requests                | 2.28.2      |
| scikit-image            | 0.24.0      |
| scipy                   | 1.15.1      |
| segment-anything        | 1.0         |
| setuptools              | 63.2.0      |
| six                     | 1.17.0      |
| soupsieve               | 2.6         |
| sympy                   | 1.11.1      |
| termcolor               | 2.2.0       |
| tifffile                | 2025.1.10   |
| tomli                   | 1.2.3       |
| torch                   | 2.1.2       |
| torchvision             | 0.16.2      |
| tqdm                    | 4.65.0      |
| typing-extensions       | 4.5.0       |
| urllib3                 | 1.26.20     |
| win32-setctime          | 1.2.0       |
| zipp                    | 3.15.0      |

Make sure to run in a venv

# segment-anything-annotator
We developed a python UI based on labelme and segment-anything for pixel-level annotation. It support generating multiple masks by SAM(box/point prompt),  efficient polygon modification and category record. We will add more features (such as incorporating CLIP-based methods for category proposal and VOS methods for mask association of video datasets)

Any feedback or suggestions are welcomed. We will continuously add features and fix bugs. 👀👀👀


## News
`28 Apr`: Change the output format with labelme format. If you want to use the old output format, please use `backup/annotator.py`.

`21 Apr`: Add annotation script for video dataset. See [Video Usage](https://github.com/haochenheheda/segment-anything-annotator#video-usage) for the details.
<img src="demo_jntm.gif" alt="VIdeo_Demo" width="720" height="480">
## Features
- [x] Interactive Segmentation by SAM (both boxes and points prompt)
- [x] Multiple Output Choices
- [x] Category Annotation
- [x] Polygon modification
- [ ] CLIP for Category Proposal
- [x] STCN for Video Dataset Annotation

## Demo
<img src="demo.gif" alt="Demo" width="720" height="480">

## Installation
  1. Python>=3.8
  2. [Pytorch](https://pytorch.org/)
  3. pip install -r requirements.txt

## Usage
### 1. Start the Annotation Platform

```
python annotator.py --app_resolution 1000,1600 --model_type vit_b --keep_input_size True --max_size 720
```
`--model_type`: vit_b, vit_l, vit_h

`--keep_input_size`: `True`: keep the origin image size for SAM; `False`: resize the input image to `--max_size` (save GPU memory)

### 2. Load the category list file if you want to annotate object categories.
Click the `Category File` on the top tool bar and choose your own one, such as the `categories.txt` in this repo.

### 3. Specify the image and save folds
Click the 'Image Directory' on the top tool bar to specify the fold containing images (in .jpg or .png).
Click the 'Save Directory' on the top tool bar to specify the fold for saving the annotations. The annotations of each image will be saved as json file in the following format
```
[
  #object1
  {
      'label':<category>, 
      'group_id':<id>,
      'shape_type':'polygon',
      'points':[[x1,y1],[x2,y2],[x3,y3],...]
  },
  #object2
  ...
]
```

### 4. Load SAM model
Click the "Load SAM" on the top tool bar to load the SAM model. The model will be automatically downloaded at the first time. Please be patient. Or you can manually download the [models](https://github.com/facebookresearch/segment-anything#model-checkpoints) and put them in the root directory named `vit_b.pth`, `vit_l.pth` and `vit_h.pth`.

### 5. Annotating Functions
`Manual Polygons`: manually add masks by clicking on the boundary of the objects, just like the Labelme (Press right button and drag to draw the arcs easily).

`Point Prompt`: generate mask proposals with clicks. The mouse leftpress/rightpress represent positive/negative clicks respectively.
You can see several mask proposals below in the boxes: `Proposal1-4`, and you could choose one by clicking or shortcuts `1`,`2`,`3`,`4`.

`Box Prompt`: generate mask proposals with boxes.

`Accept`(shortcut:`a`): accept the chosen proposal and add to the annotation dock.

`Reject`(shortcut:`r`): reject the proposals and clean the workspace.

`Save`(shortcut:'s'): save annotations to file. Do not forget to save your annotation for each image, or it will be lost when you switch to the next image.

`Edit Polygons`: in this mode, you could modify the annotated objects, such as changing the category labels or ids by double click on object items in the
annotation dock. And you can modify the boundary by draging the points on the boundary.

`Delete`(shortcut:'d'): under `Edit Mode`, delete selected/hightlight objects from annotation dock.

`Reduce Point`: under `Edit Mode`, if you find the polygon is too dense to edit, you could use this button to reduce the points on the selected polygons. But this will slightly reduce the annotation quality.

`Zoom in/out`: press 'CTRL' and scroll wheel on the mouse

`Class On/Off`: if the Class is turned on, a dialog will show after you accept a mask to record category and id, or the catgeory will be default value "Object".


## Video Usage
### 1. clone [STCN](https://github.com/hkchengrex/STCN), download the [stcn.pth](https://drive.google.com/file/d/1mRrE0uCI2ktdWlUgapJI_KmgeIiF2eOm/view) and put them in the root directory like this:
```
-| segment-anything-annotator
    -| annotation_video.py
    .....
    -| STCN
    -| stcn.pth
```

### 2. Start the Annotation Platform
```
python annotator_video.py --app_resolution 1000,1600 --model_type vit_b --keep_input_size True --max_size 720 --max_size_STCN 600
```
`--model_type`: vit_b, vit_l, vit_h
`--keep_input_size`: `True`: keep the origin image size for SAM; `False`: resize the input image to `--max_size` (save GPU memory)
`--max_size_STCN`: the maximum input image size for STCN (don't be too large) 

### 3. Specify the video and save folds
Click 'Video Directory' and 'Save Directory'.
The folds containing videos should be structured like this:
```
-| video_fold
    -| video1
        -| 00000.jpg
        -| 00001.jpg
        ...
    -| video2
        -| 00000.jpg
        -| 00001.jpg     
    ...
```
### 3. Load STCN and SAM
Click 'Load STCN' and 'Load SAM'.

### 4. Video Annotating Functions
(a) Finish the annotations of the first frame with SAM

(b) Press and hold `left control`, then press `left mouse button` to select the objects you want to track (should be highlighted by colors)

(c) Click `add object to memory` to initialize the tracklets

(d) move to next frame, and click `Propagate` to obtain tracked masks on new frame. (the results will be automatically saved when you change frames)

(e) if the propagated masks are not good enough, press 'e' to enter Edit mode, then you could manually correct the masks. You could also use `Add as key frame` to add a certain frame as reference to improve the propagation stability.

ShortCuts: `b`: `last frame`, `n`: `next frame`, `e`: `edit model`, `a`: `accept proposal`, `r`: `reject proposal`, `d`: `delete`, `s`: `save`, `space`: `propagate`



## To Do
- [ ] CLIP for Category Proposal
- [x] STCN for Video Dataset Annotation
- [ ] Fix bugs and optimize the UI
- [ ] Annotation Files -> Labelme Format/COCO Format/Youtube-VIS Format

## Acknowledgement 
This repo is built on [SAM](https://github.com/facebookresearch/segment-anything) and [Labelme]().


