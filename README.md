# segment-anything-annotator
We developed a python UI based on labelme and segment-anything for pixel-level annotation. It support generating multiple masks by SAM(box/point prompt),  efficient polygon modification and category record. We will add more features (such as incorporating CLIP-based methods for category proposal and VOS methods for mask association of video datasets)


## Features
- [x] Interactive Segmentation by SAM (both boxes and points prompt)
- [x] Multiple Output Choices
- [x] Category Annotation
- [x] Polygon modification
- [ ] CLIP for Category Proposal
- [ ] STCN for Video Dataset Annotation

## Demo
<img src="demo.gif" alt="Demo" width="720" height="480">

## Installation
  1. Python>=3.8
  2. [Pytorch](https://pytorch.org/)
  3. pip install -r requirements.txt

## Usage
### 1. Start the Annotation Platform

```
python annnotator.py --app_resolution 1000,1600 --model_type vit_b  #model_type in [vit_b, vit_l, vit_h], default: vit_b
```
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

## To Do
- [ ] CLIP for Category Proposal
- [ ] STCN for Video Dataset Annotation
- [ ] Fix bugs and optimize the UI

## Acknowledgement 
This repo is built on [SAM](https://github.com/facebookresearch/segment-anything) and [Labelme]().


