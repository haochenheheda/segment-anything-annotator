import sys
import functools
import cv2
import glob
import os
import os.path as osp
import imgviz
import html
import json
import math
import argparse
import numpy as np
import tempfile
import torch
import torch.nn.functional as F
import base64

from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QApplication, QPushButton, QLabel, QFileDialog, QProgressBar, QComboBox, QScrollArea, QDockWidget, QMessageBox
from PyQt5.QtGui import QPixmap, QIcon, QImage
from PyQt5.Qt import QSize
from qtpy.QtCore import Qt
from qtpy import QtCore
from qtpy import QtGui, QtWidgets
from canvas import Canvas
import utils
from utils.download_model import download_model

from labelme.widgets import ToolBar, UniqueLabelQListWidget, LabelDialog, LabelListWidget, LabelListWidgetItem, ZoomWidget
from labelme import PY2
from labelme.label_file import LabelFile
from labelme.label_file import LabelFileError


from shape import Shape

from PIL import Image

from collections import namedtuple
Click = namedtuple('Click', ['is_positive', 'coords'])

from segment_anything import sam_model_registry, SamPredictor

sys.path.insert(0, 'STCN/')
from model.eval_network import STCN
import torch
from torchvision import transforms
from dataset.range_transform import im_normalization
from app_inference_core import InferenceCore
from util.tensor_util import unpad, pad_divide_by



LABEL_COLORMAP = imgviz.label_colormap()

class MainWindow(QMainWindow):

    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = 0, 1, 2

    def __init__(self, parent=None, global_w=1000, global_h=1800, model_type='vit_b', keep_input_size=True, max_size=720,max_size_STCN=600):
        super(MainWindow, self).__init__(parent)
        self.resize(global_w, global_h)
        self.model_type = model_type
        self.keep_input_size = keep_input_size
        self.max_size = float(max_size)
        self.max_size_STCN = float(max_size_STCN)

        self.setWindowTitle('segment-anything-annotator')
        self.canvas = Canvas(self,
            epsilon=10.0,
            double_click='close',
            num_backups=10,
            app=self,
        )

        
        self._noSelectionSlot = False
        self.current_output_dir = 'output'
        os.makedirs(self.current_output_dir, exist_ok=True)
        self.current_output_filename = ''
        self.canvas.zoomRequest.connect(self.zoomRequest)

        self.memory_shapes = []
        self.sam_mask = []
        self.sam_mask_proposal = []
        self.image_encoded_flag = False
        self.min_point_dis = 4

        self.predictor = None
        self.prop_model = None
        self.processor = None
        self.tracked_id_list = None
        self.tracked_label_list = None

        self.scroll_values = {
            Qt.Horizontal: {},
            Qt.Vertical: {},
        }
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidget(self.canvas)
        self.scrollArea.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: self.scrollArea.verticalScrollBar(),
            Qt.Horizontal: self.scrollArea.horizontalScrollBar(),
        }
        self.canvas.scrollRequest.connect(self.scrollRequest)
        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)

        self.uniqLabelList = UniqueLabelQListWidget()
        self.uniqLabelList.setToolTip(
            self.tr(
                "Select label to start annotating for it. "
                "Press 'Esc' to deselect."
            )
        )
        self.labelDialog = LabelDialog(
            parent=self,
            labels=[],
            sort_labels=False,
            show_text_field=True,
            completion='contains',
            fit_to_content={'column': True, 'row': False},
        )

        self.labelList = LabelListWidget()
        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.itemDoubleClicked.connect(self.editLabel)
        self.labelList.itemChanged.connect(self.labelItemChanged)
        self.labelList.itemDropped.connect(self.labelOrderChanged)

        self.shape_dock = QDockWidget(
            self.tr("Polygon Labels"), self
        )
        self.shape_dock.setObjectName("Labels")
        self.shape_dock.setWidget(self.labelList)

        self.category_list = [i.strip() for i in open('categories.txt', 'r', encoding='utf-8').readlines()]
        self.labelDialog = LabelDialog(
            parent=self,
            labels=self.category_list,
            sort_labels=False,
            show_text_field=True,
            completion='contains',
            fit_to_content={'column': True, 'row': False},
        )
        self.zoom_values = {}
        self.video_directory = ''
        self.video_list = []
        self.video_len = len(self.video_list)

        self.img_list = []
        self.img_len = len(self.img_list)
        self.current_img_index = 0
        self.current_img = ''
        self.current_img_data = ''

        self.video_list = []
        self.video_len = len(self.video_list)
        self.current_video_index = 0
        self.current_video = ''


        self.button_next = QPushButton('Next Image (n)', self)
        self.button_next.clicked.connect(self.clickButtonNext)
        self.button_next.setShortcut('n')
        self.button_last = QPushButton('Last Image (b)', self)
        self.button_last.clicked.connect(self.clickButtonLast)
        self.button_last.setShortcut('b')

        self.button_next_video = QPushButton('Next Video', self)
        self.button_next_video.clicked.connect(self.clickButtonNextVideo)
        self.button_last_video = QPushButton('Last Video', self)
        self.button_last_video.clicked.connect(self.clickButtonLastVideo)

        self.img_progress_bar = QProgressBar(self)
        self.img_progress_bar.setMinimum(0)
        self.img_progress_bar.setMaximum(1)
        self.img_progress_bar.setValue(0)

        self.video_progress_bar = QProgressBar(self)
        self.video_progress_bar.setMinimum(0)
        self.video_progress_bar.setMaximum(1)
        self.video_progress_bar.setValue(0)

        self.button_proposal1 = QPushButton('Proposal1', self)
        self.button_proposal1.clicked.connect(self.choose_proposal1)
        self.button_proposal1.setShortcut('1')
        self.button_proposal2 = QPushButton('Proposal2', self)
        self.button_proposal2.clicked.connect(self.choose_proposal2)
        self.button_proposal2.setShortcut('2')
        self.button_proposal3 = QPushButton('Proposal3', self)
        self.button_proposal3.clicked.connect(self.choose_proposal3)
        self.button_proposal3.setShortcut('3')
        self.button_proposal_list = [self.button_proposal1, self.button_proposal2, self.button_proposal3]
        
        
        self.button_add_track = QPushButton('Add objects to memory', self)
        self.button_add_track.clicked.connect(self.clickAddMemory)
        self.button_add_key_frame = QPushButton('Add as key frame', self)
        self.button_add_key_frame.clicked.connect(self.clickAddKeyFrame)
        self.button_propagate = QPushButton('Propagate (SPACE)', self)
        self.button_propagate.setShortcut(' ')
        self.button_propagate.clicked.connect(self.clickPropagate)
        self.button_clear_track_memory = QPushButton('Clear track memory', self)
        self.button_clear_track_memory.clicked.connect(self.clickClearTrackMemory)
        
        self.class_on_flag = True
        self.class_on_text = QLabel("Class On", self)
        self.tracked_object_text = QLabel("Tracked_object:", self)

        #naive layout
        self.scrollArea.move(int(0.02 * global_w), int(0.08 * global_h))
        self.scrollArea.resize(int(0.75 * global_w), int(0.7 * global_h))
        self.shape_dock.move(int(0.79 * global_w), int(0.08 * global_h))
        self.shape_dock.resize(int(0.2 * global_w), int(0.7 * global_h))
        self.button_next.move(int(0.14 * global_w), int(0.82 * global_h))
        self.button_next.resize(int(0.13 * global_w),int(0.04 * global_h))
        self.button_last.move(int(0.01 * global_w), int(0.82 * global_h))
        self.button_last.resize(int(0.13 * global_w),int(0.04 * global_h))
        self.button_next_video.resize(int(0.13 * global_w),int(0.04 * global_h))
        self.button_next_video.move(int(0.14 * global_w), int(0.89 * global_h))
        self.button_last_video.resize(int(0.13 * global_w),int(0.04 * global_h))
        self.button_last_video.move(int(0.01 * global_w), int(0.89 * global_h))
        self.img_progress_bar.move(int(0.01 * global_w), int(0.8 * global_h))
        self.img_progress_bar.resize(int(0.25 * global_w),int(0.02 * global_h))
        self.video_progress_bar.move(int(0.01 * global_w), int(0.87 * global_h))
        self.video_progress_bar.resize(int(0.25 * global_w),int(0.02 * global_h))

        self.button_proposal1.resize(int(0.17 * global_w),int(0.14 * global_h))
        self.button_proposal1.move(int(0.27 * global_w), int(0.8 * global_h))
        self.button_proposal2.resize(int(0.17 * global_w),int(0.14 * global_h))
        self.button_proposal2.move(int(0.44 * global_w), int(0.8 * global_h))
        self.button_proposal3.resize(int(0.17 * global_w),int(0.14 * global_h))
        self.button_proposal3.move(int(0.61 * global_w), int(0.8 * global_h))

        
        self.class_on_text.move(int(0.8 * global_w), int(0.78 * global_h))
        self.tracked_object_text.resize(int(0.2 * global_w), int(0.04 * global_h))
        self.tracked_object_text.move(int(0.8 * global_w), int(0.81 * global_h))
        self.button_add_track.resize(int(0.2 * global_w), int(0.04 * global_h))
        self.button_add_track.move(int(0.8 * global_w), int(0.85 * global_h))
        self.button_add_key_frame.resize(int(0.2 * global_w), int(0.04 * global_h))
        self.button_add_key_frame.move(int(0.8 * global_w), int(0.89 * global_h))
        self.button_propagate.resize(int(0.2 * global_w), int(0.04 * global_h))
        self.button_propagate.move(int(0.8 * global_w), int(0.93 * global_h))
        self.button_clear_track_memory.resize(int(0.2 * global_w), int(0.03 * global_h))
        self.button_clear_track_memory.move(int(0.8 * global_w), int(0.97 * global_h))
        
        self.zoomWidget = ZoomWidget()

        action = functools.partial(utils.newAction, self)
        

        categoryFile = action(
            self.tr("Category File"),
            lambda: self.clickCategoryChoose(),
            'None',
            "objects",
            self.tr("Category File"),
            enabled=True,
        )
        imageDirectory = action(
            self.tr("Video Directory"),
            lambda: self.clickFileChoose(),
            'None',
            "objects",
            self.tr("Video Directory"),
            enabled=True,
        )
        LoadSAM = action(
            self.tr("Load SAM"),
            lambda: self.clickLoadSAM(),
            'None',
            "objects",
            self.tr("Load SAM"),
            enabled=True,
        )
        LoadSTCN = action(
            self.tr("Load STCN"),
            lambda: self.clickLoadSTCN(),
            'None',
            "objects",
            self.tr("Load STCN"),
            enabled=True,
        )
        AutoSeg = action(
            self.tr("AutoSeg"),
            lambda: self.clickAutoSeg(),
            'None',
            "objects",
            self.tr("AutoSeg"),
            enabled=False,
        )
        promptSeg = action(
            self.tr("(A)ccept"),
            lambda: self.addSamMask(),
            'a',
            "objects",
            self.tr("(A)ccept"),
            enabled=False,
        )

        saveDirectory = action(
            self.tr("Save Directory"),
            lambda: self.clickSaveChoose(),
            'None',
            "objects",
            self.tr("Save Directory"),
            enabled=True,
        )

        createMode = action(
            self.tr("Manual Polygons"),
            lambda: self.toggleDrawMode(False, createMode="polygon"),
            'Ctrl+W',
            "objects",
            self.tr("Start drawing polygons"),
            enabled=True,
        )
        createPointMode = action(
            self.tr("Point Prompt"),
            lambda: self.toggleDrawMode(False, createMode="point"),
            'None',
            "objects",
            self.tr("Point Prompt"),
            enabled=True,
        )
        createRectangleMode = action(
            self.tr("Box Prompt"),
            lambda: self.toggleDrawMode(False, createMode="rectangle"),
            'None',
            "objects",
            self.tr("Box Prompt"),
            enabled=True,
        )
        cleanPrompt = action(
            self.tr("(R)eject"),
            lambda: self.cleanPrompt(),
            'r',
            "objects",
            self.tr("(R)eject"),
            enabled=True,
        )
        
        self.switchClass = action(
            self.tr("Class On/Off"),
            lambda: self.clickSwitchClass(),
            'none',
            "objects",
            self.tr("Class On/Off"),
            enabled=True,
        )

        editMode = action(
            self.tr("(E)dit Polygons"),
            self.setEditMode,
            'e',
            "edit",
            self.tr("Move and edit the selected polygons"),
            enabled=False,
        )
        saveAs = action(
            self.tr("&Save As"),
            self.saveFileAs,
            'ALT+s',
            "save-as",
            self.tr("Save labels to a different file"),
            enabled=True,
        )


        hideAll = action(
            self.tr("&Hide\nPolygons"),
            functools.partial(self.togglePolygons, False),
            icon="eye",
            tip=self.tr("Hide all polygons"),
            enabled=False,
        )
        showAll = action(
            self.tr("&Show\nPolygons"),
            functools.partial(self.togglePolygons, True),
            icon="eye",
            tip=self.tr("Show all polygons"),
            enabled=False,
        )


        save = action(
            self.tr("&(S)ave"),
            self.saveFile,
            'S',
            "save",
            self.tr("Save labels to file"),
            enabled=False,
        )

        delete = action(
            self.tr("(D)elete Polygons"),
            self.deleteSelectedShape,
            'd',
            "cancel",
            self.tr("Delete the selected polygons"),
            enabled=False,
        )
        reduce_point = action(
            self.tr("Reduce Points"),
            self.reducePoint,
            'None',
            "copy",
            self.tr("Reduce Points"),
            enabled=True,
        )


        self.actions = utils.struct(
            categoryFile=categoryFile,
            imageDirectory=imageDirectory,
            saveDirectory=saveDirectory,
            switchClass=self.switchClass,
            loadSTCN=LoadSTCN,
            loadSAM=LoadSAM,
            #autoSeg=AutoSeg,
            promptSeg=promptSeg,
            cleanPrompt=cleanPrompt,
            createMode=createMode,
            createPointMode=createPointMode,
            createRectangleMode=createRectangleMode,
            editMode=editMode,
            delete=delete,
            reduce_point=reduce_point,
            save=save,
            onShapesPresent=(saveAs, hideAll, showAll),
            menu=(
                createMode,
                editMode,
                save,
            )
            )

        # Custom context menu for the canvas widget:
        utils.addActions(self.canvas.menus[0], self.actions.menu)
        utils.addActions(
            self.canvas.menus[1],
            (
                action("&Copy here", self.copyShape),
                action("&Move here", self.moveShape),
            ),
        )

        self.toolbar = self.addToolBar('Tool')
        self.toolbar.addAction(categoryFile)
        self.toolbar.addAction(imageDirectory)
        self.toolbar.addAction(saveDirectory)
        self.toolbar.addAction(self.switchClass)
        self.toolbar.addAction(LoadSTCN)
        self.toolbar.addAction(LoadSAM)
        #self.toolbar.addAction(AutoSeg)
        self.toolbar.addAction(promptSeg)
        self.toolbar.addAction(cleanPrompt)
        self.toolbar.addAction(createMode)
        self.toolbar.addAction(createPointMode)
        self.toolbar.addAction(createRectangleMode)
        self.toolbar.addAction(editMode)
        self.toolbar.addAction(delete)
        self.toolbar.addAction(reduce_point)
        self.toolbar.addAction(save)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextOnly)

        zoom = QtWidgets.QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(
            str(
                self.tr(
                    "Zoom in or out of the image. Also accessible with "
                    "{} from the canvas."
                )
            ).format(
                #utils.fmtShortcut(
                #    "{},{}".format(shortcuts["zoom_in"], shortcuts["zoom_out"])
                #),
                utils.fmtShortcut(self.tr("Ctrl+Wheel")),
            )
        )
        self.zoomWidget.setEnabled(True)

        self.zoomWidget.valueChanged.connect(self.paintCanvas)
        self.canvas.actions = self.actions


    def saveFileAs(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        self._saveFile(self.saveFileDialog())

    def saveFile(self, _value=False):
        # assert not self.image.isNull(), "cannot save empty image"
        # if self.labelFile:
        #     # DL20180323 - overwrite when in directory
        #     self._saveFile(self.labelFile.filename)
        # elif self.output_file:
        #     self._saveFile(self.output_file)
        #     self.close()
        # else:
        #     self._saveFile(self.saveFileDialog())
        #self._saveFile(self.saveFileDialog())
        #print(self.current_output_filename)
        self._saveFile(self.current_output_filename)

    def _saveFile(self, filename):
        video_name = os.path.basename(os.path.basename(self.current_video))
        os.makedirs(os.path.join(self.current_output_dir, video_name), exist_ok=True)
        if filename and self.saveLabels(filename):
            self.setClean()

    def saveLabels(self, filename):
        lf = LabelFile()

        def format_shape(s):
            data = s.other_data.copy()
            data.update(
                dict(
                    label=s.label.encode("utf-8") if PY2 else s.label,
                    points=[[p.x(), p.y()] for p in s.points],
                    group_id=s.group_id,
                    description="",
                    shape_type=s.shape_type,
                    flags=s.flags,
                )
            )
            return data

        shapes = [format_shape(item.shape()) for item in self.labelList]
        imageData = base64.b64encode(self.current_img_data).decode("utf-8")
        save_data = {
            "version": "1.0.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": self.current_img,
            "imageData": imageData,
            "imageHeight": self.raw_h,
            "imageWidth": self.raw_w
        }

        with open(filename, 'w') as f:
            json.dump(save_data, f)
        return True

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.createMode.setEnabled(True)

    def saveFileDialog(self):
        caption = self.tr("Choose File")
        filters = self.tr("Label files")
        if self.output_dir:
            dlg = QtWidgets.QFileDialog(
                self, caption, self.output_dir, filters
            )
        else:
            dlg = QtWidgets.QFileDialog(
                self, caption, self.currentPath(), filters
            )
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
        basename = os.path.basename(self.current_img)[:-4]
        if self.output_dir:
            default_labelfile_name = osp.join(
                self.output_dir, basename + LabelFile.suffix
            )
        else:
            default_labelfile_name = osp.join(
                self.currentPath(), basename + LabelFile.suffix
            )
        filename = dlg.getSaveFileName(
            self,
            self.tr("Choose File"),
            default_labelfile_name,
            self.tr("Label files (*%s)") % LabelFile.suffix,
        )
        if isinstance(filename, tuple):
            filename, _ = filename
        return filename

    def currentPath(self):
        #return osp.dirname(str(self.filename)) if self.filename else "."
        return "."

    def loadAnno(self, filename):
        with open(filename,'r') as f:
            data = json.load(f)
        for shape in data['shapes']:
            label = shape["label"]
            try:
                ttt = int(label)
                label = self.category_list[ttt]
            except:
                pass

            points = shape["points"]
            shape_type = shape["shape_type"]
            flags = shape["flags"]
            group_id = shape["group_id"]
            if not points:
                # skip point-empty shape
                continue
            shape = Shape(
                label=label,
                shape_type=shape_type,
                group_id=group_id,
                flags=flags
            )
            for x, y in points:
                shape.addPoint(QtCore.QPointF(x, y))
            shape.close()
            self.addLabel(shape)
        self.canvas.loadShapes([item.shape() for item in self.labelList])

    def clickButtonNext(self):
        if self.actions.save.isEnabled():
            self.saveFile()
        if self.current_img_index < self.img_len - 1:
            self.current_img_index += 1
            self.current_img = self.img_list[self.current_img_index]
            self.loadImg()
            

    def clickButtonLast(self):
        if self.actions.save.isEnabled():
            self.saveFile()
        if self.current_img_index > 0:
            self.current_img_index -= 1
            self.current_img = self.img_list[self.current_img_index]
            self.loadImg()

    def clickButtonNextVideo(self):
        if self.actions.save.isEnabled():
            self.saveFile()
        if self.current_video_index < self.video_len - 1:
            self.current_video_index += 1
            self.current_video = self.video_list[self.current_video_index]
            self.video_progress_bar.setValue(self.current_video_index)
            self.img_list = glob.glob(os.path.join(self.current_video, '*.jpg')) + glob.glob(os.path.join(self.current_video, '*.png'))
            self.img_list.sort()
            self.img_len = len(self.img_list)
            self.current_img_index = 0
            self.current_img = self.img_list[self.current_img_index]
            self.loadImg()
            self.clickClearTrackMemory()

    def clickButtonLastVideo(self):
        if self.actions.save.isEnabled():
            self.saveFile()
        if self.current_video_index > 0:
            self.current_video_index -= 1
            self.current_video = self.video_list[self.current_video_index]
            self.video_progress_bar.setValue(self.current_video_index)
            self.img_list = glob.glob(os.path.join(self.current_video, '*.jpg')) + glob.glob(os.path.join(self.current_video, '*.png'))
            self.img_list.sort()
            self.img_len = len(self.img_list)
            self.current_img_index = 0
            self.current_img = self.img_list[self.current_img_index]
            self.loadImg()
            self.clickClearTrackMemory()


    def choose_proposal1(self):
        if len(self.sam_mask_proposal) > 0:
            self.sam_mask = self.sam_mask_proposal[0]
            self.canvas.setHiding()
            self.canvas.update()

    def choose_proposal2(self):
        if len(self.sam_mask_proposal) > 1:
            self.sam_mask = self.sam_mask_proposal[1]
            self.canvas.setHiding()
            self.canvas.update()
            
    def choose_proposal3(self):
        if len(self.sam_mask_proposal) > 2:
            self.sam_mask = self.sam_mask_proposal[2]
            self.canvas.setHiding()
            self.canvas.update()
            
    def choose_proposal4(self):
        if len(self.sam_mask_proposal) > 3:
            self.sam_mask = self.sam_mask_proposal[3]
            self.canvas.setHiding()
            self.canvas.update()
            
    def loadImg(self):
        self.raw_h, self.raw_w = cv2.imread(self.current_img).shape[:2]
        pixmap = QPixmap(self.current_img)
        #pixmap = pixmap.scaled(int(0.75 * global_w), int(0.7 * global_h))
        self.canvas.loadPixmap(pixmap)

        img_name = os.path.basename(self.current_img)[:-4]
        video_name = os.path.basename(self.current_video)
        self.current_output_filename = os.path.join(self.current_output_dir, video_name, img_name + '.json')
        self.labelList.clear()
        if os.path.isfile(self.current_output_filename):
            self.loadAnno(self.current_output_filename)
        self.image_encoded_flag = False
        
        self.img_progress_bar.setMinimum(0)
        self.img_progress_bar.setMaximum(self.img_len-1)
        self.img_progress_bar.setValue(self.current_img_index)

        self.current_img_data = LabelFile.load_image_file(self.current_img)

    def clickFileChoose(self):
        directory = QFileDialog.getExistingDirectory(self, 'choose target fold','.')
        if directory == '':
            return
        #self.img_list = glob.glob(directory + '/*.{jpg,png,JPG,PNG}')
        self.video_list = glob.glob(directory + '/*')
        self.video_list.sort()
        self.video_len = len(self.video_list)
        if self.video_len == 0:
            return
        self.video_progress_bar.setMinimum(0)
        self.video_progress_bar.setMaximum(self.video_len-1)
        self.current_video_index = 0
        self.current_video = self.video_list[self.current_video_index]
        self.img_list = glob.glob(self.video_list[0] + '/*.jpg') + glob.glob(self.video_list[0] + '/*.png')
        self.img_list.sort()
        self.img_len = len(self.img_list)
        if self.img_len == 0:
            return
        self.current_img_index = 0
        self.current_img = self.img_list[self.current_img_index]
        self.loadImg()

    def clickSaveChoose(self):
        directory = QFileDialog.getExistingDirectory(self, 'choose target fold','.')
        if directory == '':
            return
        else:
            self.current_output_dir = directory
            os.makedirs(self.current_output_dir, exist_ok=True)
            self.loadImg()
            return directory


    def clickSwitchClass(self):
        if self.class_on_flag:
            self.class_on_flag = False
            self.class_on_text.setText('Class Off')
        else:
            self.class_on_flag = True
            self.class_on_text.setText('Class On')


    def clickCategoryChoose(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'choose target file','.')
        try:
            with open(filename, 'r') as f:
                data = f.readlines()
                self.category_list = [i.strip() for i in data]
                self.category_list.sort()
                self.labelDialog = LabelDialog(
                    parent=self,
                    labels=self.category_list,
                    sort_labels=False,
                    show_text_field=True,
                    completion='contains',
                    fit_to_content={'column': True, 'row': False},
                )
        except Exception as e:
            pass

    def clickLoadSAM(self):
        download_model(self.model_type)
        self.sam = sam_model_registry[self.model_type](checkpoint='{}.pth'.format(self.model_type))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        self.actions.loadSAM.setEnabled(False)
        #self.actions.autoSeg.setEnabled(True)
        self.actions.promptSeg.setEnabled(True)

    def clickLoadSTCN(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prop_model = STCN().to(device=self.device).eval()
        # Performs input mapping such that stage 0 model can be loaded
        prop_saved = torch.load('stcn.pth',map_location=torch.device(self.device))
        for k in list(prop_saved.keys()):
            if k == 'value_encoder.conv1.weight':
                if prop_saved[k].shape[1] == 4:
                    pads = torch.zeros((64,1,7,7), device=self.device)
                    prop_saved[k] = torch.cat([prop_saved[k], pads], 1)
        self.prop_model.load_state_dict(prop_saved)

        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])
        self.actions.loadSTCN.setEnabled(False)

    def clickAddMemory(self):
        if self.prop_model == None:
            return
        def format_shape(s):
            data = s.other_data.copy()
            data.update(
                dict(
                    label=s.label.encode("utf-8") if PY2 else s.label,
                    points=[(p.x(), p.y()) for p in s.points],
                    group_id=s.group_id,
                    shape_type=s.shape_type,
                    flags=s.flags,
                )
            )
            return data
        if len(self.img_list) != 0 and len(self.canvas.selectedShapes) > 0:
            rgb = torch.stack([self.im_transform(Image.open(tmp_img).convert('RGB')) for tmp_img in self.img_list],0)
            self.raw_h,self.raw_w = rgb.shape[-2:]
            scale_ratio = self.max_size_STCN / max(self.raw_h, self.raw_w)
            self.scaled_h, self.scaled_w = int(self.raw_h * scale_ratio), int(self.raw_w * scale_ratio)
            
            rgb = F.interpolate(rgb, size=(self.scaled_h, self.scaled_w), mode='bilinear').unsqueeze(0)
            self.memory_shapes = [format_shape(item) for item in self.canvas.selectedShapes]
            used_mask_dic = {}
            for shape in self.memory_shapes:
                mask = torch.Tensor(self.polygon2mask(shape['points'],(self.raw_h,self.raw_w)))
                if shape['group_id'] in used_mask_dic.keys():
                    used_mask_dic[shape['group_id']]['mask'] = used_mask_dic[shape['group_id']]['mask'] + mask
                    used_mask_dic[shape['group_id']]['mask'][used_mask_dic[shape['group_id']]['mask'] > 1] = 0
                else:
                    used_mask_dic[shape['group_id']] = {}
                    used_mask_dic[shape['group_id']]['mask'] = mask
                    used_mask_dic[shape['group_id']]['label'] = shape['label']
            
            self.processor = InferenceCore(self.prop_model, rgb.to(device=self.device), len(used_mask_dic.keys()), top_k=20,
                mem_every=1000, include_last=True, device=self.device)
            self.tracked_object_text.setText('Tracked_object:{}'.format(list(used_mask_dic.keys())))
            mask = torch.stack([v['mask'] for _, v in used_mask_dic.items()],0).unsqueeze(1)
            mask = F.interpolate(mask, size=(self.scaled_h, self.scaled_w), mode='nearest')
            self.tracked_id_list = [k for k, _ in used_mask_dic.items()]
            self.tracked_label_list = [v['label'] for _, v in used_mask_dic.items()]
            self.addmemory(mask, self.current_img_index)
            #self.canvas.selectedShapes = []
            


    def addmemory(self, mask, frame_idx, is_temp=False):
        with torch.no_grad():
            mask, _ = pad_divide_by(mask, 16)
            #self.prob[:, frame_idx] = aggregate(mask, keep_bg=True)

            # KV pair for the interacting frame
            key_k, _, qf16, _, _ = self.processor.encode_key(frame_idx)
            key_v = self.processor.prop_net.encode_value(self.processor.images[:,frame_idx], qf16, mask.to(device=self.device))
            key_k = key_k.unsqueeze(2)
            self.processor.mem_bank.add_memory(key_k, key_v, is_temp)


    def clickAddKeyFrame(self):
        if self.processor == None or len(self.tracked_id_list) == 0:
            return
            
        def format_shape(s):
            data = s.other_data.copy()
            data.update(
                dict(
                    label=s.label.encode("utf-8") if PY2 else s.label,
                    points=[(p.x(), p.y()) for p in s.points],
                    group_id=s.group_id,
                    shape_type=s.shape_type,
                    flags=s.flags,
                )
            )
            return data
            
        self.memory_shapes = [format_shape(item.shape()) for item in self.labelList]
        used_mask_dic = {}
        for shape in self.memory_shapes:
            if shape['group_id'] not in self.tracked_id_list:
                continue
            mask = torch.Tensor(self.polygon2mask(shape['points'],(self.raw_h,self.raw_w)))
            if shape['group_id'] in used_mask_dic.keys():
                used_mask_dic[shape['group_id']]['mask'] = used_mask_dic[shape['group_id']]['mask'] + mask
                used_mask_dic[shape['group_id']]['mask'][used_mask_dic[shape['group_id']]['mask'] > 1] = 0
            else:
                used_mask_dic[shape['group_id']] = {}
                used_mask_dic[shape['group_id']]['mask'] = mask
                used_mask_dic[shape['group_id']]['label'] = shape['label']
        if len(used_mask_dic) != len(self.tracked_id_list):
            return
        
        mask = torch.stack([used_mask_dic[tid]['mask'] for tid in self.tracked_id_list],0).unsqueeze(1)
        mask = F.interpolate(mask, size=(self.scaled_h, self.scaled_w), mode='nearest')
        self.addmemory(mask, self.current_img_index)

        self.processor.mem_bank.temp_k = None
        self.processor.mem_bank.temp_v = None

    def clickPropagate(self):
        if self.processor == None or len(self.tracked_id_list) == 0:
            return
        with torch.no_grad():
            k16, qv16, qf16, qf8, qf4 = self.processor.encode_key(self.current_img_index)
            out_mask = self.processor.prop_net.segment_with_query(self.processor.mem_bank, qf8, qf4, k16, qv16)

            prev_value = self.processor.prop_net.encode_value(self.processor.images[:,self.current_img_index], qf16, out_mask)
            prev_key = k16.unsqueeze(2)
            #self.processor.mem_bank.add_memory(prev_key, prev_value,is_temp=True)
            self.processor.mem_bank.temp_k = prev_key.flatten(start_dim=2)
            self.processor.mem_bank.temp_v = prev_value.flatten(start_dim=2)
            prob = unpad(out_mask, self.processor.pad)
            prob = (prob > 0.5).cpu().numpy()[:,0].astype(np.uint8)
            out_masks = []
            for i in range(prob.shape[0]):
                out_masks.append(prob[i])
            del prob, k16, qv16, qf16, qf8, qf4, prev_value, prev_key
            torch.cuda.empty_cache()



        updated_id_list = []
        cur_shape = [item.shape() for item in self.labelList]
        add_shape = []
        vaild_contours_index = []
        for i in range(len(out_masks)):
            out_mask = out_masks[i]
            if out_mask.sum() == 0:
                continue
            out_mask = cv2.resize(out_mask, (self.raw_w, self.raw_h), interpolation=cv2.INTER_NEAREST)
            
            points_list = cv2.findContours(out_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            area_list = []
            for idx in range(len(points_list)):
                area = cv2.contourArea(points_list[idx])
                area_list.append(area)
            max_idx = np.argmax(np.array(area_list))
            #for points in points_list[max_idx:max_idx+1]:
            for iii, points in enumerate(points_list):
                if cv2.contourArea(points) < 100 and iii != max_idx:
                    continue
                pointsx = points[:,0,0]
                pointsy = points[:,0,1]

                label = self.tracked_label_list[i]
                shape_type = 'polygon'
                group_id = int(self.tracked_id_list[i])
                updated_id_list.append(group_id)
                shape = Shape(
                    label=label,
                    shape_type=shape_type,
                    group_id=group_id,
                )
                for point_index in range(pointsx.shape[0]):
                    shape.addPoint(QtCore.QPointF(pointsx[point_index], pointsy[point_index]))
                shape.close()
                add_shape.append(shape)

        add_shape_raw = [i for i in cur_shape if int(i.group_id) not in updated_id_list]
        self.labelList.clear()
        for shape in add_shape_raw + add_shape:
            self.addLabel(shape)

        self.canvas.loadShapes([item.shape() for item in self.labelList])
        self.actions.save.setEnabled(True)

    def clickClearTrackMemory(self):
        del self.processor, self.tracked_id_list, self.tracked_label_list
        self.processor = None
        self.tracked_id_list = None
        self.tracked_label_list = None
        self.tracked_object_text.setText("Tracked_object:")
        

    def clickAutoSeg(self):
        pass
    
    def getMaxId(self):
        max_id = -1
        for label in self.labelList:
            if label.shape().group_id != None:
                max_id = max(max_id, int(label.shape().group_id))
        return max_id
        
    def show_proposals(self, masks=None, flag=1):
        if flag != 1:
            img = cv2.imread(self.current_img)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for msk_idx in range(masks.shape[0]):
                tmp_mask = masks[msk_idx]
                tmp_vis = img.copy()
                tmp_vis[tmp_mask > 0] = 0.5 * tmp_vis[tmp_mask > 0] + 0.5 * np.array([30,30,220])
                tmp_vis = cv2.resize(tmp_vis,(int(0.17 * global_w),int(0.14 * global_h)))
                tmp_vis = tmp_vis.astype(np.uint8)
                pixmap = QPixmap.fromImage(QImage(tmp_vis, tmp_vis.shape[1], tmp_vis.shape[0], tmp_vis.shape[1] * 3 , QImage.Format_RGB888))
                #self.button_proposal_list[msk_idx].setPixmap(pixmap)
                self.button_proposal_list[msk_idx].setIcon(QIcon(pixmap))
                self.button_proposal_list[msk_idx].setIconSize(QSize(tmp_vis.shape[1], tmp_vis.shape[0]))
                self.button_proposal_list[msk_idx].setShortcut(str(msk_idx+1))
        else:
            for idx, button_proposal in enumerate(self.button_proposal_list):
                button_proposal.setText('proprosal{}'.format(idx))
                button_proposal.setIconSize(QSize(0,0))
                self.button_proposal_list[idx].setShortcut(str(idx+1))

    def transform_input(self, image, box=None, points=None):
        if self.keep_input_size == True:
            return image, box, points
        else:
            h,w = image.shape[:2]
            scale_ratio = self.max_size / max(h,w)
            image = cv2.resize(image, (int(w*scale_ratio), int(h*scale_ratio)))
            if box is not None:
                box = box * scale_ratio
            if points is not None:
                points = points * scale_ratio
            return image, box, points
    
    def transform_output(self, masks, size):
        if self.keep_input_size == True:
            return masks
        else:
            h,w = size
            N = masks.shape[0]
            new_masks = np.zeros((N,h,w), dtype=np.uint8)
            for idx in range(N):
                new_masks[idx] = cv2.resize(masks[idx], (w,h))
            return new_masks

    def clickManualSegBBox(self):
        Box = self.canvas.currentBox
        if self.predictor is None or self.current_img == '' or Box == None:
            return
        img = cv2.imread(self.current_img)[:,:,::-1]
        rh, rw = img.shape[:2]
        input_box = np.array([Box[0].x(), Box[0].y(), Box[1].x(), Box[1].y()])
        img, input_box, _ = self.transform_input(img, box=input_box)
        if self.image_encoded_flag == False:
            self.predictor.set_image(img)
            self.image_encoded_flag = True
        masks, iou_prediction, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True,
        )
        masks = self.transform_output(masks.astype(np.uint8), (rh,rw))

        target_idx = np.argmax(iou_prediction)
        self.show_proposals(masks, 0)
        self.sam_mask_proposal = []
        for msk_idx in range(masks.shape[0]):
            mask = masks[msk_idx].astype(np.uint8)

            points_list = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            shape_type = 'polygon'
            tmp_sam_mask = []
            for points in points_list:
                area = cv2.contourArea(points)
                if area < 100 and len(points_list) > 1:
                    continue
                pointsx = points[:,0,0]
                pointsy = points[:,0,1]

                shape = Shape(
                    label='Object',
                    shape_type=shape_type,
                    group_id=self.getMaxId() + 1,
                )
                for point_index in range(pointsx.shape[0]):
                    shape.addPoint(QtCore.QPointF(pointsx[point_index], pointsy[point_index]))
                shape.close()
                #self.addLabel(shape)
                tmp_sam_mask.append(shape)
            if msk_idx == target_idx:
                self.sam_mask = tmp_sam_mask
            self.sam_mask_proposal.append(tmp_sam_mask)


    def clickManualSegBox(self):
        ClickPos = self.canvas.currentPos
        ClickNeg = self.canvas.currentNeg
        if self.predictor is None or self.current_img == '' or (ClickPos == None and ClickNeg == None):
            return
        img = cv2.imread(self.current_img)[:,:,::-1]
        rh, rw = img.shape[:2]

        input_clicks = []
        input_types = []
        if ClickPos != None:
            for pos in ClickPos:
                input_clicks.append([int(pos.x()), int(pos.y())])
                input_types.append(1)

        if ClickNeg != None:
            for neg in ClickNeg:
                input_clicks.append([int(neg.x()), int(neg.y())])
                input_types.append(0)
        if len(input_clicks) == 0:
            input_clicks = None
            input_types = None
        else:
            input_clicks = np.array(input_clicks)
            input_types = np.array(input_types)

        img, _, input_clicks = self.transform_input(img, points=input_clicks)

        if self.image_encoded_flag == False:
            self.predictor.set_image(img)
            self.image_encoded_flag = True
        masks, iou_prediction, _ = self.predictor.predict(
            point_coords=input_clicks,
            point_labels=input_types,
            multimask_output=True,
        )
        masks = self.transform_output(masks.astype(np.uint8), (rh,rw))
        
        target_idx = np.argmax(iou_prediction)
        self.show_proposals(masks,0)
        self.sam_mask_proposal = []
        
        for msk_idx in range(masks.shape[0]):
            mask = masks[msk_idx].astype(np.uint8)
            
            points_list = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            shape_type = 'polygon'
            tmp_sam_mask = []
            for points in points_list:
                area = cv2.contourArea(points)
                if area < 100 and len(points_list) > 1:
                    continue
                pointsx = points[:,0,0]
                pointsy = points[:,0,1]

                shape = Shape(
                    label='Object',
                    shape_type=shape_type,
                    group_id=self.getMaxId() + 1,
                )
                for point_index in range(pointsx.shape[0]):
                    shape.addPoint(QtCore.QPointF(pointsx[point_index], pointsy[point_index]))
                shape.close()
                #self.addLabel(shape)
                tmp_sam_mask.append(shape)
            if msk_idx == target_idx:
                self.sam_mask = tmp_sam_mask
            self.sam_mask_proposal.append(tmp_sam_mask)
            
    def addSamMask(self):
        if len(self.sam_mask) > 0:
            label = 'Object'
            group_id = self.getMaxId() + 1
            if self.class_on_flag:
                xx = self.labelDialog.popUp(
                    text=label,
                    flags={},
                    group_id=group_id,
                )
                if len(xx) == 4:
                    label, _, group_id,_ = xx
                else:
                    label, _, group_id = xx
            if label == None:
                label = 'Object'
            if type(group_id) != int:
                group_id=self.getMaxId() + 1
            for sam_mask in self.sam_mask:
                sam_mask.label = label
                sam_mask.group_id = group_id
                self.addLabel(sam_mask)
        self.canvas.currentBox = None
        self.canvas.currentPos = None
        self.canvas.currentNeg = None
        self.sam_mask = []
        self.sam_mask_proposal = []
        self.show_proposals()
        self.canvas.loadShapes([item.shape() for item in self.labelList])
        self.actions.save.setEnabled(True)
        self.actions.editMode.setEnabled(True)



    def cleanPrompt(self):
        self.canvas.currentBox = None
        self.canvas.currentPos = None
        self.canvas.currentNeg = None
        self.canvas.current = None
        self.sam_mask = []
        self.sam_mask_proposal = []
        self.show_proposals()
        self.canvas.setHiding()
        self.canvas.update()
        self.actions.editMode.setEnabled(True)



    def zoomRequest(self, delta, pos):
        canvas_width_old = self.canvas.width()
        units = 1.1
        if delta < 0:
            units = 0.9
        self.addZoom(units)

        canvas_width_new = self.canvas.width()
        if canvas_width_old != canvas_width_new:
            canvas_scale_factor = canvas_width_new / canvas_width_old

            x_shift = round(pos.x() * canvas_scale_factor) - pos.x()
            y_shift = round(pos.y() * canvas_scale_factor) - pos.y()

            self.setScroll(
                Qt.Horizontal,
                self.scrollBars[Qt.Horizontal].value() + x_shift,
            )
            self.setScroll(
                Qt.Vertical,
                self.scrollBars[Qt.Vertical].value() + y_shift,
            )

    def scrollRequest(self, delta, orientation):
        units = -delta * 0.1  # natural scroll
        bar = self.scrollBars[orientation]
        value = bar.value() + bar.singleStep() * units
        self.setScroll(orientation, value)

    def newShape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        items = self.uniqLabelList.selectedItems()
        text = None
        if items:
            text = items[0].data(Qt.UserRole)
        flags = {}
        group_id = None
        if not text:
            previous_text = self.labelDialog.edit.text()
            xx = self.labelDialog.popUp(text)
            if len(xx) == 4:
                text, flags, group_id, _ = xx
            else:
                text, flags, group_id = xx
            if not text:
                self.labelDialog.edit.setText(previous_text)

        if text and not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            text = ""
        if text:
            self.labelList.clearSelection()
            shape = self.canvas.setLastLabel(text, flags)
            shape.group_id = group_id
            self.addLabel(shape)
            self.actions.editMode.setEnabled(True)
            self.setDirty()
        else:
            self.canvas.undoLastLine()
            self.canvas.shapesBackups.pop()

    def setDirty(self):
        # Even if we autosave the file, we keep the ability to undo

        # if self._config["auto_save"] or self.actions.saveAuto.isChecked():
        #     label_file = osp.splitext(self.imagePath)[0] + ".json"
        #     if self.output_dir:
        #         label_file_without_path = osp.basename(label_file)
        #         label_file = osp.join(self.output_dir, label_file_without_path)
        #     self.saveLabels(label_file)
        #     return
        # self.dirty = True
        self.actions.save.setEnabled(True)
        # title = __appname__
        # if self.filename is not None:
        #     title = "{} - {}*".format(title, self.filename)
        # self.setWindowTitle(title)

    # React to canvas signals.
    def shapeSelectionChanged(self, selected_shapes):
        self._noSelectionSlot = True
        for shape in self.canvas.selectedShapes:
            shape.selected = False
        self.labelList.clearSelection()
        self.canvas.selectedShapes = selected_shapes
        for shape in self.canvas.selectedShapes:
            shape.selected = True
            item = self.labelList.findItemByShape(shape)
            self.labelList.selectItem(item)
            self.labelList.scrollToItem(item)
        self._noSelectionSlot = False
        n_selected = len(selected_shapes)
        self.actions.delete.setEnabled(n_selected)

    def toggleDrawingSensitive(self, drawing=True):
        """Toggle drawing sensitive.

        In the middle of drawing, toggling between modes should be disabled.
        """
        self.actions.editMode.setEnabled(not drawing)
        # self.actions.undoLastPoint.setEnabled(drawing)
        # self.actions.undo.setEnabled(not drawing)
        # self.actions.delete.setEnabled(not drawing)
    def setScroll(self, orientation, value):
        self.scrollBars[orientation].setValue(int(value))
        self.scroll_values[orientation][self.current_img] = value

    def toolbar(self, title, actions=None):
        toolbar = self.addToolBar("%sToolBar" % title)
        # toolbar.setOrientation(Qt.Vertical)
        if actions:
            utils.addActions(toolbar, actions)
        return toolbar

    def setEditMode(self):
        self.toggleDrawMode(True)

    def toggleDrawMode(self, edit=True, createMode="polygon"):
        self.canvas.setEditing(edit)
        self.canvas.createMode = createMode
        if edit:
            self.actions.createMode.setEnabled(True)
            self.actions.createPointMode.setEnabled(True)
            self.actions.createRectangleMode.setEnabled(True)

        else:
            if createMode == "polygon":
                self.actions.createPointMode.setEnabled(True)
                self.actions.createMode.setEnabled(False)
                self.actions.createRectangleMode.setEnabled(True)

            elif createMode == "point":
                self.actions.createMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(False)
                self.actions.createRectangleMode.setEnabled(True)
            elif createMode == "rectangle":
                self.actions.createMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(False)
            else:
                raise ValueError("Unsupported createMode: %s" % createMode)
        self.actions.editMode.setEnabled(not edit)

    def validateLabel(self, label):
        return True

    def labelSelectionChanged(self):
        if self._noSelectionSlot:
            return
        if self.canvas.editing():
            selected_shapes = []
            for item in self.labelList.selectedItems():
                selected_shapes.append(item.shape())
            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)
            else:
                self.canvas.deSelectShape()

    def iou(self, target_mask, mask_list):
        target_mask = target_mask.reshape(1,-1)
        mask_list = mask_list.reshape(mask_list.shape[0], -1)
        i = (target_mask * mask_list)
        u = target_mask + mask_list - i
        return i.sum(1)/u.sum(1)


    def polygon2mask(self,polygon, size):
        mask = np.zeros((size)) # h,w
        contours = np.array(polygon)
        mask = cv2.fillPoly(mask, [contours.astype(np.int32)],1)
        return mask.astype(np.uint8)

    def mask2polygon(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = np.array(contours[0])
        return contours

    def editLabel(self, item=None):
        if item and not isinstance(item, LabelListWidgetItem):
            raise TypeError("item must be LabelListWidgetItem type")

        if not self.canvas.editing():
            return
        if not item:
            item = self.currentItem()
        if item is None:
            return
        shape = item.shape()
        if shape is None:
            return
        xx = self.labelDialog.popUp(
            text=shape.label,
            flags=shape.flags,
            group_id=shape.group_id,
        )
        if len(xx) == 4:
            text, flags, group_id,_ = xx
        else:
            text, flags, group_id = xx
        if text is None:
            return
        if not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            return
        shape.label = text
        shape.flags = flags
        shape.group_id = group_id

        self._update_shape_color(shape)
        if shape.group_id is None:
            item.setText(
                '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                    html.escape(shape.label), *shape.fill_color.getRgb()[:3]
                )
            )
        else:
            item.setText("({}) {}".format(shape.group_id, shape.label))
        self.setDirty()
        if self.uniqLabelList.findItemByLabel(shape.label) is None:
            item = self.uniqLabelList.createItemFromLabel(shape.label)
            self.uniqLabelList.addItem(item)
            # rgb = self._get_rgb_by_label(shape.label)
            rgb = self._get_rgb_by_label(shape.group_id)
            self.uniqLabelList.setItemLabel(item, shape.label, rgb)

    def labelItemChanged(self, item):
        shape = item.shape()
        self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    def labelOrderChanged(self):
        self.setDirty()
        self.canvas.loadShapes([item.shape() for item in self.labelList])

    def addLabel(self, shape):
        if shape.group_id is None:
            text = shape.label
        else:
            text = "({}) {}".format(shape.group_id, shape.label)
        label_list_item = LabelListWidgetItem(text, shape)
        self.labelList.addItem(label_list_item)
        if self.uniqLabelList.findItemByLabel(shape.label) is None:
            item = self.uniqLabelList.createItemFromLabel(shape.label)
            self.uniqLabelList.addItem(item)
            # rgb = self._get_rgb_by_label(shape.label)
            rgb = self._get_rgb_by_label(shape.group_id)
            self.uniqLabelList.setItemLabel(item, shape.label, rgb)
        self.labelDialog.addLabelHistory(shape.label)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)

        self._update_shape_color(shape)
        label_list_item.setText(
            '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                html.escape(text), *shape.fill_color.getRgb()[:3]
            )
        )
    def _get_rgb_by_label(self, label):
        label = str(label)
        item = self.uniqLabelList.findItemByLabel(label)
        if item is None:
            item = self.uniqLabelList.createItemFromLabel(label)
            self.uniqLabelList.addItem(item)
            rgb = self._get_rgb_by_label(label)
            self.uniqLabelList.setItemLabel(item, label, rgb)
        label_id = self.uniqLabelList.indexFromItem(item).row() + 1
        label_id += 0
        return LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)]

    def togglePolygons(self, value):
        for item in self.labelList:
            item.setCheckState(Qt.Checked if value else Qt.Unchecked)

    def _update_shape_color(self, shape):
        # r, g, b = self._get_rgb_by_label(shape.label)
        r, g, b = self._get_rgb_by_label(shape.group_id)
        shape.line_color = QtGui.QColor(r, g, b)
        shape.vertex_fill_color = QtGui.QColor(r, g, b)
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)
        shape.fill_color = QtGui.QColor(r, g, b, 128)
        shape.select_line_color = QtGui.QColor(255, 255, 255)
        shape.select_fill_color = QtGui.QColor(r, g, b, 155)

    def undoShapeEdit(self):
        self.canvas.restoreShape()
        self.labelList.clear()
        self.loadShapes(self.canvas.shapes)

    def loadShapes(self, shapes, replace=True):
        self._noSelectionSlot = True
        for shape in shapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self._noSelectionSlot = False
        self.canvas.loadShapes(shapes, replace=replace)


    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def copyShape(self):
        self.canvas.endMove(copy=True)
        for shape in self.canvas.selectedShapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self.setDirty()
    def deleteSelectedShape(self):
        self.remLabels(self.canvas.deleteSelected())
        self.setDirty()
        if self.noShapes():
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)
    def duplicateSelectedShape(self):
        added_shapes = self.canvas.duplicateSelectedShapes()
        self.labelList.clearSelection()
        for shape in added_shapes:
            self.addLabel(shape)
        self.setDirty()

    def reducePoint(self):
        def format_shape(s):
            data = s.other_data.copy()
            data.update(
                dict(
                    label=s.label.encode("utf-8") if PY2 else s.label,
                    points=[(p.x(), p.y()) for p in s.points],
                    group_id=s.group_id,
                    shape_type=s.shape_type,
                    flags=s.flags,
                )
            )
            return data
        shapes = self.current_img
        shapes = [format_shape(item.shape()) for item in self.labelList.selectedItems()]
        rm_shapes = [item.shape() for item in self.labelList.selectedItems()]
        self.remLabels(rm_shapes)
        for shape in shapes:
            points = shape['points']
            min_dis = self.get_min_dis(points)
            points_new = [points[0]]
            for i in range(1,len(points)):
                d = math.sqrt((points[i][0] - points_new[-1][0]) ** 2 + (points[i][1] - points_new[-1][1]) ** 2)
                if d > (min_dis * 1.5):
                    points_new.append(points[i])
            shape['points'] = points_new
        #self.labelList.clear()
        for tmp_shape in shapes:
            shape = Shape(
                label=tmp_shape['label'],
                shape_type=tmp_shape['shape_type'],
                group_id=tmp_shape['group_id'],
            )
            for point_index in range(len(tmp_shape['points'])):
                shape.addPoint(QtCore.QPointF(tmp_shape['points'][point_index][0], tmp_shape['points'][point_index][1]))
            shape.close()
            self.addLabel(shape)
            tmp_item = self.labelList.findItemByShape(shape)
            self.labelList.selectItem(tmp_item)
            self.labelList.scrollToItem(tmp_item)
        self.canvas.loadShapes([item.shape() for item in self.labelList])
        self.actions.save.setEnabled(True)

    def get_min_dis(self, points):
        min_dis = 10000
        if len(points) >= 2:
            points_new = [points[0]]
            for i in range(1,len(points)):
                d = math.sqrt((points[i][0] - points_new[-1][0]) ** 2 + (points[i][1] - points_new[-1][1]) ** 2)
                min_dis = min(min_dis, d)
                points_new.append(points[i])
        return min_dis



    def pasteSelectedShape(self):
        self.loadShapes(self._copied_shapes, replace=False)
        self.setDirty()

    def copySelectedShape(self):
        self._copied_shapes = [s.copy() for s in self.canvas.selectedShapes]
        self.actions.paste.setEnabled(len(self._copied_shapes) > 0)

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def remLabels(self, shapes):
        for shape in shapes:
            item = self.labelList.findItemByShape(shape)
            self.labelList.removeItem(item)


    def noShapes(self):
        return not len(self.labelList)

    def addZoom(self, increment=1.1):
        zoom_value = self.zoomWidget.value() * increment
        if increment > 1:
            zoom_value = math.ceil(zoom_value)
        else:
            zoom_value = math.floor(zoom_value)
        self.setZoom(zoom_value)

    def setZoom(self, value):
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)
        self.zoom_values[self.current_img] = (self.zoomMode, value)

    def paintCanvas(self):
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()


def get_parser():
    parser = argparse.ArgumentParser(description="pixel annotator by GroundedSAM")
    parser.add_argument(
        "--app_resolution",
        default='1000,1600',
    )
    parser.add_argument(
        "--model_type",
        default='vit_b',
    )
    parser.add_argument(
        "--keep_input_size",
        type=bool,
        default=True,
    )   
    parser.add_argument(
        "--max_size",
        default=720,
    )
    parser.add_argument(
        "--max_size_STCN",
        default=600,
    )
    return parser

if __name__ == '__main__':
    parser = get_parser()
    global_h, global_w = [int(i) for i in parser.parse_args().app_resolution.split(',')]
    model_type = parser.parse_args().model_type
    keep_input_size = parser.parse_args().keep_input_size
    max_size = parser.parse_args().max_size
    max_size_STCN = parser.parse_args().max_size_STCN
    app = QApplication(sys.argv)
    main = MainWindow(global_h=global_h, global_w=global_w, model_type=model_type, keep_input_size=keep_input_size, max_size=max_size, max_size_STCN=max_size_STCN)
    main.show()
    sys.exit(app.exec_())
