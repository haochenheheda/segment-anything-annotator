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

from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QApplication, QPushButton, QLabel, QFileDialog, QProgressBar, QComboBox, QScrollArea, QDockWidget, QMessageBox
from PyQt5.QtGui import QPixmap
from qtpy.QtCore import Qt
from qtpy import QtCore
from qtpy import QtGui, QtWidgets
from canvas import Canvas
import utils

from labelme.widgets import ToolBar, UniqueLabelQListWidget, LabelDialog, LabelListWidget, LabelListWidgetItem, ZoomWidget
from labelme import PY2
from labelme.label_file import LabelFile
from labelme.label_file import LabelFileError

from shape import Shape

from PIL import Image

from collections import namedtuple
Click = namedtuple('Click', ['is_positive', 'coords'])

from mask_predictor import SegAutoMaskPredictor, SegManualMaskPredictor





LABEL_COLORMAP = imgviz.label_colormap()

class MainWindow(QMainWindow):

    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = 0, 1, 2

    def __init__(self, parent=None, global_w=1000, global_h=1800):
        super(MainWindow, self).__init__(parent)
        self.resize(global_w, global_h)
        self.setWindowTitle('标注平台')
        self.canvas = Canvas(self,
            epsilon=10.0,
            double_click='close',
            num_backups=10,
        )

        self._noSelectionSlot = False
        self.current_output_dir = 'output'
        self.current_output_filename = ''
        self.canvas.zoomRequest.connect(self.zoomRequest)

        self.memory_shapes = []

        self.SegAuto = None
        self.SegManual = None

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

        self.button_next = QPushButton('Next Image', self)
        self.button_next.clicked.connect(self.clickButtonNext)
        self.button_last = QPushButton('Last Image', self)
        self.button_last.clicked.connect(self.clickButtonLast)

        self.img_progress_bar = QProgressBar(self)
        self.img_progress_bar.setMinimum(0)
        self.img_progress_bar.setMaximum(1)
        self.img_progress_bar.setValue(0)


        self.scrollArea.move(int(0.02 * global_w), int(0.1 * global_h))
        self.scrollArea.resize(int(0.75 * global_w), int(0.8 * global_h))
        self.shape_dock.move(int(0.79 * global_w), (0.1 * global_h))
        self.shape_dock.resize(int(0.2 * global_w), int(0.8 * global_h))
        self.button_next.move(int(0.6 * global_w), int(0.9 * global_h))
        self.button_next.resize(int(0.1 * global_w),int(0.06 * global_h))
        self.button_last.move(int(0.1 * global_w), int(0.9 * global_h))
        self.button_last.resize(int(0.1 * global_w),int(0.06 * global_h))
        self.img_progress_bar.move(int(0.2 * global_w), int(0.9 * global_h))
        self.img_progress_bar.resize(int(0.4 * global_w),int(0.06 * global_h))

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
            self.tr("Image Directory"),
            lambda: self.clickFileChoose(),
            'None',
            "objects",
            self.tr("Image Directory"),
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
        AutoSeg = action(
            self.tr("AutoSeg"),
            lambda: self.clickAutoSeg(),
            'None',
            "objects",
            self.tr("AutoSeg"),
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
            self.tr("Create Polygons"),
            lambda: self.toggleDrawMode(False, createMode="polygon"),
            'Ctrl+W',
            "objects",
            self.tr("Start drawing polygons"),
            enabled=True,
        )
        createPointMode = action(
            self.tr("Create Point"),
            lambda: self.toggleDrawMode(False, createMode="point"),
            'None',
            "objects",
            self.tr("Start drawing points"),
            enabled=False,
        )
        editMode = action(
            self.tr("Edit Polygons"),
            self.setEditMode,
            'Ctrl+E',
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

        undoLastPoint = action(
            self.tr("Undo last point"),
            self.canvas.undoLastPoint,
            'U',
            "undo",
            self.tr("Undo last drawn point"),
            enabled=False,
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

        undo = action(
            self.tr("Undo"),
            self.undoShapeEdit,
            'Ctrl+U',
            "undo",
            self.tr("Undo last add and edit of shape"),
            enabled=False,
        )

        save = action(
            self.tr("&Save"),
            self.saveFile,
            'S',
            "save",
            self.tr("Save labels to file"),
            enabled=False,
        )

        delete = action(
            self.tr("Delete Polygons"),
            self.deleteSelectedShape,
            'Alt+2',
            "cancel",
            self.tr("Delete the selected polygons"),
            enabled=False,
        )
        duplicate = action(
            self.tr("Duplicate Polygons"),
            self.duplicateSelectedShape,
            'None',
            "copy",
            self.tr("Create a duplicate of the selected polygons"),
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
        edit = action(
            self.tr("&Edit Label"),
            self.editLabel,
            'None',
            "edit",
            self.tr("Modify the label of the selected polygon"),
            enabled=False,
        )
        

        self.actions = utils.struct(
            categoryFile=categoryFile,
            imageDirectory=imageDirectory,
            saveDirectory=saveDirectory,
            loadSAM=LoadSAM,
            autoSeg=AutoSeg,
            createMode=createMode,
            createPointMode=createPointMode,
            editMode=editMode,
            undoLastPoint=undoLastPoint,
            undo=undo,
            delete=delete,
            edit=edit,
            duplicate=duplicate,
            reduce_point=reduce_point,
            save=save,
            onShapesPresent=(saveAs, hideAll, showAll),
            menu=(
                createMode,
                editMode,
                undoLastPoint,
                undo,
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
        self.toolbar.addAction(LoadSAM)
        self.toolbar.addAction(AutoSeg)
        self.toolbar.addAction(createMode)
        self.toolbar.addAction(editMode)
        self.toolbar.addAction(undoLastPoint)
        self.toolbar.addAction(undo)
        self.toolbar.addAction(delete)
        self.toolbar.addAction(edit)
        self.toolbar.addAction(duplicate)
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
        if filename and self.saveLabels(filename):
            self.setClean()

    def saveLabels(self, filename):
        lf = LabelFile()

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

        shapes = [format_shape(item.shape()) for item in self.labelList]
        with open(filename, 'w') as f:
            json.dump(shapes, f)
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
        basename = self.current_img.split('\\')[-1][:-4]
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
        for shape in data:
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
        if self.current_img_index < self.img_len - 1:
            self.current_img_index += 1
            self.current_img = self.img_list[self.current_img_index]
            self.loadImg()



    def clickButtonLast(self):
        if self.current_img_index > 0:
            self.current_img_index -= 1
            self.current_img = self.img_list[self.current_img_index]
            self.loadImg()

    def loadImg(self):
        pixmap = QPixmap(self.current_img)
        self.canvas.loadPixmap(pixmap)
        self.img_progress_bar.setValue(self.current_img_index)

        img_name = self.current_img.split('\\')[-1][:-4]
        self.current_output_filename = osp.join(self.current_output_dir, img_name + '.json')
        self.labelList.clear()
        if os.path.isfile(self.current_output_filename):
            self.loadAnno(self.current_output_filename)            


    def clickFileChoose(self):
        directory = QFileDialog.getExistingDirectory(self, 'choose target fold','.')
        if directory == '':
            return
        #self.img_list = glob.glob(directory + '/*.{jpg,png,JPG,PNG}')
        self.img_list = glob.glob(directory + '/*.png')
        self.img_list.sort()
        self.img_len = len(self.img_list)
        if self.img_len == 0:
            return
        self.current_img_index = 0
        self.current_img = self.img_list[self.current_img_index]
        self.img_progress_bar.setMinimum(0)
        self.img_progress_bar.setMaximum(self.img_len-1)
        self.loadImg()

    def clickSaveChoose(self):
        directory = QFileDialog.getExistingDirectory(self, 'choose target fold','.')
        if directory == '':
            return
        else:
            self.output_dir = directory
            return directory

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
        self.SegAuto = SegAutoMaskPredictor()
        self.SegAuto.load_model('vit_b')
        self.SegManual = SegManualMaskPredictor()
        self.SegManual.load_model('vit_b')
        self.actions.loadSAM.setEnabled(False)
        self.actions.autoSeg.setEnabled(True)
    
    def clickAutoSeg(self):
        if self.SegAuto is None or self.current_img == '':
            return 
        _, masks = self.SegAuto.predict(self.current_img, 
                points_per_side=16, 
                points_per_batch=64,
                min_area=0)
        print(len(masks))
        return
        


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
            text, flags, group_id = self.labelDialog.popUp(text)
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
            self.actions.undoLastPoint.setEnabled(False)
            self.actions.undo.setEnabled(True)
            self.setDirty()
        else:
            self.canvas.undoLastLine()
            self.canvas.shapesBackups.pop()

    def setDirty(self):
        # Even if we autosave the file, we keep the ability to undo
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)

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
        self.actions.duplicate.setEnabled(n_selected)
        self.actions.edit.setEnabled(n_selected == 1)

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
        else:
            if createMode == "polygon":
                self.actions.createPointMode.setEnabled(True)
                self.actions.createMode.setEnabled(False)

            elif createMode == "point":
                self.actions.createMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(False)
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
        text, flags, group_id = self.labelDialog.popUp(
            text=shape.label,
            flags=shape.flags,
            group_id=shape.group_id,
        )
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
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)

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
        #yes, no = QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
        #msg = self.tr(
        #    "You are about to permanently delete {} polygons, "
        #    "proceed anyway?"
        #).format(len(self.canvas.selectedShapes))
        #if yes == QtWidgets.QMessageBox.warning(
        #    self, self.tr("Attention"), msg, yes | no, yes
        #):
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
        default=(800,1400),
    )
    return parser

if __name__ == '__main__':
    parser = get_parser()
    global_h, global_w = parser.parse_args().app_resolution
    app = QApplication(sys.argv)
    main = MainWindow(global_h=global_h, global_w=global_w)
    main.show()
    sys.exit(app.exec_())