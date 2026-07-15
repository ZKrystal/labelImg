#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys as _sys
import os as _os
import shutil as _shutil
import tempfile as _tempfile

_user_site = _os.path.normpath(_os.path.expanduser(
    r"~\AppData\Roaming\Python\Python312\site-packages"))
if _user_site in list(_sys.path):
    # Ensure typing_extensions is available (only in user site-packages)
    try:
        import typing_extensions
    except ImportError:
        _te_src = _os.path.join(_user_site, "typing_extensions.py")
        if _os.path.exists(_te_src):
            _te_dir = _tempfile.mkdtemp()
            _shutil.copy2(_te_src, _os.path.join(_te_dir, "typing_extensions.py"))
            _sys.path.insert(0, _te_dir)
    # Remove broken cv2 from user site-packages access
    _sys.path = [p for p in _sys.path
                 if _os.path.normpath(p) != _user_site]
del _sys, _os, _shutil, _tempfile, _user_site

import onnxruntime as ort
import libs
from libs.labelFile import LabelFile, LabelFileError, LabelFileFormat
from libs.yolo_io import YOLOWriter
import codecs
import math
import traceback


# import torch
from PyQt5 import QtWidgets
import os
import platform
import glob
import sys
import logging
from logging.handlers import RotatingFileHandler

import shutil
import webbrowser as wb
import json
from functools import partial

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    # needed for py3+qt4
    # Ref:
    # http://pyqt.sourceforge.net/Docs/PyQt4/incompatible_apis.html
    # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string
    if sys.version_info.major >= 3:
        import sip

        sip.setapi('QVariant', 2)
    # from PyQt4.QtGui import *
    # from PyQt4.QtCore import *

# from libs.combobox import ComboBox
from libs.constants import *
from libs.utils import *
from libs.settings import Settings
from libs.shape import Shape, DEFAULT_LINE_COLOR, DEFAULT_FILL_COLOR
from libs.stringBundle import StringBundle
from libs.canvas import Canvas
from libs.sam_client import SamClient
from libs.zoomWidget import ZoomWidget
from libs.labelDialog import LabelDialog
from libs.colorDialog import ColorDialog
from libs.colorDialog import ColorDialog
from libs.resources import *
from libs.toolBar import ToolBar
from libs.pascal_voc_io import PascalVocReader
from libs.pascal_voc_io import XML_EXT
from libs.yolo_io import YoloReader
from libs.yolo_io import TXT_EXT, ENCODE_METHOD
from libs.create_ml_io import CreateMLReader
from libs.create_ml_io import JSON_EXT
from libs.ustr import ustr
from libs.hashableQListWidgetItem import HashableQListWidgetItem

from PIL import Image, ImageDraw, ImageFont
import argparse
from typing import List, Tuple
import time
import datetime

import cv2
import numpy as np


def _imread_unicode(path):
    # Read image file supporting Unicode (Chinese) paths on Windows.
    try:
        with open(path, "rb") as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


_LOG_FILE_PATH = None


def _setup_app_logging(app_name):
    """
    Create a persistent log file for packaged/GUI runs (no console).
    Writes to a user-writable AppData location with rotation.
    """
    global _LOG_FILE_PATH
    if _LOG_FILE_PATH:
        return _LOG_FILE_PATH

    try:
        log_dir = QStandardPaths.writableLocation(
            QStandardPaths.AppDataLocation)
    except Exception:
        log_dir = ""

    if not log_dir:
        log_dir = os.path.join(os.path.expanduser("~"), f".{app_name}")

    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        log_dir = os.path.abspath(".")

    log_path = os.path.join(log_dir, f"{app_name}.log")

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    try:
        already = False
        for h in list(root.handlers):
            try:
                if getattr(
                        h, "baseFilename", None) and os.path.abspath(
                    h.baseFilename) == os.path.abspath(log_path):
                    already = True
                    break
            except Exception:
                continue
        if not already:
            handler = RotatingFileHandler(
                log_path,
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            root.addHandler(handler)
    except Exception:
        pass

    _LOG_FILE_PATH = log_path

    try:
        logging.getLogger(__name__).info("==== %s started ====", app_name)
        logging.getLogger(__name__).info("Log file: %s", log_path)
    except Exception:
        pass

    def _qt_message_handler(mode, context, message):
        try:
            text = str(message)
        except Exception:
            text = repr(message)
        try:
            if mode == QtDebugMsg:
                logging.getLogger("qt").debug(text)
            elif mode == QtInfoMsg:
                logging.getLogger("qt").info(text)
            elif mode == QtWarningMsg:
                logging.getLogger("qt").warning(text)
            elif mode == QtCriticalMsg:
                logging.getLogger("qt").error(text)
            elif mode == QtFatalMsg:
                logging.getLogger("qt").critical(text)
            else:
                logging.getLogger("qt").info(text)
        except Exception:
            pass

    try:
        qInstallMessageHandler(_qt_message_handler)
    except Exception:
        pass

    def _excepthook(exctype, value, tb):
        try:
            logging.getLogger("crash").exception(
                "Uncaught exception", exc_info=(
                    exctype, value, tb))
        except Exception:
            pass
        try:
            sys.__excepthook__(exctype, value, tb)
        except Exception:
            pass

    try:
        sys.excepthook = _excepthook
    except Exception:
        pass

    class _TeeToLogger:
        def __init__(self, original_stream, logger_name, level=logging.INFO):
            self._stream = original_stream
            self._logger = logging.getLogger(logger_name)
            self._level = level
            self._buf = ""

        def write(self, msg):
            try:
                if self._stream:
                    self._stream.write(msg)
            except Exception:
                pass

            try:
                self._buf += str(msg)
                while "\n" in self._buf:
                    line, self._buf = self._buf.split("\n", 1)
                    line = line.rstrip("\r")
                    if line.strip():
                        self._logger.log(self._level, line)
            except Exception:
                pass

        def flush(self):
            try:
                if self._stream:
                    self._stream.flush()
            except Exception:
                pass

        @property
        def encoding(self):
            try:
                return getattr(self._stream, "encoding", "utf-8")
            except Exception:
                return "utf-8"

    try:
        if getattr(sys, "stdout", None) is not None:
            sys.stdout = _TeeToLogger(sys.stdout, "stdout", logging.INFO)
        if getattr(sys, "stderr", None) is not None:
            sys.stderr = _TeeToLogger(sys.stderr, "stderr", logging.ERROR)
    except Exception:
        pass

    return _LOG_FILE_PATH


class ZoomableImageView(QScrollArea):
    """可缩放和拖动的图片查看器"""

    def __init__(self, parent=None):
        super(ZoomableImageView, self).__init__(parent)
        self.setWidgetResizable(True)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "QScrollArea { background-color: #f0f0f0; border: 1px solid #ccc; }")

        # 创建内部标签用于显示图片
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(False)  # 不自动缩放，由我们控制
        self.setWidget(self.image_label)

        # 缩放和拖动相关变量
        self.scale_factor = 1.0
        self.min_scale = 0.1
        self.max_scale = 10.0
        self.pan_start_pos = QPoint()
        self.panning = False

        # 原始图片
        self.original_pixmap = None

    def setPixmap(self, pixmap):
        """设置图片"""
        if pixmap is None or pixmap.isNull():
            self.original_pixmap = None
            self.image_label.clear()
            self.scale_factor = 1.0
            return
        self.original_pixmap = pixmap
        # 初始时自动适应窗口大小
        self._fitToWindow()

    def _fitToWindow(self):
        """适应窗口大小（初始显示时）"""
        if self.original_pixmap is None or self.original_pixmap.isNull():
            return
        # 计算适应窗口的缩放比例
        view_size = self.size()
        pixmap_size = self.original_pixmap.size()

        if view_size.width() > 0 and view_size.height() > 0:
            scale_x = (view_size.width() - 20) / pixmap_size.width()
            scale_y = (view_size.height() - 20) / pixmap_size.height()
            # 允许放大，但初始时适应窗口（取较小值）
            self.scale_factor = min(scale_x, scale_y)
        else:
            self.scale_factor = 1.0

        self._updatePixmap()

    def _updatePixmap(self):
        """更新显示的图片（根据当前缩放比例）"""
        if self.original_pixmap is None or self.original_pixmap.isNull():
            self.image_label.clear()
            return

        scaled_size = self.original_pixmap.size() * self.scale_factor
        scaled_pixmap = self.original_pixmap.scaled(
            scaled_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.resize(scaled_pixmap.size())

    def wheelEvent(self, event):
        """鼠标滚轮缩放"""
        if self.original_pixmap is None or self.original_pixmap.isNull():
            super(ZoomableImageView, self).wheelEvent(event)
            return

        if event.modifiers() & Qt.ControlModifier:
            # Ctrl + 滚轮：缩放
            delta = event.angleDelta().y()
            old_scale = self.scale_factor

            if delta > 0:
                self.scale_factor *= 1.15
            else:
                self.scale_factor *= 0.85

            self.scale_factor = max(
                self.min_scale, min(
                    self.max_scale, self.scale_factor))

            # 获取鼠标在视图中的位置
            mouse_pos = event.pos()
            scroll_bar_h = self.horizontalScrollBar()
            scroll_bar_v = self.verticalScrollBar()

            # 计算鼠标在图片上的相对位置（考虑当前滚动位置）
            old_scroll_h = scroll_bar_h.value()
            old_scroll_v = scroll_bar_v.value()

            # 鼠标在图片上的位置 = 鼠标在视图中的位置 + 滚动条的值
            old_image_x = mouse_pos.x() + old_scroll_h
            old_image_y = mouse_pos.y() + old_scroll_v

            # 更新图片
            self._updatePixmap()

            # 计算新的滚动位置，使鼠标指向的图片位置保持不变
            new_image_x = old_image_x * (self.scale_factor / old_scale)
            new_image_y = old_image_y * (self.scale_factor / old_scale)

            new_scroll_h = int(new_image_x - mouse_pos.x())
            new_scroll_v = int(new_image_y - mouse_pos.y())

            scroll_bar_h.setValue(new_scroll_h)
            scroll_bar_v.setValue(new_scroll_v)

            event.accept()
        else:
            # 普通滚轮：滚动
            super(ZoomableImageView, self).wheelEvent(event)

    def mousePressEvent(self, event):
        """鼠标按下：开始拖动"""
        if event.button() == Qt.LeftButton and self.original_pixmap is not None and not self.original_pixmap.isNull():
            # 只要有图片就可以拖动
            self.panning = True
            self.pan_start_pos = event.globalPos()  # 使用全局坐标
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super(ZoomableImageView, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """鼠标移动：拖动图片"""
        if self.panning:
            # 使用全局坐标计算移动距离
            current_pos = event.globalPos()
            delta = current_pos - self.pan_start_pos
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            self.pan_start_pos = current_pos
            event.accept()
            return
        super(ZoomableImageView, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """鼠标释放：结束拖动"""
        if event.button() == Qt.LeftButton and self.panning:
            self.panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super(ZoomableImageView, self).mouseReleaseEvent(event)

    def setText(self, text):
        """设置文本（兼容QLabel接口）"""
        self.image_label.setText(text)
        self.image_label.setAlignment(Qt.AlignCenter)

    def clear(self):
        """清空图片"""
        self.original_pixmap = None
        self.image_label.clear()
        self.scale_factor = 1.0

    def resizeEvent(self, event):
        """窗口大小改变时"""
        super(ZoomableImageView, self).resizeEvent(event)
        # 不自动适应，保持用户当前的缩放比例


class ComboBox(QWidget):
    def __init__(self, parent=None, items=[]):
        super(ComboBox, self).__init__(parent)

        layout = QHBoxLayout()
        self.cb = QComboBox()
        self.items = items
        self.cb.addItems(self.items)

        self.cb.currentIndexChanged.connect(parent.combo_selection_changed)

        layout.addWidget(self.cb)
        self.setLayout(layout)

    def update_items(self, items):
        self.items = items

        self.cb.clear()
        self.cb.addItems(self.items)


def save(self, class_list=[], target_file=None):
    out_file = None  # Update yolo .txt
    out_class_file = None  # Update class list .txt
    label_list = []
    classes_file = ''
    if target_file is None:
        out_file = open(
            self.filename + TXT_EXT, 'w', encoding=ENCODE_METHOD)
        classes_file = os.path.join(
            os.path.dirname(
                os.path.abspath(
                    self.filename)),
            "classes.txt")
        out_class_file = open(classes_file, 'w')

    else:

        out_file = codecs.open(target_file, 'w', encoding=ENCODE_METHOD)
        classes_file = os.path.join(
            os.path.dirname(
                os.path.abspath(target_file)),
            "classes.txt")
        a = 0
        if os.path.exists(classes_file):
            a = 1
            with open(classes_file, 'r', encoding='utf-8') as f:
                for line in f:
                    label_list.append(line.rstrip())
        else:
            out_class_file = open(classes_file, 'w')

    for box in self.box_list:
        class_index, x_center, y_center, w, h = self.bnd_box_to_yolo_line(
            box, class_list)

        if class_list[class_index] not in label_list:
            label_list.append(class_list[class_index])
            class_index = len(label_list) - 1
        else:
            class_index = label_list.index(class_list[class_index])
        out_file.write(
            "%d %.6f %.6f %.6f %.6f\n" %
            (class_index, x_center, y_center, w, h))

    for poly in self.polygon_list:
        name = poly['name']
        if name not in class_list:
            class_list.append(name)
        if name not in label_list:
            label_list.append(name)
        class_index = label_list.index(
            name) if a == 1 else class_list.index(name)
        parts = [str(class_index)]
        for x, y in poly['points']:
            xn = float(x) / self.img_size[1]
            yn = float(y) / self.img_size[0]
            parts.append("%.6f" % xn)
            parts.append("%.6f" % yn)
        out_file.write(" ".join(parts) + "\n")

    if a == 0:
        for c in class_list:
            out_class_file.write(c + '\n')

    if a == 1:
        with open(classes_file, 'w') as f:
            for c in label_list:
                f.write(c + '\n')

    if out_class_file is not None:
        out_class_file.close()

    out_file.close()


YOLOWriter.save = save

__appname__ = 'labelImg'


def __load_bundle(self, path):
    PROP_SEPERATOR = '='
    f = QFile(path)
    if f.exists():
        if f.open(QIODevice.ReadOnly | QFile.Text):
            text = QTextStream(f)
            text.setCodec("UTF-8")

        while not text.atEnd():
            line = ustr(text.readLine())
            key_value = line.split(PROP_SEPERATOR)
            key = key_value[0].strip()
            value = PROP_SEPERATOR.join(key_value[1:]).strip().strip('"')
            self.id_to_message[key] = value
        self.id_to_message['loadModel'] = 'load Model'
        self.id_to_message['loadModelDetail'] = 'load Model Detail'
        self.id_to_message['syncModelClasses'] = 'Use Model Classes'
        self.id_to_message[
            'syncModelClassesDetail'] = 'Write model classes to classes.txt (overwrite if exists)'
        self.id_to_message['detect'] = 'Detect'
        self.id_to_message['detectImg'] = 'Detect Img'

        self.id_to_message['savetxt'] = 'Save all Pre labels'
        self.id_to_message['saveTxt'] = 'save Txt'
        self.id_to_message['savecurrentxt'] = 'Save Curren Txt'
        self.id_to_message['saveCurrentTxt'] = 'Save txt'

        self.id_to_message['errordata'] = 'Save error data'
        self.id_to_message['saveCurrentTxt'] = 'Save txt'

        self.id_to_message['loaddata'] = 'load infer data'
        # self.id_to_message['loaddata'] = 'Save txt'
        f.close()


libs.stringBundle.StringBundle._StringBundle__load_bundle = __load_bundle


def _cv_bgr_to_qimage(bgr: np.ndarray) -> QImage:
    if bgr is None:
        return QImage()
    if bgr.ndim == 2:
        h, w = bgr.shape
        bytes_per_line = w
        return QImage(bgr.data, w, h, bytes_per_line,
                      QImage.Format_Grayscale8).copy()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()


def _format_seconds(seconds: float) -> str:
    try:
        seconds = float(seconds)
    except Exception:
        seconds = 0.0
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


class VideoPlayerWidget(QWidget):
    """
    Lightweight video player based on OpenCV + QLabel.
    Avoids depending on QtMultimedia (often missing in packaged builds).
    """

    videoLoaded = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cap = None
        self._video_path = None
        self._fps = 0.0
        self._frame_count = 0
        self._duration_s = 0.0
        self._width = 0
        self._height = 0
        self._playing = False
        self._updating_slider = False
        self._sequence_paths = None
        self._sequence_index = 0
        self._sequence_images = None

        self.video_label = QLabel("未加载视频")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(320, 180)
        self.video_label.setStyleSheet(
            "QLabel { background: #111; color: #ddd; border: 1px solid #444; }")

        self.play_btn = QPushButton("播放")
        self.pause_btn = QPushButton("暂停")
        self.stop_btn = QPushButton("停止")

        self.time_label = QLabel("00:00.00 / 00:00.00")
        self.time_label.setAlignment(Qt.AlignCenter)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setSingleStep(1)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.play_btn)
        btn_row.addWidget(self.pause_btn)
        btn_row.addWidget(self.stop_btn)

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.video_label, 1)
        layout.addWidget(self.time_label)
        layout.addWidget(self.slider)
        layout.addLayout(btn_row)
        self.setLayout(layout)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)

        self.play_btn.clicked.connect(self.play)
        self.pause_btn.clicked.connect(self.pause)
        self.stop_btn.clicked.connect(self.stop)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.valueChanged.connect(self._on_slider_value_changed)

    def isLoaded(self) -> bool:
        return (
                self._cap is not None and self._video_path is not None) or (
                self._sequence_paths is not None)

    def videoPath(self) -> str:
        return self._video_path or ""

    def fps(self) -> float:
        return float(self._fps or 0.0)

    def frameCount(self) -> int:
        return int(self._frame_count or 0)

    def durationSeconds(self) -> float:
        return float(self._duration_s or 0.0)

    def width(self) -> int:
        return int(self._width or 0)

    def height(self) -> int:
        return int(self._height or 0)

    def openVideo(self, video_path: str) -> bool:
        self.stop()
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False
        except Exception:
            return False

        self._cap = cap
        self._video_path = video_path
        self._fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if self._fps > 0 and self._frame_count > 0:
            self._duration_s = self._frame_count / self._fps
        else:
            self._duration_s = 0.0

        self.slider.setRange(0, max(0, self._frame_count - 1))
        self._set_pos_frames(0)
        self._read_and_show_current()
        self._update_time_label()
        self.videoLoaded.emit(video_path)
        return True

    def play(self):
        if not self.isLoaded():
            return
        self._playing = True
        interval_ms = 33
        if self._fps and self._fps > 0:
            interval_ms = max(1, int(1000.0 / self._fps))
        self._timer.start(interval_ms)

    def pause(self):
        self._playing = False
        self._timer.stop()

    def stop(self):
        self._playing = False
        self._timer.stop()
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None
        self._video_path = None
        self._sequence_paths = None
        self._sequence_index = 0
        self._sequence_images = None
        self._fps = 0.0
        self._frame_count = 0
        self._duration_s = 0.0
        self.slider.setRange(0, 0)
        self.video_label.setText("未加载视频")
        self.time_label.setText("00:00.00 / 00:00.00")

    def openImageSequence(self, paths: list, fps: float = 10.0) -> bool:
        self.stop()
        paths = [
            p for p in (
                    paths or []) if isinstance(
                p, str) and os.path.exists(p)]
        if not paths:
            return False
        self._sequence_paths = paths
        self._sequence_images = None
        self._sequence_index = 0
        self._video_path = ""
        self._cap = None
        self._fps = float(fps or 10.0)
        self._frame_count = len(paths)
        self._duration_s = (
                self._frame_count /
                self._fps) if self._fps > 0 else 0.0
        self.slider.setRange(0, max(0, self._frame_count - 1))
        self._read_and_show_sequence_index(0)
        self._update_time_label()
        return True

    def openQImageSequence(self, images: list, fps: float = 10.0) -> bool:
        self.stop()
        images = [
            im for im in (
                    images or []) if isinstance(
                im,
                QImage) and not im.isNull()]
        if not images:
            return False
        self._sequence_images = images
        self._sequence_paths = None
        self._sequence_index = 0
        self._video_path = ""
        self._cap = None
        self._fps = float(fps or 10.0)
        self._frame_count = len(images)
        self._duration_s = (
                self._frame_count /
                self._fps) if self._fps > 0 else 0.0
        self.slider.setRange(0, max(0, self._frame_count - 1))
        self._read_and_show_sequence_index(0)
        self._update_time_label()
        return True

    def closeEvent(self, event):
        try:
            self.stop()
        except Exception:
            pass
        super().closeEvent(event)

    def _pos_frames(self) -> int:
        if not self.isLoaded():
            return 0
        if self._sequence_paths is not None or self._sequence_images is not None:
            return int(self._sequence_index or 0)
        try:
            return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        except Exception:
            return 0

    def _set_pos_frames(self, frame_idx: int):
        if not self.isLoaded():
            return
        frame_idx = int(max(0, min(frame_idx, max(0, self._frame_count - 1))))
        if self._sequence_paths is not None or self._sequence_images is not None:
            self._sequence_index = frame_idx
            return
        try:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        except Exception:
            pass

    def _read_and_show_sequence_index(self, idx: int):
        if self._sequence_paths is None and self._sequence_images is None:
            return
        seq_len = len(
            self._sequence_images) if self._sequence_images is not None else len(
            self._sequence_paths)
        idx = int(max(0, min(idx, max(0, seq_len - 1))))
        self._sequence_index = idx
        qimg = None
        if self._sequence_images is not None:
            try:
                qimg = self._sequence_images[idx]
            except Exception:
                qimg = None
        else:
            try:
                frame = cv2.imdecode(
                    np.fromfile(
                        self._sequence_paths[idx],
                        dtype=np.uint8),
                    cv2.IMREAD_COLOR)
            except Exception:
                frame = None
            if frame is None:
                return
            qimg = _cv_bgr_to_qimage(frame)
        if qimg.isNull():
            return
        pix = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(
            pix.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation))
        self._updating_slider = True
        try:
            self.slider.setValue(idx)
        finally:
            self._updating_slider = False
        self._update_time_label()

    def _read_and_show_current(self):
        if not self.isLoaded():
            return
        if self._sequence_paths is not None or self._sequence_images is not None:
            self._read_and_show_sequence_index(self._sequence_index)
            self._sequence_index += 1
            seq_len = len(
                self._sequence_images) if self._sequence_images is not None else len(
                self._sequence_paths)
            if self._sequence_index >= seq_len:
                self.pause()
            return
        try:
            ok, frame = self._cap.read()
        except Exception:
            ok, frame = False, None
        if not ok or frame is None:
            self.pause()
            return
        qimg = _cv_bgr_to_qimage(frame)
        if qimg.isNull():
            return
        pix = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(
            pix.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation))
        # Sync slider to current frame position.
        pos = self._pos_frames()
        self._updating_slider = True
        try:
            self.slider.setValue(
                max(0, min(pos, max(0, self._frame_count - 1))))
        finally:
            self._updating_slider = False
        self._update_time_label()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Re-scale current pixmap on resize for better UX.
        try:
            pix = self.video_label.pixmap()
            if pix:
                self.video_label.setPixmap(
                    pix.scaled(
                        self.video_label.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation))
        except Exception:
            pass

    def _update_time_label(self):
        if not self.isLoaded() or not self._fps or self._fps <= 0:
            self.time_label.setText("00:00.00 / 00:00.00")
            return
        pos = max(0, self._pos_frames())
        cur_s = pos / self._fps
        self.time_label.setText(
            f"{_format_seconds(cur_s)} / {_format_seconds(self._duration_s)}")

    def _on_tick(self):
        if not self._playing:
            return
        self._read_and_show_current()

    def _on_slider_pressed(self):
        # Pause while seeking.
        self.pause()

    def _on_slider_released(self):
        if not self.isLoaded():
            return
        self._set_pos_frames(self.slider.value())
        self._read_and_show_current()

    def _on_slider_value_changed(self, value: int):
        if self._updating_slider or not self.isLoaded():
            return
        # Live-preview while dragging is expensive; only update time label.
        if self._fps and self._fps > 0:
            self.time_label.setText(
                f"{_format_seconds(value / self._fps)} / {_format_seconds(self._duration_s)}")


class VideoDetectDialog(QDialog):
    def __init__(self, parent=None, defaults=None, classes=None):
        super().__init__(parent)
        self.setWindowTitle("抽帧完成：检测设置")
        self.setModal(True)

        defaults = defaults or {}
        self._classes = list(classes or [])
        self._label_map_inputs = {}

        self.detect_chk = QCheckBox("进行检测")
        self.detect_chk.setChecked(True)

        self.use_zh_chk = QCheckBox("使用中文标签(映射)")
        self.use_zh_chk.setChecked(bool(defaults.get("use_zh", False)))
        self.show_conf_chk = QCheckBox("显示置信度")
        self.show_conf_chk.setChecked(bool(defaults.get("show_conf", True)))

        # Label mapping UI:
        # Prefer per-class Chinese input (auto from model classes). Fallback to
        # free text.
        self.label_map_edit = QPlainTextEdit()
        self.label_map_edit.setPlaceholderText(
            "标签映射：每行一个  英文=中文  (例：person=人员)")
        self.label_map_edit.setPlainText(
            str(defaults.get("label_map_text", "") or ""))
        self.label_map_edit.setFixedHeight(120)

        self.label_map_form = QWidget()
        self.label_map_form_layout = QGridLayout()
        self.label_map_form_layout.setContentsMargins(0, 0, 0, 0)
        self.label_map_form_layout.setHorizontalSpacing(8)
        self.label_map_form_layout.setVerticalSpacing(6)
        self.label_map_form.setLayout(self.label_map_form_layout)
        self.label_map_form_scroll = QScrollArea()
        self.label_map_form_scroll.setWidgetResizable(True)
        self.label_map_form_scroll.setWidget(self.label_map_form)
        self.label_map_form_scroll.setFixedHeight(180)

        self.export_video_chk = QCheckBox("导出检测视频")
        self.export_video_chk.setChecked(
            bool(defaults.get("export_video", False)))
        self.export_path_edit = QLineEdit()
        default_export = defaults.get("export_path", "")
        if default_export:
            self.export_path_edit.setText(default_export)
        self.export_path_btn = QPushButton("选择导出文件")
        self.export_fps_spin = QDoubleSpinBox()
        self.export_fps_spin.setRange(0.0, 240.0)
        self.export_fps_spin.setValue(
            float(defaults.get("export_fps", 0.0) or 0.0))
        self.export_fps_spin.setDecimals(2)
        self.export_fps_spin.setSuffix(" fps(0=原视频)")

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        top_row = QHBoxLayout()
        top_row.addWidget(self.detect_chk)
        top_row.addStretch(1)

        opts_row = QHBoxLayout()
        opts_row.addWidget(self.use_zh_chk)
        opts_row.addWidget(self.show_conf_chk)
        opts_row.addStretch(1)

        exp_grid = QGridLayout()
        exp_grid.addWidget(self.export_video_chk, 0, 0, 1, 3)
        exp_grid.addWidget(QLabel("导出路径:"), 1, 0)
        exp_grid.addWidget(self.export_path_edit, 1, 1)
        exp_grid.addWidget(self.export_path_btn, 1, 2)
        exp_grid.addWidget(QLabel("导出帧率:"), 2, 0)
        exp_grid.addWidget(self.export_fps_spin, 2, 1, 1, 2)

        layout = QVBoxLayout()
        layout.addLayout(top_row)
        layout.addLayout(opts_row)
        layout.addWidget(QLabel("标签映射:"))
        # If we have model classes, show per-class inputs; otherwise show
        # free-text mapping.
        if self._classes:
            layout.addWidget(QLabel("按模型类别逐行填写中文（留空表示不替换）："))
            layout.addWidget(self.label_map_form_scroll)
        else:
            layout.addWidget(self.label_map_edit)
        exp_box = QGroupBox("输出视频(可选)")
        exp_box.setLayout(exp_grid)
        layout.addWidget(exp_box)
        layout.addWidget(btns)
        self.setLayout(layout)

        self.export_path_btn.clicked.connect(self._choose_export_path)
        self.export_video_chk.toggled.connect(self._update_export_enable)
        self._update_export_enable()

        # Build per-class inputs (prefill from defaults mapping).
        if self._classes:
            defaults_map = defaults.get("label_map") or self.parse_label_map_text(
                defaults.get("label_map_text", "") or "")
            self._build_label_map_inputs(self._classes, defaults_map)

    def _build_label_map_inputs(self, classes, defaults_map: dict):
        # Header
        self.label_map_form_layout.addWidget(QLabel("英文"), 0, 0)
        self.label_map_form_layout.addWidget(QLabel("中文"), 0, 1)
        for row, en in enumerate(classes, start=1):
            en_str = str(en)
            self.label_map_form_layout.addWidget(QLabel(en_str), row, 0)
            edit = QLineEdit()
            edit.setPlaceholderText("输入中文(可选)")
            try:
                edit.setText(str(defaults_map.get(en_str, ""))
                             if defaults_map else "")
            except Exception:
                pass
            self.label_map_form_layout.addWidget(edit, row, 1)
            self._label_map_inputs[en_str] = edit

    def _update_export_enable(self):
        enabled = self.export_video_chk.isChecked()
        self.export_path_edit.setEnabled(enabled)
        self.export_path_btn.setEnabled(enabled)
        self.export_fps_spin.setEnabled(enabled)

    def _choose_export_path(self):
        cur = self.export_path_edit.text().strip()
        fn, _ = QFileDialog.getSaveFileName(
            self,
            "选择导出视频文件",
            cur or ".",
            "MP4 Video (*.mp4);;AVI Video (*.avi);;All Files (*.*)",
        )
        if fn:
            self.export_path_edit.setText(fn)

    @staticmethod
    def parse_label_map_text(text: str) -> dict:
        mapping = {}
        for raw in (text or "").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k:
                mapping[k] = v
        return mapping

    def values(self) -> dict:
        if self._classes:
            mapping = {}
            for en, edit in (self._label_map_inputs or {}).items():
                zh = ""
                try:
                    zh = (edit.text() or "").strip()
                except Exception:
                    zh = ""
                if zh:
                    mapping[str(en)] = zh
            label_map_text = "\n".join(
                [f"{k}={v}" for k, v in mapping.items()])
            label_map = mapping
        else:
            label_map_text = self.label_map_edit.toPlainText()
            label_map = self.parse_label_map_text(label_map_text)
        return {
            "do_detect": bool(self.detect_chk.isChecked()),
            "use_zh": bool(self.use_zh_chk.isChecked()),
            "show_conf": bool(self.show_conf_chk.isChecked()),
            "label_map_text": label_map_text,
            "label_map": label_map,
            "export_video": bool(self.export_video_chk.isChecked()),
            "export_path": self.export_path_edit.text().strip(),
            "export_fps": float(self.export_fps_spin.value()),
        }


class VideoProcessWorker(QObject):
    progress = pyqtSignal(int, str)  # percent, message
    finished = pyqtSignal(dict)  # result dict
    failed = pyqtSignal(str)  # error message

    def __init__(
            self,
            video_path: str,
            output_dir: str,
            mode: str,
            target_fps: float,
            interval_s: float,
            start_s: float,
            end_s: float,
    ):
        super().__init__()
        self._stopped = False
        self.video_path = video_path
        self.output_dir = output_dir
        self.mode = mode
        self.target_fps = float(target_fps or 0.0)
        self.interval_s = float(interval_s or 0.0)
        self.start_s = float(start_s or 0.0)
        self.end_s = float(end_s or 0.0)

    @staticmethod
    def _safe_mkdir(p: str):
        os.makedirs(p, exist_ok=True)

    @staticmethod
    def _parse_label(label: str, label_map: dict, use_zh: bool) -> str:
        if not use_zh:
            return label
        try:
            return str(label_map.get(label, label))
        except Exception:
            return label

    @staticmethod
    def _draw_boxes(
            frame_bgr: np.ndarray,
            boxes,
            scores,
            class_ids,
            classes,
            label_map: dict,
            use_zh: bool,
            show_conf: bool,
            color_palette=None,
    ):
        if frame_bgr is None:
            return frame_bgr
        # Draw rectangles first with OpenCV (fast), then draw text with PIL to
        # support Chinese.
        draw_specs = []
        for i in range(len(boxes)):
            try:
                x1, y1, x2, y2 = [int(v) for v in boxes[i]]
            except Exception:
                continue
            try:
                cid = int(class_ids[i])
            except Exception:
                cid = -1
            try:
                score = float(scores[i])
            except Exception:
                score = None
            if cid >= 0 and cid < len(classes):
                name = str(classes[cid])
            else:
                name = str(cid if cid >= 0 else "cls")
            name = VideoProcessWorker._parse_label(name, label_map, use_zh)
            label = name
            if show_conf and score is not None:
                label = f"{label} {score:.2f}"
            color_bgr = (0, 255, 0)
            try:
                if color_palette is not None and cid >= 0:
                    c = color_palette[cid]
                    color_bgr = (int(c[0]), int(c[1]), int(c[2]))
            except Exception:
                color_bgr = (0, 255, 0)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2),
                          color_bgr, 2, cv2.LINE_AA)
            draw_specs.append((x1, y1, label, color_bgr))

        if not draw_specs:
            return frame_bgr

        try:
            from PIL import Image, ImageDraw, ImageFont

            def _load_font(size: int):
                # Try common Chinese fonts on Windows, fallback to default.
                candidates = []
                try:
                    win_font_dir = os.path.join(
                        os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
                    candidates.extend([
                        os.path.join(
                            win_font_dir, "msyh.ttc"),  # Microsoft YaHei
                        os.path.join(win_font_dir, "msyhbd.ttc"),
                        os.path.join(win_font_dir, "simhei.ttf"),  # SimHei
                        os.path.join(win_font_dir, "simsun.ttc"),  # SimSun
                    ])
                except Exception:
                    pass
                candidates.extend(["msyh.ttc", "simhei.ttf", "simsun.ttc"])
                for fp in candidates:
                    try:
                        if os.path.exists(fp):
                            return ImageFont.truetype(fp, size=size)
                    except Exception:
                        continue
                try:
                    return ImageFont.load_default()
                except Exception:
                    return None

            # Convert once per frame.
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            draw = ImageDraw.Draw(pil_img)
            font = _load_font(18)
            pad_x, pad_y = 3, 2
            for x1, y1, label, color_bgr in draw_specs:
                try:
                    if font is not None:
                        bbox = draw.textbbox((0, 0), label, font=font)
                        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    else:
                        tw, th = 80, 18
                except Exception:
                    tw, th = 80, 18
                y0 = max(0, int(y1) - th - 2 * pad_y - 2)
                try:
                    fill_rgb = (int(color_bgr[2]), int(
                        color_bgr[1]), int(color_bgr[0]))
                except Exception:
                    fill_rgb = (0, 255, 0)
                # Background fill (green) and black text.
                draw.rectangle([x1, y0, x1 + tw + 2 * pad_x,
                                y0 + th + 2 * pad_y], fill=fill_rgb)
                draw.text((x1 + pad_x, y0 + pad_y), label,
                          fill=(0, 0, 0), font=font)
            frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            # Fallback: keep rectangles only.
            pass
        return frame_bgr

    def run(self):
        try:
            if not self.video_path or not os.path.exists(self.video_path):
                self.failed.emit("视频路径无效")
                return
            if not self.output_dir:
                self.failed.emit("输出目录为空")
                return

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base = os.path.splitext(os.path.basename(self.video_path))[0]
            frames_dir = os.path.join(self.output_dir, f"{base}_frames_{ts}")
            self._safe_mkdir(frames_dir)

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.failed.emit("无法打开视频")
                return

            src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

            start_frame = 0
            end_frame = total_frames - 1 if total_frames > 0 else -1
            if src_fps and src_fps > 0:
                start_frame = int(max(0, round(self.start_s * src_fps)))
                if self.end_s and self.end_s > 0:
                    end_frame = int(
                        min(end_frame, round(self.end_s * src_fps)))

            if end_frame >= 0 and start_frame > end_frame:
                self.failed.emit("起止时间设置不正确")
                cap.release()
                return

            # Determine sampling strategy
            if self.mode == "fps":
                if not src_fps or src_fps <= 0:
                    step = 1
                else:
                    tfps = max(0.0001, float(self.target_fps or 0.0))
                    step = max(1, int(round(src_fps / tfps)))
            else:
                # interval seconds
                if not src_fps or src_fps <= 0:
                    step = 1
                else:
                    interval = max(0.0001, float(self.interval_s or 0.0))
                    step = max(1, int(round(interval * src_fps)))

            frame_paths = []
            det_summary = []

            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current = start_frame
            extracted = 0
            max_to_process = (
                    end_frame -
                    start_frame +
                    1) if end_frame >= 0 else 0
            while True:
                if end_frame >= 0 and current > end_frame:
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if self._stopped:
                    cap.release()
                    self.failed.emit("用户停止")
                    return
                if ((current - start_frame) % step) == 0:
                    t_s = (
                            current /
                            src_fps) if (
                            src_fps and src_fps > 0) else 0.0
                    out_name = f"{base}_f{current:08d}_t{t_s:010.3f}.jpg"
                    out_path = os.path.join(frames_dir, out_name)
                    cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[
                        1].tofile(out_path)
                    frame_paths.append(out_path)
                    extracted += 1

                current += 1
                if max_to_process > 0:
                    percent = int(
                        min(90, (current - start_frame) * 90 / max_to_process))
                    self.progress.emit(percent, f"抽帧中... {len(frame_paths)} 张")

            cap.release()

            if not frame_paths:
                self.failed.emit("没有抽取到任何帧（请检查参数）")
                return

            self.progress.emit(100, f"抽帧完成：{len(frame_paths)} 张")
            self.finished.emit({
                "video": self.video_path,
                "output_dir": self.output_dir,
                "base": base,
                "ts": ts,
                "frames_dir": frames_dir,
                "frames": frame_paths,
                "src_fps": src_fps,
                "width": width,
                "height": height,
                "total_frames": total_frames,
            })
        except Exception as e:
            self.failed.emit(str(e))


class VideoDetectWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(
            self,
            video_path: str,
            output_dir: str,
            base: str,
            ts: str,
            frames: list,
            width: int,
            height: int,
            src_fps: float,
            yolo_model,
            use_zh: bool,
            show_conf: bool,
            label_map: dict,
            export_video: bool,
            export_video_path: str,
            export_fps: float,
    ):
        super().__init__()
        self._stopped = False
        self.video_path = video_path
        self.output_dir = output_dir
        self.base = base
        self.ts = ts
        self.frames = frames or []
        self.width = int(width or 0)
        self.height = int(height or 0)
        self.src_fps = float(src_fps or 0.0)
        self.yolo_model = yolo_model
        self.use_zh = bool(use_zh)
        self.show_conf = bool(show_conf)
        self.label_map = label_map or {}
        self.export_video = bool(export_video)
        self.export_video_path = export_video_path or ""
        self.export_fps = float(export_fps or 0.0)

    @staticmethod
    def _safe_mkdir(p: str):
        os.makedirs(p, exist_ok=True)

    def run(self):
        try:
            if not self.frames:
                self.failed.emit("没有帧可检测")
                return
            if self.yolo_model is None:
                self.failed.emit("未加载模型")
                return

            det_summary = []
            classes = getattr(self.yolo_model, "classes", [])

            for idx, fp in enumerate(self.frames):
                if self._stopped:
                    self.failed.emit("用户停止")
                    return
                try:
                    frame = cv2.imdecode(
                        np.fromfile(
                            fp,
                            dtype=np.uint8),
                        cv2.IMREAD_COLOR)
                    if isinstance(self.yolo_model, YOLOv8_Seg):
                        rendered, boxes, scores, class_ids, _, _mask = self.yolo_model.infer(
                            frame)
                    else:
                        rendered, boxes, scores, class_ids, _ = self.yolo_model.infer(
                            frame)

                    items = []
                    for i in range(len(boxes)):
                        try:
                            cid = int(class_ids[i])
                        except Exception:
                            cid = -1
                        name = str(
                            classes[cid]) if (
                                cid >= 0 and cid < len(classes)) else str(cid)
                        name_mapped = VideoProcessWorker._parse_label(
                            name, self.label_map, self.use_zh)
                        items.append({
                            "class_id": cid,
                            "label_en": name,
                            "label": name_mapped,
                            "score": float(scores[i]) if i < len(scores) else None,
                            "box_xyxy": [int(v) for v in boxes[i]],
                        })
                    det_summary.append({
                        "frame": fp,
                        "result": [boxes, scores, class_ids],
                        "detections": items,
                        "rendered": rendered,
                    })
                except Exception:
                    det_summary.append({"frame": fp, "result": [
                        [], [], []], "detections": [], "error": "detect_failed"})

                percent = int((idx + 1) * 90 / max(1, len(self.frames)))
                self.progress.emit(percent,
                                   f"检测中... {idx + 1}/{len(self.frames)}")

            export_out = ""
            if self.export_video:
                export_out = self.export_video_path
                if not export_out:
                    export_out = os.path.join(self.output_dir, f"{self.base}_annotated_{self.ts}.mp4")

                out_fps = self.export_fps if self.export_fps > 0 else (
                    self.src_fps if self.src_fps > 0 else 25.0)

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    export_out, fourcc, out_fps, (self.width, self.height))
                if not writer.isOpened():
                    export_out = ""
                else:
                    for r in det_summary:
                        try:
                            # 优先使用 infer() 返回的已渲染图像（含 boxes/masks）
                            fr = r.get("rendered")
                            if fr is None:
                                fp = r.get("frame", "")
                                fr = cv2.imdecode(
                                    np.fromfile(
                                        fp,
                                        dtype=np.uint8),
                                    cv2.IMREAD_COLOR)
                                if fr is None:
                                    continue
                                boxes, scores, class_ids = r.get(
                                    "result", [[], [], []])
                                fr = VideoProcessWorker._draw_boxes(
                                    fr,
                                    boxes=boxes,
                                    scores=scores,
                                    class_ids=class_ids,
                                    classes=classes,
                                    label_map=self.label_map,
                                    use_zh=self.use_zh,
                                    show_conf=self.show_conf,
                                )
                            if fr.shape[1] != self.width or fr.shape[0] != self.height:
                                fr = cv2.resize(
                                    fr, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                            writer.write(fr)
                        except Exception:
                            continue
                    try:
                        writer.release()
                    except Exception:
                        pass

            self.progress.emit(100, "检测完成")
            self.finished.emit({
                "video": self.video_path,
                "detections": det_summary,
                "export_video": export_out,
            })
        except Exception as e:
            self.failed.emit(str(e))


class YOLOv7:

    def __init__(
            self,
            weights,
            conf_thresh=0.3,
            iou_thresh=0.5,
            device=0):  # -> None:
        # self.img_new_shape = (size, 1280) if weights == './models\\yp' else (size, size)  # 改换大小
        # print(self.img_new_shape)
        self.weights = weights
        self.device = device
        self.conf_threshold = conf_thresh
        self.iou_threshold = iou_thresh
        self.input_width = 640
        self.input_height = 640
        self.classes = ["a"]
        self.color_palette = np.random.uniform(
            0, 255, size=(len(self.classes), 3))
        self.init_engine()
        self.img_new_shape = [self.input_height, self.input_width]

    def __call__(self, image):
        return self.infer(image)

    def warmup(self, num):
        dummy = np.zeros(
            (self.input_width,
             self.input_height,
             3),
            dtype=np.uint8) + 114  # 灰色
        for i in range(num):
            self.preprocess(dummy)
            outputs = self.predict(self.im)
            # Run inference using the preprocessed image data

    def boxes_postprocess(self, boxes, width, height, thresh=5):
        new_boxes = []
        x_min_thresh = int(width * 0.01)
        y_min_thresh = int(height * 0.01)

        x_max_thresh = width - x_min_thresh
        y_max_thresh = height - y_min_thresh

        for box in boxes:
            box[0] = min(x_max_thresh, max(box[0], x_min_thresh))
            box[1] = min(y_max_thresh, max(box[1], y_min_thresh))

            box[2] = min(x_max_thresh, max(box[2], x_min_thresh))
            box[3] = min(y_max_thresh, max(box[3], y_min_thresh))

            new_boxes.append(box)
        self.boxes = new_boxes
        return new_boxes

    def copy_postprocess_output(self, boxes, scores, class_ids):
        self.boxes = boxes.copy()
        self.scores = scores.copy()
        self.class_ids = class_ids.copy()

    def plot_one_box(
            self,
            x,
            im,
            color=(
                    128,
                    128,
                    128),
            label=None,
            score=None,
            line_thickness=3):

        # if label not in {"hight-risk", "low-risk",'birdnest'}:
        #    return im
        # 线宽（框 + 字体）
        input_is_pil = isinstance(im, Image.Image)
        lw = max(round(sum(im.size if input_is_pil else im.shape) /
                       2 * 0.0025), 5) or line_thickness

        tf = max(lw - 1, 1)  # font thickness
        sf = lw / 3  # font scale
        #
        # tl = round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1
        # tf = max(tl - 4, 1)  # 字体线宽

        # 坐标
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

        # 画边框
        cv2.rectangle(im, c1, c2, color, thickness=lw, lineType=cv2.LINE_AA)

        if label:
            # 构造文本内容
            text = label
            if score is not None:
                text += f" {score:.2f}"

            # 字体缩放
            fontScale = sf

            # 获取文本尺寸
            t_size = cv2.getTextSize(
                text, 0, fontScale=fontScale, thickness=tf)[0]
            text_width, text_height = t_size

            # 默认将文本画在框上方
            text_bottom_left = (c1[0], c1[1] - 2)
            rect_top_left = (c1[0], c1[1] - text_height - 4)
            rect_bottom_right = (c1[0] + text_width, c1[1])

            h, w = im.shape[:2]

            # ✅ 边界判断：如果顶部太靠近图像上边缘，就往下画
            if rect_top_left[1] < 0:
                text_bottom_left = (c1[0], c1[1] + text_height + 2)
                rect_top_left = (c1[0], c1[1])
                rect_bottom_right = (
                    c1[0] + text_width, c1[1] + text_height + 4)
                if rect_bottom_right[1] > h:
                    dy = rect_bottom_right[1] - h + 2  # 多留2像素边距
                    rect_top_left = (rect_top_left[0], rect_top_left[1] - dy)
                    rect_bottom_right = (
                        rect_bottom_right[0], rect_bottom_right[1] - dy)
                    text_bottom_left = (
                        text_bottom_left[0], text_bottom_left[1] - dy)

            # ✅ 左右越界修正（放在上下修正之后）
            if rect_bottom_right[0] > w:
                dx = rect_bottom_right[0] - w + 2  # 多留2像素边距
                rect_top_left = (rect_top_left[0] - dx, rect_top_left[1])
                rect_bottom_right = (
                    rect_bottom_right[0] - dx,
                    rect_bottom_right[1])
                text_bottom_left = (
                    text_bottom_left[0] - dx,
                    text_bottom_left[1])

            cv2.rectangle(im, rect_top_left, rect_bottom_right,
                          color, -1, cv2.LINE_AA)

            cv2.putText(im, text, text_bottom_left, 0, fontScale,
                        (225, 255, 255), thickness=tf, lineType=cv2.LINE_AA)
        return im

    def infer(self, image):
        if isinstance(image, str):
            with open(image, 'rb') as f:
                self.img = cv2.imdecode(
                    np.frombuffer(
                        f.read(),
                        np.uint8),
                    cv2.IMREAD_COLOR)
        else:
            self.img = image
        img_height, img_width = self.img.shape[:2]
        self.img_height = img_height
        self.img_width = img_width
        self.preprocess(self.img)

        outputs = self.predict(self.im)

        self.boxes = []
        self.scores = []
        self.class_ids = []
        self.boxesxywhns = []
        # for output in outputs[0]:
        #     print(output)
        for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
            score = round(float(score), 3)
            if (score < self.conf_threshold):
                continue
            box = np.array([x0, y0, x1, y1])
            box -= np.array(self.dwdh * 2)
            box /= self.ratio
            box = box.round().astype(np.int32).tolist()
            cls_id = int(cls_id)
            box[0] = min(max(0, box[0]), img_width - 1)
            box[1] = min(max(0, box[1]), img_height - 1)
            box[2] = min(max(0, box[2]), img_width - 1)
            box[3] = min(max(0, box[3]), img_height - 1)
            self.boxes.append(box)
            self.scores.append(score)
            self.class_ids.append(cls_id)
            # 新增：计算归一化后的 xywh 坐标
            x_center = (box[0] + box[2]) / 2.0 / img_width
            y_center = (box[1] + box[3]) / 2.0 / img_height
            width_norm = (box[2] - box[0]) / img_width
            height_norm = (box[3] - box[1]) / img_height
            self.boxesxywhns.append(
                [cls_id, x_center, y_center, width_norm, height_norm])

        return self.img, self.boxes, self.scores, self.class_ids, self.boxesxywhns

    def init_engine(self):

        so = ort.SessionOptions()
        so.log_severity_level = 1
        providers = [('CUDAExecutionProvider', {'device_id': self.device, }),
                     'CPUExecutionProvider', ] if self.device != -1 else ['CPUExecutionProvider']

        sess_options = ort.SessionOptions()
        self.session = ort.InferenceSession(self.weights, sess_options=sess_options,
                                            providers=providers)

        input_shape = self.session.get_inputs()[0].shape

        try:
            self.input_height = int(input_shape[2])
            self.input_width = int(input_shape[3])
        except Exception:
            self.input_height = 640
            self.input_width = 640

        model_meta = self.session.get_modelmeta()

        names_list = json.loads(model_meta.custom_metadata_map['names'])

        self.classes = names_list

        self.saveClassesTxt = os.path.join(
            os.path.dirname(self.weights), 'classes.txt')

        self.color_palette = np.random.uniform(
            0, 255, size=(len(self.classes), 3))

    def predict(self, im):
        outname = [i.name for i in self.session.get_outputs()]
        inname = [i.name for i in self.session.get_inputs()]
        inp = {inname[0]: im}
        outputs = self.session.run(outname, inp)[0]
        # print(outputs.shape)
        return outputs

    def preprocess(self, img):

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('img.png',img)
        image = img.copy()
        self.im, self.ratio, self.dwdh = self.letterbox(image, auto=False)

    def letterbox(
            self,
            im,
            color=(
                    114,
                    114,
                    114),
            auto=True,
            scaleup=True,
            stride=32):
        # 调整大小和垫图像，同时满足跨步多约束
        shape = im.shape[:2]  # current shape [height, width]
        new_shape = self.img_new_shape

        # 如果是单个size的话，就在这里变成一双
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # 尺度比 (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # 只缩小，不扩大(为了更好的val mAP)
            r = min(r, 1.0)

        # 计算填充
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
                 new_unpad[1]  # wh padding

        if auto:  # 最小矩形区域
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(
            im,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=color)  # add border

        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, 0)
        im = np.ascontiguousarray(im)
        im = im.astype(np.float32)
        im /= 255
        return im, r, (dw, dh)

    # def boxes_copy(self, boxes):

    #     self.boxes = boxes.copy()
    # def classids_copy(self, class_ids):

    #     self.class_ids = class_ids.copy()
    # def scores_copy(self, scores):

    #     self.scores = scores.copy()

    def crop_boxes(self, img):
        ret = []
        for box in self.boxes:
            left = box[0]
            top = box[1]
            right = box[2]
            bottom = box[3]

            roi = img[top:bottom, left:right].copy()
            ret.append(roi)
        return ret

    # def draw_detections(self, image, class_names, draw_scores=True,
    # mask_alpha=0.4,temperatures=[], zj_infos=[],obj_sizes=[]):

    #     return draw_detections(image, class_names, self.boxes, self.scores,
    # self.class_ids, mask_alpha,temperatures, zj_infos,obj_sizes)

    def draw_and_show(
            self,
            image,
            class_names,
            boxes,
            scores,
            class_ids,
            mask_alpha=0.3):
        rng = np.random.default_rng(3)
        colors = rng.uniform(0, 255, size=(10, 3))
        mask_img = image.copy()
        det_img = image.copy()

        img_height, img_width = image.shape[:2]
        size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        # Draw bounding boxes and labels of detections
        for box, score, class_id in zip(boxes, scores, class_ids):

            color = self.color_palette[class_id]
            if isinstance(box, list):
                x1, y1, x2, y2 = box
            else:
                x1, y1, x2, y2 = box.astype(int)

            # Draw rectangle
            cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)

            # Draw fill rectangle in mask image
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

            label = class_names[class_id]
            caption = f'{label} {int(score * 100)}%'
            (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                          fontScale=size, thickness=text_thickness)
            th = int(th * 1.2)

            cv2.rectangle(det_img, (x1, y1),
                          (x1 + tw, y1 - th), color, -1)
            cv2.rectangle(mask_img, (x1, y1),
                          (x1 + tw, y1 - th), color, -1)
            cv2.putText(det_img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

            cv2.putText(mask_img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

        return cv2.addWeighted(
            mask_img,
            mask_alpha,
            det_img,
            1 - mask_alpha,
            0)

    def UpdateResults(self, boxes, scores, class_ids):
        self.boxes = boxes
        self.scores = scores
        self.class_ids = class_ids


def compute_iou(box, boxes):
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area
    iou = intersection_area / union_area
    return iou


def nms(boxes, scores, iou_threshold):
    sorted_indices = np.argsort(scores)[::-1]
    keep_boxes = []
    while sorted_indices.size > 0:
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
        keep_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[keep_indices + 1]
    return keep_boxes


def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def boxes_postprocess(boxes, width, height):
    new_boxes = []
    x_min_thresh = 0
    y_min_thresh = 0
    x_max_thresh = width - 1
    y_max_thresh = height - 1
    for box in boxes:
        box[0] = min(x_max_thresh, max(box[0], x_min_thresh))
        box[1] = min(y_max_thresh, max(box[1], y_min_thresh))
        box[2] = min(x_max_thresh, max(box[2], x_min_thresh))
        box[3] = min(y_max_thresh, max(box[3], y_min_thresh))
        new_boxes.append(box)
    return new_boxes


class Colors:
    def __init__(self):
        hex = ('FF3838', 'FF9D97', 'E2535A', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


class YOLOv8:
    def __init__(
            self,
            onnx_model: str,
            confidence_thres: float,
            iou_thres: float):
        """
        Initialize an instance of the YOLOv8 class.

        Args:
            onnx_model (str): Path to the ONNX model.
            input_image (str): Path to the input image.
            confidence_thres (float): Confidence threshold for filtering detections.
            iou_thres (float): IoU threshold for non-maximum suppression.
        """
        self.onnx_model = onnx_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # Load the class names from the COCO dataset
        self.classes = ["a"]
        self.color_palette = np.random.uniform(
            0, 255, size=(len(self.classes), 3))

        self.model_inputs = 0
        self.session = None

        self.input_width = 640
        self.input_height = 640
        self.model_outputs = None

        self.loadonnx()

    def warmup(self, num):
        dummy = np.zeros((640, 640, 3), dtype=np.uint8) + 114  # 灰色
        for i in range(num):
            img_data, pad = self.preprocess(dummy)
            # Run inference using the preprocessed image data
            outputs = self.session.run(
                None, {self.model_inputs[0].name: img_data})

    def letterbox(self, img: np.ndarray, new_shape: Tuple[int, int] = (
            640, 640)) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Resize and reshape images while maintaining aspect ratio by adding padding.

        Args:
            img (np.ndarray): Input image to be resized.
            new_shape (Tuple[int, int]): Target shape (height, width) for the image.

        Returns:
            img (np.ndarray): Resized and padded image.
            pad (Tuple[int, int]): Padding values (top, left) applied to the image.
        """
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / \
                 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(
                114,
                114,
                114))

        return img, (top, left)

    def plot_one_box(
            self,
            x,
            im,
            color=(
                    128,
                    128,
                    128),
            label=None,
            score=None,
            line_thickness=3):

        # if label not in {"hight-risk", "low-risk",'birdnest'}:
        #    return im
        # 线宽（框 + 字体）
        input_is_pil = isinstance(im, Image.Image)
        lw = max(round(sum(im.size if input_is_pil else im.shape) /
                       2 * 0.0025), 5) or line_thickness

        tf = max(lw - 1, 1)  # font thickness
        sf = lw / 3  # font scale
        #
        # tl = round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1
        # tf = max(tl - 4, 1)  # 字体线宽

        # 坐标
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

        # 画边框
        cv2.rectangle(im, c1, c2, color, thickness=lw, lineType=cv2.LINE_AA)

        if label:
            # 构造文本内容
            text = label
            if score is not None:
                text += f" {score:.2f}"

            # 字体缩放
            fontScale = sf

            # 获取文本尺寸
            t_size = cv2.getTextSize(
                text, 0, fontScale=fontScale, thickness=tf)[0]
            text_width, text_height = t_size

            # 默认将文本画在框上方
            text_bottom_left = (c1[0], c1[1] - 2)
            rect_top_left = (c1[0], c1[1] - text_height - 4)
            rect_bottom_right = (c1[0] + text_width, c1[1])

            h, w = im.shape[:2]

            # ✅ 边界判断：如果顶部太靠近图像上边缘，就往下画
            if rect_top_left[1] < 0:
                text_bottom_left = (c1[0], c1[1] + text_height + 2)
                rect_top_left = (c1[0], c1[1])
                rect_bottom_right = (
                    c1[0] + text_width, c1[1] + text_height + 4)
                if rect_bottom_right[1] > h:
                    dy = rect_bottom_right[1] - h + 2  # 多留2像素边距
                    rect_top_left = (rect_top_left[0], rect_top_left[1] - dy)
                    rect_bottom_right = (
                        rect_bottom_right[0], rect_bottom_right[1] - dy)
                    text_bottom_left = (
                        text_bottom_left[0], text_bottom_left[1] - dy)

            # ✅ 左右越界修正（放在上下修正之后）
            if rect_bottom_right[0] > w:
                dx = rect_bottom_right[0] - w + 2  # 多留2像素边距
                rect_top_left = (rect_top_left[0] - dx, rect_top_left[1])
                rect_bottom_right = (
                    rect_bottom_right[0] - dx,
                    rect_bottom_right[1])
                text_bottom_left = (
                    text_bottom_left[0] - dx,
                    text_bottom_left[1])

            cv2.rectangle(im, rect_top_left, rect_bottom_right,
                          color, -1, cv2.LINE_AA)

            cv2.putText(im, text, text_bottom_left, 0, fontScale,
                        (225, 255, 255), thickness=tf, lineType=cv2.LINE_AA)
        return im

    def draw_detections(self, img, box, score, class_id):
        """Draw bounding boxes and labels on the input image based on the detected objects."""
        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # 线宽（框 + 字体）
        tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        tf = max(tl - 4, 1)  # 字体线宽

        fontScale = tl / 5.5

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)),
                      (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x +
                                                     label_width, label_y + label_height), color, cv2.FILLED
        )

        # Draw the label text on the image
        cv2.putText(
            img,
            label,
            (label_x,
             label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            (0,
             0,
             0),
            tf,
            cv2.LINE_AA)

    def preprocess(self, input) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocess the input image before performing inference.

        This method reads the input image, converts its color space, applies letterboxing to maintain aspect ratio,
        normalizes pixel values, and prepares the image data for model input.

        Returns:
            image_data (np.ndarray): Preprocessed image data ready for inference with shape (1, 3, height, width).
            pad (Tuple[int, int]): Padding values (top, left) applied during letterboxing.
        """
        # Read the input image using OpenCV
        if isinstance(input, str):
            with open(input, 'rb') as f:
                self.img = cv2.imdecode(
                    np.frombuffer(
                        f.read(),
                        np.uint8),
                    cv2.IMREAD_COLOR)
        else:
            self.img = input

        # self.img = cv2.imread(input)

        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        img, pad = self.letterbox(img, (self.input_height, self.input_width))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first
        # dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input
        # shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data, pad

    def postprocess(self, input_image, output, pad):
        """
        Perform post-processing on the model's output to extract and visualize detections.

        This method processes the raw model output to extract bounding boxes, scores, and class IDs.
        It applies non-maximum suppression to filter overlapping detections and draws the results on the input image.

        Args:
            input_image (np.ndarray): The input image.
            output (List[np.ndarray]): The output arrays from the model.
            pad (Tuple[int, int]): Padding values (top, left) used during letterboxing.

        Returns:
            (np.ndarray): The input image with detections drawn on it.
        """
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the
        # detections
        boxes = []
        boxesxyxys = []
        boxesxywhns = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        gain = min(
            self.input_height /
            self.img_height,
            self.input_width /
            self.img_width)

        # Some ONNX exports output normalized xywh (0..1), others output pixel
        # xywh (0..input_size).
        xywh = outputs[:, 0:4].copy()
        if np.nanmax(xywh) <= 1.5:
            xywh[:, 0] *= self.input_width
            xywh[:, 2] *= self.input_width
            xywh[:, 1] *= self.input_height
            xywh[:, 3] *= self.input_height

        # Undo letterbox padding (x/y are center coords in input space)
        xywh[:, 0] -= pad[1]
        xywh[:, 1] -= pad[0]

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = xywh[i][0], xywh[i][1], xywh[i][2], xywh[i][3]

                image_height, image_width = input_image.shape[:2]
                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) / gain)
                top = int((y - h / 2) / gain)
                width = int(w / gain)
                height = int(h / gain)

                # Clamp to image bounds
                left = max(0, min(left, image_width - 1))
                top = max(0, min(top, image_height - 1))
                right = max(0, min(left + width, image_width - 1))
                bottom = max(0, min(top + height, image_height - 1))
                width = max(1, right - left)
                height = max(1, bottom - top)

                x_center = left + width / 2
                y_center = top + height / 2

                # x_center_normalized = x_center
                # y_center_normalized = y_center
                # width_normalized = w
                # height_normalized = h

                x_center_normalized = x_center / image_width
                y_center_normalized = y_center / image_height
                width_normalized = width / image_width
                height_normalized = height / image_height

                # Add the class ID, score, and box coordinates to the
                # respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
                boxesxyxys.append([left, top, left + width, top + height])
                boxesxywhns.append(
                    [class_id, x_center_normalized, y_center_normalized, width_normalized, height_normalized])

        # Apply non-maximum suppression to filter hedao-250818 overlapping
        # bounding boxes
        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            self.confidence_thres,
            self.iou_thres)

        # Iterate over the selected indices after non-maximum suppression
        resultboxes = []
        resultscores = []
        resultclass_ids = []
        resultboxesxywhns = []

        for i in np.array(indices).flatten():
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            boxesxyxy = boxesxyxys[i]
            boxesxywhn = boxesxywhns[i]
            resultboxes.append(boxesxyxy)
            resultscores.append(score)
            resultclass_ids.append(class_id)
            resultboxesxywhns.append(boxesxywhn)

            # Draw the detection on the input image
            # self.draw_detections(input_image, box, score,
            # class_id,classes,color_palette)
            input_image = self.plot_one_box(x=boxesxyxy, im=input_image, color=self.color_palette[class_id],
                                            label=self.classes[class_id], score=score)

        # Return the modified input image
        return input_image, resultboxes, resultscores, resultclass_ids, resultboxesxywhns

    def loadonnx(self) -> np.ndarray:
        """
        Perform inference using an ONNX model and return the output image with drawn detections.

        Returns:
            (np.ndarray): The output image with drawn detections.
        """
        # Create an inference session using the ONNX model and specify
        # execution providers
        self.session = ort.InferenceSession(self.onnx_model,
                                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        self.model_outputs = self.session.get_outputs()

        # Get the model inputs
        self.model_inputs = self.session.get_inputs()

        model_meta = self.session.get_modelmeta()
        names_str = model_meta.custom_metadata_map['names']

        self.classes = list(eval(names_str).values())

        self.saveClassesTxt = os.path.join(
            os.path.dirname(self.onnx_model), 'classes.txt')

        self.color_palette = np.random.uniform(
            0, 255, size=(len(self.classes), 3))

        # Store the shape of the input for later use
        input_shape = self.model_inputs[0].shape
        try:
            self.input_height = int(input_shape[2])
            self.input_width = int(input_shape[3])
        except Exception:
            # Fallback for dynamic shapes
            self.input_height = 640
            self.input_width = 640
        print(self.input_width, self.input_height)

    def infer(self, input):
        # Preprocess the image data
        img_data, pad = self.preprocess(input)

        # Run inference using the preprocessed image data
        outputs = self.session.run(None, {self.model_inputs[0].name: img_data})

        return self.postprocess(self.img, outputs, pad)


class YOLOv8_Seg:
    def __init__(
            self,
            onnx_model: str,
            confidence_thres: float,
            iou_thres: float):
        self.onnx_model = onnx_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.vis_mode = "both"
        self.colors = Colors()
        self.session = ort.InferenceSession(self.onnx_model,
                                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.get_input_output_details()
        self.saveClassesTxt = os.path.join(
            os.path.dirname(self.onnx_model), 'classes.txt')
        self.model_outputs = self.session.get_outputs()

    def get_input_output_details(self):
        model_inputs = self.session.get_inputs()
        model_outputs = self.session.get_outputs()
        self.input_names = [inp.name for inp in model_inputs]
        self.output_names = [out.name for out in model_outputs]
        self.input_shape = model_inputs[0].shape
        try:
            self.input_height = int(self.input_shape[2])
            self.input_width = int(self.input_shape[3])
        except Exception:
            self.input_height = 640
            self.input_width = 640

        try:
            model_meta = self.session.get_modelmeta()
            names_str = model_meta.custom_metadata_map['names']
            self.classes = list(eval(names_str).values())
        except Exception:
            self.classes = []

        if not self.classes:
            self.nm = 32
            num_classes = model_outputs[0].shape[1] - self.nm - 4
            self.classes = [f"class_{i}" for i in range(max(1, num_classes))]

        self.nm = model_outputs[0].shape[1] - 4 - len(self.classes)
        if self.nm < 0:
            self.nm = 0
        self.color_palette = [self.colors(i, bgr=True)
                              for i in range(len(self.classes))]

    def prepare_input(self, image):
        if isinstance(image, str):
            with open(image, 'rb') as f:
                self.img = cv2.imdecode(
                    np.frombuffer(
                        f.read(),
                        np.uint8),
                    cv2.IMREAD_COLOR)
        else:
            self.img = image
        self.img_height, self.img_width = self.img.shape[:2]
        input_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(
            input_img, (self.input_width, self.input_height))
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def inference(self, input_tensor):
        outputs = self.session.run(
            self.output_names, {
                self.input_names[0]: input_tensor})
        return outputs

    def process_box_output(self, box_output):
        predictions = np.squeeze(box_output).T
        num_classes = len(self.classes)
        self.nm = box_output.shape[1] - 4 - num_classes
        if self.nm < 0:
            self.nm = 0

        scores = np.max(predictions[:, 4:4 + num_classes], axis=1)
        predictions = predictions[scores > self.confidence_thres, :]
        scores = scores[scores > self.confidence_thres]

        if len(scores) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        box_predictions = predictions[..., :num_classes + 4]
        mask_predictions = predictions[..., num_classes + 4:]

        class_ids = np.argmax(box_predictions[:, 4:], axis=1)
        boxes = self.extract_boxes(box_predictions)
        indices = nms(boxes, scores, self.iou_thres)

        return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]

    def extract_boxes(self, box_predictions):
        boxes = box_predictions[:, :4]
        boxes = self.rescale_boxes(boxes,
                                   (self.input_height, self.input_width),
                                   (self.img_height, self.img_width))
        boxes = xywh2xyxy(boxes)
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)
        return boxes

    def segment_objects(self, image):
        input_tensor = self.prepare_input(image)
        outputs = self.inference(input_tensor)
        self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(
            outputs[0])
        self.mask_maps = self.process_mask_output(mask_pred, outputs[1])
        return self.boxes, self.scores, self.class_ids, self.mask_maps

    def process_mask_output(self, mask_predictions, mask_output):
        if mask_predictions.shape[0] == 0:
            return np.zeros((0, self.img_height, self.img_width))

        mask_output = np.squeeze(mask_output)
        num_mask, mask_height, mask_width = mask_output.shape
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        scale_boxes = self.rescale_boxes(self.boxes,
                                         (self.img_height, self.img_width),
                                         (mask_height, mask_width))

        mask_maps = np.zeros(
            (len(scale_boxes), self.img_height, self.img_width))
        blur_size = (int(self.img_width / mask_width),
                     int(self.img_height / mask_height))

        for i, (box, scale_box) in enumerate(zip(self.boxes, scale_boxes)):
            scale_x1, scale_y1, scale_x2, scale_y2 = map(
                int, map(math.floor, scale_box))
            x1, y1, x2, y2 = map(int, map(math.floor, box))

            scale_y1, scale_y2 = max(0, scale_y1), min(mask_height, scale_y2)
            scale_x1, scale_x2 = max(0, scale_x1), min(mask_width, scale_x2)
            if scale_y2 <= scale_y1 or scale_x2 <= scale_x1:
                continue

            scale_crop_mask = masks[i, scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(
                scale_crop_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
            crop_mask = cv2.blur(crop_mask, blur_size)
            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps

    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        if boxes.size == 0:
            return boxes
        input_shape = np.array(
            [input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0],
                           image_shape[1], image_shape[0]])
        return boxes

    def warmup(self, num):
        dummy = np.zeros(
            (self.input_height,
             self.input_width,
             3),
            dtype=np.uint8) + 114
        input_img = cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB)
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        for _ in range(num):
            self.session.run(
                self.output_names, {
                    self.input_names[0]: input_tensor})

    def plot_one_box(
            self,
            x,
            im,
            color=(
                    128,
                    128,
                    128),
            label=None,
            score=None,
            line_thickness=3):

        # if label not in {"hight-risk", "low-risk",'birdnest'}:
        #    return im
        # 线宽（框 + 字体）
        input_is_pil = isinstance(im, Image.Image)
        lw = max(round(sum(im.size if input_is_pil else im.shape) /
                       2 * 0.0025), 5) or line_thickness

        tf = max(lw - 1, 1)  # font thickness
        sf = lw / 3  # font scale
        #
        # tl = round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1
        # tf = max(tl - 4, 1)  # 字体线宽

        # 坐标
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

        # 画边框
        cv2.rectangle(im, c1, c2, color, thickness=lw, lineType=cv2.LINE_AA)

        if label:
            # 构造文本内容
            text = label
            if score is not None:
                text += f" {score:.2f}"

            # 字体缩放
            fontScale = sf

            # 获取文本尺寸
            t_size = cv2.getTextSize(
                text, 0, fontScale=fontScale, thickness=tf)[0]
            text_width, text_height = t_size

            # 默认将文本画在框上方
            text_bottom_left = (c1[0], c1[1] - 2)
            rect_top_left = (c1[0], c1[1] - text_height - 4)
            rect_bottom_right = (c1[0] + text_width, c1[1])

            h, w = im.shape[:2]

            # ✅ 边界判断：如果顶部太靠近图像上边缘，就往下画
            if rect_top_left[1] < 0:
                text_bottom_left = (c1[0], c1[1] + text_height + 2)
                rect_top_left = (c1[0], c1[1])
                rect_bottom_right = (
                    c1[0] + text_width, c1[1] + text_height + 4)
                if rect_bottom_right[1] > h:
                    dy = rect_bottom_right[1] - h + 2  # 多留2像素边距
                    rect_top_left = (rect_top_left[0], rect_top_left[1] - dy)
                    rect_bottom_right = (
                        rect_bottom_right[0], rect_bottom_right[1] - dy)
                    text_bottom_left = (
                        text_bottom_left[0], text_bottom_left[1] - dy)

            # ✅ 左右越界修正（放在上下修正之后）
            if rect_bottom_right[0] > w:
                dx = rect_bottom_right[0] - w + 2  # 多留2像素边距
                rect_top_left = (rect_top_left[0] - dx, rect_top_left[1])
                rect_bottom_right = (
                    rect_bottom_right[0] - dx,
                    rect_bottom_right[1])
                text_bottom_left = (
                    text_bottom_left[0] - dx,
                    text_bottom_left[1])

            cv2.rectangle(im, rect_top_left, rect_bottom_right,
                          color, -1, cv2.LINE_AA)

            cv2.putText(im, text, text_bottom_left, 0, fontScale,
                        (225, 255, 255), thickness=tf, lineType=cv2.LINE_AA)
        return im

    def draw_masks(
            self,
            image,
            boxes,
            class_ids,
            mask_alpha=0.3,
            mask_maps=None):
        mask_img = image.copy()
        for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
            color = self.colors(class_id)
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            if mask_maps is None:
                cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
            else:
                crop_mask = mask_maps[i][y1:y2, x1:x2, np.newaxis]
                crop_mask_img = mask_img[y1:y2, x1:x2]
                crop_mask_img = crop_mask_img * \
                                (1 - crop_mask) + crop_mask * color
                mask_img[y1:y2, x1:x2] = crop_mask_img
        return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)

    def infer(self, input, draw=True):
        boxes, scores, class_ids, mask_maps = self.segment_objects(input)

        resultboxes = []
        resultscores = []
        resultclass_ids = []
        resultboxesxywhns = []

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].tolist()
            score = float(scores[i])
            cls_id = int(class_ids[i])

            resultboxes.append([x1, y1, x2, y2])
            resultscores.append(score)
            resultclass_ids.append(cls_id)

            x_center = (x1 + x2) / 2 / self.img_width
            y_center = (y1 + y2) / 2 / self.img_height
            width = (x2 - x1) / self.img_width
            height = (y2 - y1) / self.img_height
            resultboxesxywhns.append(
                [cls_id, x_center, y_center, width, height])

        if draw:
            input_image = self.img.copy()
            if self.vis_mode in ("mask", "both") and len(mask_maps) > 0:
                try:
                    input_image = self.draw_masks(
                        input_image, boxes, class_ids, 0.4, mask_maps)
                except Exception as e:
                    print(f"Error occurred while drawing masks: {e}")
            for i in range(len(resultboxes)):
                cls_id = int(class_ids[i])
                sc = float(scores[i])
                x1, y1, x2, y2 = boxes[i].tolist()
                if self.vis_mode in ("box", "both"):
                    input_image = self.plot_one_box(
                        x=[x1, y1, x2, y2], im=input_image,
                        color=self.color_palette[cls_id] if cls_id < len(
                            self.color_palette) else (0, 255, 0),
                        label=self.classes[cls_id] if cls_id < len(
                            self.classes) else str(cls_id),
                        score=sc)
                # cv2.imwrite("output.jpg",input_image)

        else:
            input_image = self.img

        return input_image, resultboxes, resultscores, resultclass_ids, resultboxesxywhns, mask_maps

    @staticmethod
    def _masks_to_polygons(mask_maps, class_ids, img_w, img_h):
        polygons = []
        for i in range(mask_maps.shape[0]):
            mask_bin = mask_maps[i]
            if mask_bin.sum() < 10:
                continue
            cid = class_ids[i] if i < len(class_ids) else 0
            if mask_bin.dtype != np.uint8:
                mask_bin = (mask_bin * 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_points = []
            for contour in contours:
                if len(contour) < 3:
                    continue
                epsilon = 0.002 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                for pt in approx:
                    xn = float(pt[0][0]) / img_w
                    yn = float(pt[0][1]) / img_h
                    all_points.extend([xn, yn])
            if all_points:
                polygons.append([cid, all_points])
        return polygons


class WindowMixin(object):

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            add_actions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName(u'%sToolBar' % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            add_actions(toolbar, actions)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        return toolbar


class ResizeDialog(QDialog):
    def __init__(self, parent=None, current_dir=""):
        super().__init__(parent)
        self.setWindowTitle("批量图像缩放")
        self.setMinimumWidth(400)

        layout = QVBoxLayout()

        # 模式选择
        mode_box = QGroupBox("缩放模式")
        mode_layout = QVBoxLayout()
        self.scale_radio = QRadioButton("按比例缩放")
        self.wh_radio = QRadioButton("按宽高缩放")
        self.scale_radio.setChecked(True)
        mode_layout.addWidget(self.scale_radio)
        mode_layout.addWidget(self.wh_radio)
        mode_box.setLayout(mode_layout)
        layout.addWidget(mode_box)

        # 比例模式
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("缩放比例:"))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.01, 10.0)
        self.scale_spin.setValue(0.5)
        self.scale_spin.setDecimals(2)
        self.scale_spin.setSingleStep(0.1)
        scale_row.addWidget(self.scale_spin)
        layout.addLayout(scale_row)

        # 宽高模式
        wh_row = QHBoxLayout()
        wh_row.addWidget(QLabel("宽:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(0, 20000)
        self.width_spin.setValue(0)
        self.width_spin.setSuffix(" px (0=不限)")
        wh_row.addWidget(self.width_spin)
        wh_row.addWidget(QLabel("高:"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(0, 20000)
        self.height_spin.setValue(0)
        self.height_spin.setSuffix(" px (0=不限)")
        wh_row.addWidget(self.height_spin)
        layout.addLayout(wh_row)
        self.width_spin.setEnabled(False)
        self.height_spin.setEnabled(False)

        self.keep_ratio_chk = QCheckBox("保持比例（改宽自动算高）")
        self.keep_ratio_chk.setChecked(True)
        self.keep_ratio_chk.setEnabled(False)
        layout.addWidget(self.keep_ratio_chk)

        # 选项
        self.backup_chk = QCheckBox("备份原图到 _backup_ 子目录")
        self.backup_chk.setChecked(False)
        layout.addWidget(self.backup_chk)

        # 目录
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("处理目录:"))
        self.dir_edit = QLineEdit(current_dir)
        dir_row.addWidget(self.dir_edit, 1)
        dir_btn = QPushButton("浏览")
        dir_btn.clicked.connect(self._browse_dir)
        dir_row.addWidget(dir_btn)
        layout.addLayout(dir_row)

        # 按钮
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        start_btn = QPushButton("开始处理")
        start_btn.clicked.connect(self.accept)
        btn_row.addWidget(start_btn)
        layout.addLayout(btn_row)

        self.setLayout(layout)

        self.scale_radio.toggled.connect(self._on_mode_changed)
        self.width_spin.valueChanged.connect(self._on_width_changed)

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "选择图片目录", self.dir_edit.text())
        if d:
            self.dir_edit.setText(d)

    def _on_mode_changed(self):
        is_scale = self.scale_radio.isChecked()
        self.scale_spin.setEnabled(is_scale)
        self.width_spin.setEnabled(not is_scale)
        self.height_spin.setEnabled(not is_scale)
        self.keep_ratio_chk.setEnabled(not is_scale)

    def _on_width_changed(self):
        if self.wh_radio.isChecked() and self.keep_ratio_chk.isChecked():
            # 无法自动获取比例，需要外部设置；此处仅占位
            pass

    def should_backup(self):
        return self.backup_chk.isChecked()

    def get_params(self):
        if self.scale_radio.isChecked():
            return {"mode": "scale", "scale": self.scale_spin.value()}
        return {
            "mode": "wh",
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "keep_ratio": self.keep_ratio_chk.isChecked(),
        }

    def get_dir(self):
        return self.dir_edit.text()


class LabelMergeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("标签追加合并")
        self.setMinimumSize(700, 500)
        self._src_classes = []
        self._tgt_classes = []
        self._widgets = []  # (check, target_id_edit) per source class

        layout = QVBoxLayout()

        # 路径输入行
        path_grid = QGridLayout()
        path_grid.addWidget(QLabel("源目录(追加数据):"), 0, 0)
        self.src_dir_edit = QLineEdit()
        path_grid.addWidget(self.src_dir_edit, 0, 1)
        src_btn = QPushButton("浏览")
        src_btn.clicked.connect(self._browse_src)
        path_grid.addWidget(src_btn, 0, 2)

        path_grid.addWidget(QLabel("目标目录(被追加):"), 1, 0)
        self.tgt_dir_edit = QLineEdit()
        path_grid.addWidget(self.tgt_dir_edit, 1, 1)
        tgt_btn = QPushButton("浏览")
        tgt_btn.clicked.connect(self._browse_tgt)
        path_grid.addWidget(tgt_btn, 1, 2)
        layout.addLayout(path_grid)

        # 左右类别列表
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # 左侧：源类别（可勾选 + 目标ID）
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("源类别（勾选要追加的）:"))
        self.src_scroll = QScrollArea()
        self.src_scroll.setWidgetResizable(True)
        self.src_container = QWidget()
        self.src_form = QFormLayout()
        self.src_container.setLayout(self.src_form)
        self.src_scroll.setWidget(self.src_container)
        left_layout.addWidget(self.src_scroll, 1)
        left_widget.setLayout(left_layout)
        splitter.addWidget(left_widget)

        # 右侧：目标类别（只读参考）
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QLabel("目标类别（参考）:"))
        self.tgt_list = QListWidget()
        right_layout.addWidget(self.tgt_list, 1)
        right_widget.setLayout(right_layout)
        splitter.addWidget(right_widget)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        # 底部选项
        self.force_append_chk = QCheckBox("强制追加（跳过 IoU 检查，所有框直接追加）")
        layout.addWidget(self.force_append_chk)

        iou_row = QHBoxLayout()
        iou_row.addWidget(QLabel("IoU 阈值:"))
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.1, 1.0)
        self.iou_spin.setValue(0.5)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setSingleStep(0.05)
        iou_row.addWidget(self.iou_spin)
        iou_row.addWidget(QLabel("(非强制追加模式生效)"))
        iou_row.addStretch()
        layout.addLayout(iou_row)

        # 全选/反选
        sel_row = QHBoxLayout()
        select_all_btn = QPushButton("全选")
        select_all_btn.clicked.connect(lambda: self._set_all_checks(True))
        sel_row.addWidget(select_all_btn)
        select_none_btn = QPushButton("全不选")
        select_none_btn.clicked.connect(lambda: self._set_all_checks(False))
        sel_row.addWidget(select_none_btn)
        invert_btn = QPushButton("反选")
        invert_btn.clicked.connect(self._invert_checks)
        sel_row.addWidget(invert_btn)
        sel_row.addStretch()
        layout.addLayout(sel_row)

        # 按钮
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        start_btn = QPushButton("开始合并")
        start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(start_btn)
        layout.addLayout(btn_row)

        self.setLayout(layout)

    def _browse_src(self):
        d = QFileDialog.getExistingDirectory(
            self, "选择源目录", self.src_dir_edit.text())
        if d:
            self.src_dir_edit.setText(d)
            self._load_src_classes(d)

    def _browse_tgt(self):
        d = QFileDialog.getExistingDirectory(
            self, "选择目标目录", self.tgt_dir_edit.text())
        if d:
            self.tgt_dir_edit.setText(d)
            self._load_tgt_classes(d)

    def _read_classes(self, dir_path):
        classes_path = os.path.join(dir_path, "classes.txt")
        if not os.path.exists(classes_path):
            return []
        classes = []
        try:
            with open(classes_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        classes.append(s)
        except Exception:
            pass
        return classes

    def _load_src_classes(self, dir_path):
        self._src_classes = self._read_classes(dir_path)
        self._rebuild_src_list()

    def _load_tgt_classes(self, dir_path):
        self._tgt_classes = self._read_classes(dir_path)
        self.tgt_list.clear()
        for i, name in enumerate(self._tgt_classes):
            self.tgt_list.addItem(f"{i}  {name}")

    def _rebuild_src_list(self):
        # Clear old form
        while self.src_form.rowCount() > 0:
            self.src_form.removeRow(0)
        self._widgets = []

        if not self._src_classes:
            self.src_form.addRow(QLabel("（未找到 classes.txt）"))
            return

        for i, name in enumerate(self._src_classes):
            row = QHBoxLayout()
            row.setSpacing(6)
            chk = QCheckBox()
            id_label = QLabel(str(i))
            id_label.setFixedWidth(24)
            id_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            name_label = QLabel(name)
            name_label.setMinimumWidth(80)
            arrow = QLabel("→ 目标ID:")
            target_edit = QLineEdit(str(i))
            target_edit.setFixedWidth(50)
            target_edit.setPlaceholderText("ID")

            row.addWidget(chk)
            row.addWidget(id_label)
            row.addWidget(name_label)
            row.addWidget(arrow)
            row.addWidget(target_edit)
            row.addStretch()
            container = QWidget()
            container.setLayout(row)
            self.src_form.addRow(container)
            self._widgets.append((chk, target_edit))

    def _set_all_checks(self, checked):
        for chk, _ in self._widgets:
            chk.setChecked(checked)

    def _invert_checks(self):
        for chk, _ in self._widgets:
            chk.setChecked(not chk.isChecked())

    def get_src_dir(self):
        return self.src_dir_edit.text()

    def get_tgt_dir(self):
        return self.tgt_dir_edit.text()

    def get_src_classes(self):
        """Return list of source class names."""
        return self._src_classes

    def get_mapping(self):
        """Return {src_class_id: tgt_class_id} for checked classes only."""
        mapping = {}
        for i, (chk, edit) in enumerate(self._widgets):
            if chk.isChecked():
                try:
                    tgt_id = int(edit.text())
                except ValueError:
                    tgt_id = i
                mapping[i] = tgt_id
        return mapping

    def is_force_append(self):
        return self.force_append_chk.isChecked()

    def get_iou_threshold(self):
        return self.iou_spin.value()

    def _on_start(self):
        if not self.get_src_dir() or not self.get_tgt_dir():
            QMessageBox.warning(self, "路径未设置", "请先选择源目录和目标目录")
            return
        if not self.get_mapping():
            QMessageBox.warning(self, "未勾选类别", "请至少勾选一个要合并的类别")
            return
        self.accept()


class DetectionAppendDialog(QDialog):
    """检测结果追加到标签的对话框 — 支持修改目标类别ID和批量应用"""

    def __init__(self, parent=None, detected_classes=None):
        super().__init__(parent)
        self.setWindowTitle("检测结果追加到标签")
        self.setMinimumWidth(580)

        self._detected_classes = detected_classes or []
        self._widgets = []  # (chk, target_edit, original_cid)

        layout = QVBoxLayout()

        layout.addWidget(QLabel("勾选要追加的检测类别，可修改目标类别ID:"))

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.container = QWidget()
        self.form = QFormLayout()
        self.form.setContentsMargins(0, 0, 0, 0)
        self.container.setLayout(self.form)
        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll, 1)

        for cid, name, count in self._detected_classes:
            row = QHBoxLayout()
            row.setSpacing(6)
            chk = QCheckBox()
            chk.setChecked(False)
            src_info = QLabel(f"ID:{cid}  {name}  ({count}个框)")
            src_info.setMinimumWidth(160)
            arrow = QLabel("→ 目标ID:")
            target_edit = QLineEdit(str(cid))
            target_edit.setFixedWidth(50)
            # 类别名输入框 — 优先从数据集 classes.txt 查找；不存在则用检测类别名（源名）
            default_name = self._lookup_target_name(str(cid))
            if default_name == str(cid) or default_name == "?":
                default_name = name if name else default_name
            tgt_name_edit = QLineEdit(default_name)
            tgt_name_edit.setFixedWidth(120)
            tgt_name_edit.setPlaceholderText("输入类别名")
            # 目标ID改变时，尝试从数据集 classes.txt 查找对应名称
            target_edit.textChanged.connect(
                lambda text, edit=tgt_name_edit, src_name=name:
                edit.setText(self._resolve_target_name(text, src_name))
            )

            row.addWidget(chk)
            row.addWidget(src_info)
            row.addWidget(arrow)
            row.addWidget(target_edit)
            row.addWidget(tgt_name_edit)
            row.addStretch()
            cw = QWidget()
            cw.setLayout(row)
            self.form.addRow(cw)
            self._widgets.append((chk, target_edit, tgt_name_edit, cid))

        # 全选/全不选/反选
        sel_row = QHBoxLayout()
        all_btn = QPushButton("全选")
        all_btn.clicked.connect(lambda: self._set_all_checks(True))
        sel_row.addWidget(all_btn)
        none_btn = QPushButton("全不选")
        none_btn.clicked.connect(lambda: self._set_all_checks(False))
        sel_row.addWidget(none_btn)
        invert_btn = QPushButton("反选")
        invert_btn.clicked.connect(self._invert_checks)
        sel_row.addWidget(invert_btn)
        sel_row.addStretch()
        layout.addLayout(sel_row)

        # 批量应用
        self.batch_chk = QCheckBox("应用到所有已检测图片（批量追加）")
        self.batch_chk.setChecked(True)
        layout.addWidget(self.batch_chk)

        # IoU 阈值
        iou_row = QHBoxLayout()
        iou_row.addWidget(QLabel("IoU 阈值:"))
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.1, 1.0)
        self.iou_spin.setValue(0.5)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setSingleStep(0.05)
        iou_row.addWidget(self.iou_spin)
        iou_row.addStretch()
        layout.addLayout(iou_row)

        # 按钮
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        ok_btn = QPushButton("追加")
        ok_btn.clicked.connect(self._on_ok)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

        self.setLayout(layout)

    def _lookup_target_name(self, class_id_str):
        """根据目标类别ID查找类别名称 — 优先当前标注数据集 classes.txt，其次模型"""
        try:
            cid = int(class_id_str)
        except ValueError:
            return "?"
        p = self.parent()
        if not p:
            return str(cid)
        # 1) 优先：已加载的标注数据集类别
        if hasattr(p, 'label_hist') and p.label_hist:
            try:
                if 0 <= cid < len(p.label_hist) and p.label_hist[cid]:
                    return str(p.label_hist[cid])
            except Exception:
                pass
        # 2) 其次：从 txt_path 直接读 classes.txt
        if hasattr(
                p, 'txt_path') and p.txt_path and os.path.exists(
            p.txt_path):
            try:
                with open(p.txt_path, "r", encoding="utf-8", errors="ignore") as f:
                    names = [line.strip() for line in f if line.strip()]
                if 0 <= cid < len(names) and names[cid]:
                    return names[cid]
            except Exception:
                pass
        # 3) 再次：模型内置类别
        if hasattr(
                p,
                'yolo_model') and p.yolo_model and hasattr(
            p.yolo_model,
            'classes'):
            try:
                return str(p.yolo_model.classes[cid])
            except (IndexError, Exception):
                pass
        # 4) 兜底：数字字符串
        return str(cid)

    def _resolve_target_name(self, class_id_str, src_name):
        """目标ID对应的名称；数据集无记录时回退为源类别名（检测名称）"""
        looked = self._lookup_target_name(class_id_str)
        if looked == class_id_str or looked == "?":
            return src_name if src_name else looked
        return looked

    def _set_all_checks(self, checked):
        for chk, _, _, _ in self._widgets:
            chk.setChecked(checked)

    def _invert_checks(self):
        for chk, _, _, _ in self._widgets:
            chk.setChecked(not chk.isChecked())

    def _on_ok(self):
        if not any(chk.isChecked() for chk, _, _, _ in self._widgets):
            QMessageBox.warning(self, "未勾选", "请至少勾选一个类别")
            return
        self.accept()

    def get_class_mapping(self):
        """返回 {原始类别ID: 目标类别ID} 映射，仅包含勾选的类别"""
        mapping = {}
        for chk, target_edit, _, original_cid in self._widgets:
            if chk.isChecked():
                try:
                    tgt_id = int(target_edit.text())
                except ValueError:
                    tgt_id = original_cid
                mapping[original_cid] = tgt_id
        return mapping

    def get_class_names(self):
        """返回 {目标类别ID: 用户输入的目标类别名}"""
        names = {}
        for chk, target_edit, tgt_name_edit, _ in self._widgets:
            if chk.isChecked():
                try:
                    tgt_id = int(target_edit.text())
                except ValueError:
                    continue
                names[tgt_id] = tgt_name_edit.text().strip() or str(tgt_id)
        return names

    def get_selected_classes(self):
        """返回勾选的原始类别ID集合（向后兼容）"""
        return {cid for chk, _, _, cid in self._widgets if chk.isChecked()}

    def is_batch_apply(self):
        return self.batch_chk.isChecked()

    def get_iou_threshold(self):
        return self.iou_spin.value()


class BatchModifyWindow(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.setWindowTitle("批量修改 / 删除数据集标签")
        self.setMinimumWidth(420)
        self.main = main_window
        self.classes = self.load_classes()
        self.class_counts = self.count_classes()  # 统计各类别数量
        self.widgets = []
        self._refresh_main_window_classes()

        layout = QVBoxLayout()
        group = QGroupBox("选择要操作的类别")
        form_layout = QFormLayout()

        for cls_id in range(len(self.classes)):
            row = QHBoxLayout()

            # ✅ 统一间距（关键）
            row.setSpacing(12)
            row.setContentsMargins(5, 2, 5, 2)

            check = QCheckBox()

            id_label = QLabel(str(cls_id))
            id_label.setFixedWidth(40)
            id_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            name_label = QLabel(self.classes[cls_id])
            name_label.setMinimumWidth(120)
            name_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

            # ✅ 显示该类别的标注数量
            count_label = QLabel(f"({self.class_counts.get(cls_id, 0)})")
            count_label.setFixedWidth(50)
            count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            count_label.setStyleSheet("color: #000; font-size: 11px;")

            arrow = QLabel("→")
            arrow.setAlignment(Qt.AlignCenter)

            edit = QLineEdit()
            edit.setPlaceholderText("新ID")
            edit.setFixedWidth(60)

            # ✅ 布局顺序（关键点）
            row.addWidget(check)
            row.addSpacing(10)

            row.addWidget(id_label)
            row.addSpacing(5)

            row.addWidget(name_label)
            row.addWidget(count_label)  # 数量显示在类别名后面

            # ✅ 拉伸，让右边输入框靠右（关键）
            row.addStretch()

            row.addWidget(arrow)
            row.addSpacing(5)
            row.addWidget(edit)

            form_layout.addRow(row)

            self.widgets.append((check, edit))

        group.setLayout(form_layout)
        layout.addWidget(group)

        # =========================
        # 全选/全不选按钮
        # =========================
        select_all_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("全选")
        self.select_none_btn = QPushButton("全不选")
        self.invert_select_btn = QPushButton("反选")
        select_all_layout.addWidget(self.select_all_btn)
        select_all_layout.addWidget(self.select_none_btn)
        select_all_layout.addWidget(self.invert_select_btn)
        select_all_layout.addStretch(1)
        layout.addLayout(select_all_layout)

        btn_layout = QHBoxLayout()
        self.modify_btn = QPushButton("修改类别")
        self.delete_btn = QPushButton("删除类别")
        self.modify_classes_checkbox = QCheckBox("同步修改 classes.txt（类别名称/顺序）")
        self.modify_classes_checkbox.setChecked(True)  # 默认开启
        form_layout.addRow(self.modify_classes_checkbox)
        btn_layout.addWidget(self.modify_btn)
        btn_layout.addWidget(self.delete_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

        # 连接按钮信号
        self.modify_btn.clicked.connect(self.on_modify)
        self.delete_btn.clicked.connect(self.on_delete)
        self.select_all_btn.clicked.connect(self.select_all)
        self.select_none_btn.clicked.connect(self.select_none)
        self.invert_select_btn.clicked.connect(self.invert_select)

    def _refresh_main_window_classes(self):
        """
        刷新主窗口的类别相关UI，确保画框选择标签时类别列表是最新的
        """
        # 重新加载 classes.txt 到主窗口
        if self.main.default_save_dir:
            label_dir = self.main.default_save_dir
            classes_path = os.path.join(label_dir, "classes.txt")
            if os.path.exists(classes_path):
                # 清空并重新加载类别
                self.main.label_hist = []
                self.main.load_predefined_classes(classes_path)
                self.main._dataset_classes_from_file = list(
                    self.main.label_hist)

        # 更新类别筛选下拉框
        self.main._update_class_filter_items()

        # 更新标签选择下拉框（画框时用的那个）
        self.main.update_combo_box()

        # 如果当前有打开的图片，刷新标签列表显示
        if self.main.file_path:
            self.main.label_list.clear()
            self.main.items_to_shapes.clear()
            self.main.shapes_to_items.clear()

            # 重新加载当前标注
            self.main.show_bounding_box_from_annotation_file(
                self.main.file_path)

            # 更新画布的标签显示
            for shape in self.main.canvas.shapes:
                shape.paint_label = self.main.display_label_option.isChecked()

            self.main.canvas.update()

    def load_classes(self):
        try:
            with open(os.path.join(self.main.default_save_dir, "classes.txt"), "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        except:
            return []

    def count_classes(self):
        """
        统计当前目录下所有标注文件（.txt / .xml / .json）中各类别出现次数
        返回: {class_id: count}
        """
        import glob
        from xml.etree import ElementTree

        label_dir = self.main.default_save_dir
        if not label_dir:
            return {}

        counts = {}

        # --- YOLO .txt 标注 ---
        for txt_file in glob.glob(os.path.join(label_dir, "*.txt")):
            if os.path.basename(txt_file) == "classes.txt":
                continue
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) < 2:  # 至少 class_id + 1个坐标
                            continue
                        try:
                            class_id = int(parts[0])
                            counts[class_id] = counts.get(class_id, 0) + 1
                        except ValueError:
                            continue
            except Exception:
                continue

        # --- Pascal VOC .xml 标注 ---
        for xml_file in glob.glob(os.path.join(label_dir, "*.xml")):
            try:
                tree = ElementTree.parse(xml_file)
                root = tree.getroot()
                for obj in root.findall("object"):
                    name_el = obj.find("name")
                    if name_el is not None and name_el.text:
                        name = name_el.text.strip()
                        if name in self.main.label_hist:
                            cid = self.main.label_hist.index(name)
                        elif name.isdigit():
                            cid = int(name)
                        else:
                            continue
                        counts[cid] = counts.get(cid, 0) + 1
            except Exception:
                continue

        # --- JSON 标注 (LabelMe 格式) ---
        for json_file in glob.glob(os.path.join(label_dir, "*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for shape in data.get("shapes", []):
                    label = shape.get("label", "")
                    if not label:
                        continue
                    if label in self.main.label_hist:
                        cid = self.main.label_hist.index(label)
                    elif label.isdigit():
                        cid = int(label)
                    else:
                        continue
                    counts[cid] = counts.get(cid, 0) + 1
            except Exception:
                continue

        return counts

    def select_all(self):
        """全选所有复选框"""
        for check, _edit in self.widgets:
            check.setChecked(True)

    def select_none(self):
        """全不选"""
        for check, _edit in self.widgets:
            check.setChecked(False)

    def invert_select(self):
        """反选"""
        for check, _edit in self.widgets:
            check.setChecked(not check.isChecked())

    def get_mapping(self):
        mapping = {}
        for idx, (check, edit) in enumerate(self.widgets):
            if check.isChecked():
                txt = edit.text().strip()
                if txt.isdigit():
                    mapping[idx] = int(txt)
        return mapping

    def get_delete_ids(self):
        delete_ids = []
        for idx, (check, edit) in enumerate(self.widgets):
            if check.isChecked():
                delete_ids.append(idx)
        return delete_ids

    def on_modify(self):
        mapping = self.get_mapping()
        modify_classes = self.modify_classes_checkbox.isChecked()
        if mapping:
            self.main.safe_batch_modify(mapping, modify_classes)
        self.load_classes()
        self._refresh_main_window_classes()
        self.accept()

    def on_delete(self):
        ids = self.get_delete_ids()
        modify_classes = self.modify_classes_checkbox.isChecked()
        if ids:
            self.main.batch_delete_labels(ids, modify_classes)
        self.load_classes()
        self._refresh_main_window_classes()
        self.accept()


class MainWindow(QMainWindow, WindowMixin):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))

    def __init__(
            self,
            default_filename=None,
            default_prefdef_class_file=None,
            default_save_dir=None):
        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)

        # Load setting in the main thread
        self.settings = Settings()
        self.settings.load()
        settings = self.settings

        self.os_name = platform.system()

        # Load string bundle for i18n
        self.string_bundle = StringBundle.get_bundle()
        get_str = lambda str_id: self.string_bundle.get_string(str_id)

        # Save as Pascal voc xml
        self.default_save_dir = default_save_dir
        self.label_file_format = settings.get(
            SETTING_LABEL_FILE_FORMAT, LabelFileFormat.PASCAL_VOC)

        # For loading all image under a directory
        self.pre_img_txt = []
        self.pre_img_seg = []
        self.pre_error_img_txt = []
        self.m_img_list = []
        self.m_img_list_all = []
        self.dir_name = None
        self.label_hist = []
        self.last_open_dir = None
        self.cur_img_idx = 0
        self.img_count = 1
        self._selected_class_filter = None
        self._selected_class_filter_set = None
        self._only_show_selected_class_labels = False
        self._label_uncheck_filter_set = None
        self._image_label_cache = {}
        self._dataset_has_yolo_annotations = False
        self._dataset_has_classes_txt = False
        self._dataset_uses_numeric_labels = False
        self._dataset_can_generate_classes_txt = True
        self._dataset_classes_from_file = []
        self._dataset_numeric_label_ids = set()
        self._asked_generate_classes_txt = False
        self._last_filter_cancelled = False
        self._class_link_groups = []

        # Whether we need to save or not.
        self.dirty = False

        self._no_selection_slot = False
        self._beginner = True
        self.screencast = "https://youtu.be/p0nR2YsCY_U"

        # Batch detect from current image setting
        self.batch_detect_from_current = True

        # Load predefined classes to the list
        self.load_predefined_classes(default_prefdef_class_file)

        # Main widgets and related state.
        self.label_dialog = LabelDialog(parent=self, list_item=self.label_hist)

        self.items_to_shapes = {}
        self.shapes_to_items = {}
        self.prev_label_text = ''

        list_layout = QVBoxLayout()
        list_layout.setContentsMargins(0, 0, 0, 0)

        # Create a widget for using default label
        self.use_default_label_checkbox = QCheckBox(get_str('useDefaultLabel'))
        self.use_default_label_checkbox.setChecked(False)
        self.default_label_text_line = QLineEdit()
        use_default_label_qhbox_layout = QHBoxLayout()
        use_default_label_qhbox_layout.addWidget(
            self.use_default_label_checkbox)
        use_default_label_qhbox_layout.addWidget(self.default_label_text_line)
        use_default_label_container = QWidget()
        use_default_label_container.setLayout(use_default_label_qhbox_layout)

        # Create a widget for edit and diffc button
        self.diffc_button = QCheckBox(get_str('useDifficult'))
        self.diffc_button.setChecked(False)
        self.diffc_button.stateChanged.connect(self.button_state)
        self.edit_button = QToolButton()
        self.edit_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        # Add some of widgets to list_layout
        list_layout.addWidget(self.edit_button)
        list_layout.addWidget(self.diffc_button)
        list_layout.addWidget(use_default_label_container)

        # Create and add combobox for showing unique labels in group
        self.combo_box = ComboBox(self)
        list_layout.addWidget(self.combo_box)

        # Create and add a widget for showing current label items
        self.label_list = QListWidget()
        label_list_container = QWidget()
        label_list_container.setLayout(list_layout)
        self.label_list.itemActivated.connect(self.label_selection_changed)
        self.label_list.itemSelectionChanged.connect(
            self.label_selection_changed)
        self.label_list.itemDoubleClicked.connect(self.edit_label)
        # Connect to itemChanged to detect checkbox changes.
        self.label_list.itemChanged.connect(self.label_item_changed)
        list_layout.addWidget(self.label_list)

        self.dock = QDockWidget(get_str('boxLabelText'), self)
        self.dock.setObjectName(get_str('labels'))
        self.dock.setWidget(label_list_container)

        self.file_list_widget = QListWidget()
        self.file_list_widget.itemDoubleClicked.connect(
            self.file_item_double_clicked)
        self.file_list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_list_widget.customContextMenuRequested.connect(
            self._on_file_list_context_menu)
        file_list_layout = QVBoxLayout()
        file_list_layout.setContentsMargins(0, 0, 0, 0)

        # Small utility button (not in main toolbar): write model classes to
        # dataset classes.txt
        self.sync_model_classes_button = QToolButton()
        self.sync_model_classes_button.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon)
        self.sync_model_classes_button.setAutoRaise(False)
        self.sync_model_classes_button.setText("classes")
        self.sync_model_classes_button.setToolTip(
            "使用已加载模型的类别列表写入/覆盖当前文件夹的 classes.txt")
        self.sync_model_classes_button.setEnabled(False)
        self.sync_model_classes_button.setStyleSheet(
            "QToolButton { background-color: #f0f0f0; color: #333; border: 1px solid #d0d0d0; border-radius: 4px; padding: 4px 8px; }"
            "QToolButton:pressed { background-color: #e0e0e0; }"
            "QToolButton:disabled { background-color: #f8f8f8; color: #999; border: 1px solid #e8e8e8; }"
        )
        self.sync_model_classes_button.clicked.connect(
            self.replace_classes_txt_with_model)

        # 批量修改 / 删除标签按钮（和上面按钮完全同款样式）
        self.batch_modify_btn = QToolButton()
        self.batch_modify_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.batch_modify_btn.setAutoRaise(False)
        self.batch_modify_btn.setText("标签处理")
        self.batch_modify_btn.setToolTip("批量修改/删除/筛选标签类别")
        self.batch_modify_btn.setEnabled(True)
        self.batch_modify_btn.setStyleSheet(
            "QToolButton { background-color: #f0f0f0; color: #333; border: 1px solid #d0d0d0; border-radius: 4px; padding: 4px 8px; }"
            "QToolButton:pressed { background-color: #e0e0e0; }"
            "QToolButton:disabled { background-color: #f8f8f0; color: #999; border: 1px solid #e8e8e8; }"
        )
        self.batch_modify_btn.clicked.connect(self.open_batch_modify_window)

        # ????
        self.stats_btn = QToolButton()
        self.stats_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.stats_btn.setAutoRaise(False)
        self.stats_btn.setText("\u7edf\u8ba1")
        self.stats_btn.setToolTip(
            "\u7edf\u8ba1\u5f53\u524d\u6570\u636e\u96c6\u5404\u7c7b\u522b\u76ee\u6807\u6570\u91cf")
        self.stats_btn.setEnabled(True)
        self.stats_btn.setStyleSheet(
            "QToolButton { background-color: #f0f0f0; color: #333; border: 1px solid #d0d0d0; border-radius: 4px; padding: 4px 8px; }"
            "QToolButton:pressed { background-color: #e0e0e0; }"
        )
        self.stats_btn.clicked.connect(self._show_dataset_stats)

        sync_btn_layout = QHBoxLayout()
        sync_btn_layout.setContentsMargins(5, 2, 5, 2)
        sync_btn_layout.addWidget(self.sync_model_classes_button)
        sync_btn_layout.addWidget(self.batch_modify_btn)
        sync_btn_layout.addWidget(self.stats_btn)

        # 图像缩放按钮
        self.resize_btn = QToolButton()
        self.resize_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.resize_btn.setAutoRaise(False)
        self.resize_btn.setText("图像缩放")
        self.resize_btn.setToolTip("批量缩放当前目录所有图片")
        self.resize_btn.setEnabled(True)
        self.resize_btn.setStyleSheet(
            "QToolButton { background-color: #f0f0f0; color: #333; border: 1px solid #d0d0d0; border-radius: 4px; padding: 4px 8px; }"
            "QToolButton:pressed { background-color: #e0e0e0; }"
            "QToolButton:disabled { background-color: #f8f8f0; color: #999; border: 1px solid #e8e8e8; }"
        )
        self.resize_btn.clicked.connect(self.resize_images)

        # 标签追加合并按钮
        self.label_merge_btn = QToolButton()
        self.label_merge_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.label_merge_btn.setAutoRaise(False)
        self.label_merge_btn.setText("标签追加合并")
        self.label_merge_btn.setToolTip("将源目录选中类别的标签追加合并到目标目录")
        self.label_merge_btn.setEnabled(True)
        self.label_merge_btn.setStyleSheet(
            "QToolButton { background-color: #f0f0f0; color: #333; border: 1px solid #d0d0d0; border-radius: 4px; padding: 4px 8px; }"
            "QToolButton:pressed { background-color: #e0e0e0; }"
            "QToolButton:disabled { background-color: #f8f8f0; color: #999; border: 1px solid #e8e8e8; }"
        )
        self.label_merge_btn.clicked.connect(self.open_label_merge_dialog)

        # 检测结果追加按钮
        self.detection_append_btn = QToolButton()
        self.detection_append_btn.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon)
        self.detection_append_btn.setAutoRaise(False)
        self.detection_append_btn.setText("检测追加")
        self.detection_append_btn.setToolTip("将当前图片的检测结果追加到标签文件（支持IoU过滤）")
        self.detection_append_btn.setEnabled(True)
        self.detection_append_btn.setStyleSheet(
            "QToolButton { background-color: #f0f0f0; color: #333; border: 1px solid #d0d0d0; border-radius: 4px; padding: 4px 8px; }"
            "QToolButton:pressed { background-color: #e0e0e0; }"
            "QToolButton:disabled { background-color: #f8f8f0; color: #999; border: 1px solid #e8e8e8; }"
        )
        self.detection_append_btn.clicked.connect(
            self.append_detections_to_label)

        # SAM 交互式分割
        self.sam_checkbox = QCheckBox("SAM模式")
        self.sam_checkbox.setToolTip(
            "开启后鼠标变为SAM交互模式：点提示(点击目标) / 框提示(拖拽框选) / 文字提示(输入文字概念)")
        self.sam_checkbox.setEnabled(False)  # 需先加载模型
        self.sam_output_combo = QComboBox()
        self.sam_output_combo.addItems(["目标框(BBox)", "分割Mask(Polygon)"])
        self.sam_output_combo.setToolTip("SAM 生成的标注类型")
        self.sam_prompt_combo = QComboBox()
        self.sam_prompt_combo.addItems(["点提示", "框提示", "文字提示"])
        self.sam_prompt_combo.setToolTip("SAM 提示方式：点击目标 / 拖拽框选 / 输入文字概念")
        self.sam_text_label = QLabel("文字:")
        self.sam_text_label.setVisible(False)
        self.sam_text_input = QLineEdit()
        self.sam_text_input.setPlaceholderText("提示词")
        self.sam_text_input.setMaximumWidth(150)
        self.sam_text_input.setVisible(False)
        self.sam_text_btn = QPushButton("运行")
        self.sam_text_btn.setVisible(False)
        # SAM 模型选择下拉 — 从 weights/ 扫描
        self._sam_weights_dir = os.path.join(os.path.dirname(__file__), "weights")
        self.sam_model_combo = QComboBox()
        self.sam_model_combo.setMinimumWidth(150)
        self.sam_model_combo.setToolTip("选择 weights/ 下的 SAM 模型文件（.pt）")
        self.sam_model_combo.addItem("")  # 空白初始项
        if os.path.isdir(self._sam_weights_dir):
            for f in sorted(os.listdir(self._sam_weights_dir)):
                if f.endswith(".pt"):
                    self.sam_model_combo.addItem(f)
        # 点击下拉框时热加载 SAM 模型列表
        _orig_sam_popup = self.sam_model_combo.showPopup

        def _sam_hot_popup():
            self._refresh_sam_models()
            _orig_sam_popup()

        self.sam_model_combo.showPopup = _sam_hot_popup

        self.file_search = QLineEdit()
        self.file_search.setPlaceholderText("文件搜索")
        self.file_search.textChanged.connect(self.file_search_changed)
        self.file_search.returnPressed.connect(self.file_search_jump)
        sync_btn_layout.addWidget(self.file_search)
        sync_btn_layout.addStretch(1)
        sync_btn_container = QWidget()
        sync_btn_container.setLayout(sync_btn_layout)
        file_list_layout.addWidget(sync_btn_container)

        class_filter_layout = QHBoxLayout()
        class_filter_layout.setContentsMargins(5, 5, 5, 5)
        class_filter_layout.addWidget(QLabel("类别筛选:"))
        self.class_filter_combo = QComboBox()
        try:
            self.class_filter_combo.setSizeAdjustPolicy(
                QComboBox.AdjustToMinimumContentsLengthWithIcon)
        except Exception:
            self.class_filter_combo.setSizeAdjustPolicy(
                QComboBox.AdjustToContents)
        try:
            self.class_filter_combo.setMinimumContentsLength(12)
        except Exception:
            pass
        self.class_filter_combo.setMaximumWidth(260)
        self.class_filter_combo.currentIndexChanged.connect(
            self.on_class_filter_changed)
        class_filter_layout.addWidget(self.class_filter_combo, 1)
        self.class_filter_multi_button = QToolButton()
        self.class_filter_multi_button.setAutoRaise(False)
        self.class_filter_multi_button.setText("多选")
        self.class_filter_multi_button.setToolTip("勾选多个类别后确定，用于筛选/显示")
        self.class_filter_multi_button.setMinimumWidth(70)
        self.class_filter_multi_button.setFixedHeight(24)  # 👈 这里固定高度
        self.class_filter_multi_button.setStyleSheet(
            "QToolButton { background-color: #f0f0f0; color: #333; border: 1px solid #d0d0d0; border-radius: 4px; padding: 4px 8px; }"
            "QToolButton:pressed { background-color: #e0e0e0; }"
            "QToolButton:disabled { background-color: #f8f8f8; color: #999; border: 1px solid #e8e8e8; }"
        )
        self.class_filter_multi_button.clicked.connect(
            self.open_multi_class_filter_dialog)
        class_filter_layout.addWidget(self.class_filter_multi_button)
        self.class_filter_restore_button = QToolButton()
        self.class_filter_restore_button.setAutoRaise(False)
        self.class_filter_restore_button.setText("恢复")
        self.class_filter_restore_button.setToolTip("清除过滤，恢复所有标签勾选状态")
        self.class_filter_restore_button.setMinimumWidth(60)
        self.class_filter_restore_button.setFixedHeight(24)
        self.class_filter_restore_button.setStyleSheet(
            "QToolButton { background-color: #fff3cd; color: #856404; border: 1px solid #ffc107; border-radius: 4px; padding: 4px 8px; }"
            "QToolButton:pressed { background-color: #ffe69c; }"
            "QToolButton:disabled { background-color: #f8f8f8; color: #999; border: 1px solid #e8e8e8; }"
        )
        self.class_filter_restore_button.clicked.connect(
            self._restore_label_uncheck_filter)
        self.class_filter_restore_button.setVisible(False)
        class_filter_layout.addWidget(self.class_filter_restore_button)
        self.only_show_selected_class_checkbox = QCheckBox("仅显示该类标签")
        self.only_show_selected_class_checkbox.setChecked(False)
        self.only_show_selected_class_checkbox.stateChanged.connect(
            self.on_only_show_selected_class_changed)
        class_filter_layout.addWidget(self.only_show_selected_class_checkbox)
        class_filter_container = QWidget()
        class_filter_container.setLayout(class_filter_layout)
        class_filter_container.setMinimumHeight(36)
        file_list_layout.addWidget(class_filter_container)

        file_list_layout.addWidget(self.file_list_widget)
        file_list_container = QWidget()
        file_list_container.setLayout(file_list_layout)
        self.file_dock = QDockWidget(get_str('fileList'), self)
        self.file_dock.setObjectName(get_str('files'))
        self.file_dock.setWidget(file_list_container)

        # Create detection results display widget - single image preview
        self.detection_results = {}
        self.detected_preview_label = ZoomableImageView(self)
        self.detected_preview_label.setMinimumSize(200, 200)
        self.detected_preview_label.setText("暂无检测结果")

        # Add info label for original image preview
        self.detected_info_label = ZoomableImageView(self)
        self.detected_info_label.setMinimumSize(200, 200)
        self.detected_info_label.setStyleSheet(
            "QScrollArea { background-color: #e8e8e8; border: 1px solid #aaa; }")
        self.detected_info_label.setText("原图")

        # Use splitter so user can drag the boundary between top/bottom
        # previews
        self.results_splitter = QSplitter(Qt.Vertical)
        self.results_splitter.setChildrenCollapsible(True)
        self.results_splitter.addWidget(self.detected_preview_label)
        self.results_splitter.addWidget(self.detected_info_label)
        self.results_splitter.setStretchFactor(0, 1)
        self.results_splitter.setStretchFactor(1, 1)

        results_container = QWidget()
        results_layout = QVBoxLayout()
        results_layout.setContentsMargins(5, 5, 5, 5)
        results_layout.addWidget(self.results_splitter)
        results_container.setLayout(results_layout)
        self.results_dock = QDockWidget('红色误检  黄色漏检', self)
        self.results_dock.setObjectName('detection_results')
        self.results_dock.setWidget(results_container)

        # Video tool dock (frame extraction + detection + optional video
        # export)
        self.video_player = VideoPlayerWidget(self)

        self.video_open_btn = QPushButton("选择视频")
        self.video_path_label = QLabel("未选择")
        self.video_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.video_output_dir_edit = QLineEdit()
        self.video_output_dir_btn = QPushButton("设置保存目录")
        try:
            default_out = QStandardPaths.writableLocation(
                QStandardPaths.PicturesLocation)
        except Exception:
            default_out = ""
        if not default_out:
            default_out = os.path.expanduser("~")
        self.video_output_dir_edit.setReadOnly(True)
        self.video_output_dir_edit.setText(
            self.default_save_dir if getattr(
                self, "default_save_dir", None) else default_out)

        self.video_mode_fps = QRadioButton("按帧率抽帧")
        self.video_mode_interval = QRadioButton("按时间间隔抽帧")
        self.video_mode_fps.setChecked(True)

        self.video_target_fps_spin = QDoubleSpinBox()
        self.video_target_fps_spin.setRange(0.1, 240.0)
        self.video_target_fps_spin.setValue(1.0)
        self.video_target_fps_spin.setDecimals(2)
        self.video_target_fps_spin.setSuffix(" fps")

        self.video_interval_spin = QDoubleSpinBox()
        self.video_interval_spin.setRange(0.1, 3600.0)
        self.video_interval_spin.setValue(1.0)
        self.video_interval_spin.setDecimals(2)
        self.video_interval_spin.setSuffix(" 秒")

        # 开始/结束时间滑动条（精确到0.01秒，range在选视频后绑定）
        self.video_start_slider = QSlider(Qt.Horizontal)
        self.video_start_slider.setRange(0, 0)
        self.video_start_slider.setValue(0)
        self.video_start_label = QLabel("00:00.00")
        self.video_end_slider = QSlider(Qt.Horizontal)
        self.video_end_slider.setRange(0, 0)
        self.video_end_slider.setValue(0)
        self.video_end_label = QLabel("00:00.00")

        self.video_run_btn = QPushButton("开始抽帧")
        self.video_detect_btn = QPushButton("开始检测")
        self.video_detect_btn.setEnabled(False)
        self.video_stop_btn = QPushButton("停止")
        self.video_stop_btn.setEnabled(False)

        self.video_progress = QProgressBar()
        self.video_progress.setRange(0, 100)
        self.video_progress.setValue(0)
        self.video_status = QLabel("")
        self.video_status.setWordWrap(True)

        # 视频信息标签
        self.video_info_res_label = QLabel("分辨率: --")
        self.video_info_fps_label = QLabel("帧率: --")
        self.video_info_frames_label = QLabel("总帧数: --")
        self.video_info_dur_label = QLabel("时长: --")

        # defaults for post-extract detect dialog
        self._video_detect_defaults = {
            "use_zh": False,
            "show_conf": True,
            "label_map_text": "",
            "export_video": False,
            "export_fps": 0.0,
        }
        self._last_extract_result = None

        # 视频信息
        info_box = QGroupBox("视频信息")
        info_layout = QGridLayout()
        info_layout.addWidget(self.video_info_res_label, 0, 0)
        info_layout.addWidget(self.video_info_fps_label, 0, 1)
        info_layout.addWidget(self.video_info_frames_label, 1, 0)
        info_layout.addWidget(self.video_info_dur_label, 1, 1)
        info_box.setLayout(info_layout)

        # Right panel controls (scrollable)
        controls = QWidget()
        vlayout = QVBoxLayout()
        vlayout.setContentsMargins(6, 6, 6, 6)

        vlayout.addWidget(info_box)

        row_open = QHBoxLayout()
        row_open.addWidget(self.video_open_btn)
        row_open.addWidget(self.video_path_label, 1)
        vlayout.addLayout(row_open)

        row_out = QHBoxLayout()
        row_out.addWidget(QLabel("保存目录:"))
        row_out.addWidget(self.video_output_dir_edit, 1)
        row_out.addWidget(self.video_output_dir_btn)
        vlayout.addLayout(row_out)

        mode_box = QGroupBox("抽帧参数")
        mode_layout = QGridLayout()
        mode_layout.addWidget(self.video_mode_fps, 0, 0)
        mode_layout.addWidget(self.video_target_fps_spin, 0, 1)
        mode_layout.addWidget(self.video_mode_interval, 1, 0)
        mode_layout.addWidget(self.video_interval_spin, 1, 1)
        mode_layout.addWidget(QLabel("开始时间:"), 2, 0)
        mode_layout.addWidget(self.video_start_slider, 2, 1)
        mode_layout.addWidget(self.video_start_label, 2, 2)
        mode_layout.addWidget(QLabel("结束时间:"), 3, 0)
        mode_layout.addWidget(self.video_end_slider, 3, 1)
        mode_layout.addWidget(self.video_end_label, 3, 2)
        mode_box.setLayout(mode_layout)
        vlayout.addWidget(mode_box)

        # 按钮行：抽帧、检测、停止
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.video_run_btn)
        btn_row.addWidget(self.video_detect_btn)
        btn_row.addWidget(self.video_stop_btn)
        vlayout.addLayout(btn_row)

        vlayout.addWidget(self.video_progress)
        vlayout.addWidget(self.video_status)
        vlayout.addStretch(1)
        controls.setLayout(vlayout)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls)

        # Left-right structure: big video on left, controls on right
        self.video_splitter = QSplitter(Qt.Horizontal)
        self.video_splitter.setChildrenCollapsible(False)
        self.video_splitter.addWidget(self.video_player)
        self.video_splitter.addWidget(controls_scroll)
        self.video_splitter.setStretchFactor(0, 3)
        self.video_splitter.setStretchFactor(1, 1)

        self.video_dock = QDockWidget("视频抽帧检测", self)
        self.video_dock.setObjectName("video_tool")
        # Keep dock empty by default; video UI is shown fullscreen via Ctrl+2.
        self.video_dock.setWidget(QLabel("使用 Ctrl+2 打开/关闭 视频抽帧(全屏)"))
        self.video_dock.setFeatures(QDockWidget.DockWidgetClosable |
                                    QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

        self._video_thread = None
        self._video_worker = None

        self.video_open_btn.clicked.connect(self.open_video_dialog)
        self.video_output_dir_btn.clicked.connect(self.change_save_dir_dialog)
        self.video_run_btn.clicked.connect(self.start_video_process)
        self.video_detect_btn.clicked.connect(self.start_video_detect)
        self.video_stop_btn.clicked.connect(self.stop_video_process)
        self.video_start_slider.valueChanged.connect(
            self._on_start_slider_changed)
        self.video_end_slider.valueChanged.connect(self._on_end_slider_changed)

        self.zoom_widget = ZoomWidget()
        self.color_dialog = ColorDialog(parent=self)

        self.canvas = Canvas(parent=self)
        self.canvas.zoomRequest.connect(self.zoom_request)
        self.canvas.set_drawing_shape_to_square(
            settings.get(SETTING_DRAW_SQUARE, False))

        # 模式切换弹窗提示
        self.mode_popup = QLabel(self)
        self.mode_popup.setAlignment(Qt.AlignCenter)
        self.mode_popup.setStyleSheet(
            "QLabel { background-color: rgba(0,0,0,180); color: white; font-size: 16px;"
            " font-weight: bold; border-radius: 12px; padding: 10px 24px; }"
        )
        self.mode_popup.hide()

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scroll_bars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal: scroll.horizontalScrollBar()
        }
        self.scroll_area = scroll
        self.canvas.scrollRequest.connect(self.scroll_request)

        self.canvas.newShape.connect(self.new_shape)
        self.canvas.shapeMoved.connect(self.set_dirty)
        self.canvas.selectionChanged.connect(self.shape_selection_changed)
        self.canvas.drawingPolygon.connect(self.toggle_drawing_sensitive)

        self.setCentralWidget(scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.file_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.results_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.video_dock)
        self.video_dock.hide()

        # Keep label panel and file list visible together on the right (1:1
        # height).
        try:
            self.setDockNestingEnabled(True)
            self.splitDockWidget(self.dock, self.file_dock, Qt.Vertical)
        except Exception:
            pass

        # Initialize dock widths: left/center/right = 2:4:1 (center is the
        # central widget).
        QTimer.singleShot(0, self.apply_main_layout_ratio)
        self.file_dock.setFeatures(QDockWidget.DockWidgetClosable |
                                   QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.results_dock.setFeatures(
            QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

        self.dock_features = QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        self.dock.setFeatures(self.dock_features)

        self.dock.show()
        self.file_dock.show()

        # Actions
        action = partial(new_action, self)

        load_data = action(get_str('loaddata'), self.loaddata,
                           'Ctrl+0', 'open', get_str('loaddata'))

        error_data = action(get_str('errordata'), self.errordata,
                            '', 'quit', get_str('errordata'))

        sam_toggle_action = action('SAM模式', self._on_sam_toggle_shortcut,
                                   's', None, '开启/关闭SAM模式')

        save_txt_action = action(get_str('savetxt'), self.savetxtaction,
                                 '', 'save', get_str('saveTxt'))

        save_currentTxt_action = action(get_str('savecurrentxt'), self.save_currentTxt_action,
                                        'q', 'save', get_str('saveCurrentTxt'))

        # 改
        open_model_action = action(get_str('loadModel'), self.openModel,
                                   'Ctrl+L', 'open', get_str('loadModelDetail'))

        detect_action = action(get_str('detect'), self.detectimg,
                               'f', 'format_yolo', get_str('detectImg'))

        batch_detect_action = action('Batch Detect', self.batch_detectimg,
                                     'Ctrl+f', 'format_yolo', 'Batch Detect All Images')

        batch_detect_from_current_action = action('批量检测从当前图片开始', self.batch_detect_from_current_changed,
                                                  None, None, '从当前图片开始批量检测', checkable=True)
        batch_detect_from_current_action.setChecked(
            self.batch_detect_from_current)

        resize_action = action('图像缩放', self.resize_images,
                               None, None, '批量缩放当前目录所有图片')

        label_merge_action = action('标签追加合并', self.open_label_merge_dialog,
                                    None, None, '将源目录选中类别的标签追加合并到目标目录')

        detection_append_action = action('检测追加', self.append_detections_to_label,
                                         None, None, '将当前图片的检测结果追加到标签文件')

        quit = action(get_str('quit'), self.close,
                      'Ctrl+Q', 'quit', get_str('quitApp'))

        open = action(get_str('openFile'), self.open_file,
                      'Ctrl+O', 'open', get_str('openFileDetail'))

        open_dir = action(get_str('openDir'), self.open_dir_dialog,
                          'Ctrl+u', 'open', get_str('openDir'))

        open_video = action('Open Video', self.open_video_dialog,
                            'Ctrl+Shift+V', 'open', 'Open video for frame extraction/detect')

        change_save_dir = action(get_str('changeSaveDir'), self.change_save_dir_dialog,
                                 'Ctrl+r', 'open', get_str('changeSavedAnnotationDir'))

        open_annotation = action(get_str('openAnnotation'), self.open_annotation_dialog,
                                 'Ctrl+Shift+O', 'open', get_str('openAnnotationDetail'))
        copy_prev_bounding = action(get_str('copyPrevBounding'), self.copy_previous_bounding_boxes, 'Ctrl+v', 'copy',
                                    get_str('copyPrevBounding'))

        open_next_image = action(get_str('nextImg'), self.open_next_image,
                                 'd', 'next', get_str('nextImgDetail'))

        open_prev_image = action(get_str('prevImg'), self.open_prev_image,
                                 'a', 'prev', get_str('prevImgDetail'))

        verify = action(get_str('verifyImg'), self.verify_image,
                        '', 'verify', get_str('verifyImgDetail'))

        save = action(get_str('save'), self.save_file,
                      'Ctrl+S', 'save', get_str('saveDetail'), enabled=False)

        def get_format_meta(format):
            """
            returns a tuple containing (title, icon_name) of the selected format
            """
            if format == LabelFileFormat.PASCAL_VOC:
                return '&PascalVOC', 'format_voc'
            elif format == LabelFileFormat.YOLO:
                return '&YOLO', 'format_yolo'
            elif format == LabelFileFormat.CREATE_ML:
                return '&CreateML', 'format_createml'

        save_format = action(get_format_meta(self.label_file_format)[0],
                             self.change_format, 'Ctrl+',
                             get_format_meta(self.label_file_format)[1],
                             get_str('changeSaveFormat'), enabled=True)

        save_as = action(get_str('saveAs'), self.save_file_as,
                         'Ctrl+Shift+S', 'save-as', get_str('saveAsDetail'), enabled=False)

        close = action(
            get_str('closeCur'),
            self.close_file,
            'Ctrl+W',
            'close',
            get_str('closeCurDetail'))

        delete_image = action(get_str('deleteImg'), self.delete_image, 'Ctrl+T', 'close',
                              get_str('deleteImgDetail'))

        reset_all = action(
            get_str('resetAll'),
            self.reset_all,
            None,
            'resetall',
            get_str('resetAllDetail'))

        sam_cursor_color_action = action('SAM光标颜色', self._choose_sam_cursor_color,
                                         None, None, '设置SAM穿透模式光标颜色')
        color1 = action(get_str('boxLineColor'), self.choose_color1,
                        'Ctrl+L', 'color_line', get_str('boxLineColorDetail'))

        create_mode = action(get_str('crtBox'), self.set_create_mode,
                             'w', 'new', get_str('crtBoxDetail'), enabled=False)
        edit_mode = action(get_str('editBox'), self.set_edit_mode,
                           'Ctrl+J', 'edit', get_str('editBoxDetail'), enabled=False)

        create = action(get_str('crtBox'), self.create_shape,
                        '', 'new', get_str('crtBoxDetail'), enabled=False)
        create_polygon = action('Create Polygon', self.create_polygon_shape,
                                '', 'new', 'Create Polygon', enabled=False)
        delete = action(get_str('delBox'), self.delete_selected_shape,
                        'T', 'delete', get_str('delBoxDetail'), enabled=False)
        copy = action(get_str('dupBox'), self.copy_selected_shape,
                      'Ctrl+D', 'copy', get_str('dupBoxDetail'),
                      enabled=False)

        advanced_mode = action(get_str('advancedMode'), self.toggle_advanced_mode,
                               'Ctrl+Shift+A', 'expert', get_str(
                'advancedModeDetail'),
                               checkable=True)

        hide_all = action(get_str('hideAllBox'), partial(self.toggle_polygons, False),
                          'Ctrl+H', 'hide', get_str('hideAllBoxDetail'),
                          enabled=False)
        show_all = action(get_str('showAllBox'), partial(self.toggle_polygons, True),
                          'Ctrl+A', 'hide', get_str('showAllBoxDetail'),
                          enabled=False)

        help_default = action(get_str('tutorialDefault'), self.show_default_tutorial_dialog, None, 'help',
                              get_str('tutorialDetail'))
        show_info = action(
            get_str('info'),
            self.show_info_dialog,
            None,
            'help',
            get_str('info'))
        show_shortcut = action(
            get_str('shortcut'),
            self.show_shortcuts_dialog,
            None,
            'help',
            get_str('shortcut'))

        zoom = QWidgetAction(self)
        zoom.setDefaultWidget(self.zoom_widget)
        self.zoom_widget.setWhatsThis(
            u"Zoom in or out of the image. Also accessible with"
            " %s and %s from the canvas." % (format_shortcut("Ctrl+[-+]"),
                                             format_shortcut("Ctrl+Wheel")))
        self.zoom_widget.setEnabled(False)

        zoom_in = action(get_str('zoomin'), partial(self.add_zoom, 10),
                         'Ctrl++', 'zoom-in', get_str('zoominDetail'), enabled=False)
        zoom_out = action(get_str('zoomout'), partial(self.add_zoom, -10),
                          'Ctrl+-', 'zoom-out', get_str('zoomoutDetail'), enabled=False)
        zoom_org = action(get_str('originalsize'), partial(self.set_zoom, 100),
                          'Ctrl+=', 'zoom', get_str('originalsizeDetail'), enabled=False)
        fit_window = action(get_str('fitWin'), self.set_fit_window,
                            'Ctrl+F', 'fit-window', get_str('fitWinDetail'),
                            checkable=True, enabled=False)
        fit_width = action(get_str('fitWidth'), self.set_fit_width,
                           'Ctrl+Shift+F', 'fit-width', get_str(
                'fitWidthDetail'),
                           checkable=True, enabled=False)
        # Group zoom controls into a list for easier toggling.
        zoom_actions = (self.zoom_widget, zoom_in, zoom_out,
                        zoom_org, fit_window, fit_width)
        self.zoom_mode = self.MANUAL_ZOOM
        self.scalers = {
            self.FIT_WINDOW: self.scale_fit_window,
            self.FIT_WIDTH: self.scale_fit_width,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action(get_str('editLabel'), self.edit_label,
                      'Ctrl+E', 'edit', get_str('editLabelDetail'),
                      enabled=False)
        self.edit_button.setDefaultAction(edit)

        shape_line_color = action(get_str('shapeLineColor'), self.choose_shape_line_color,
                                  icon='color_line', tip=get_str('shapeLineColorDetail'),
                                  enabled=False)
        shape_fill_color = action(get_str('shapeFillColor'), self.choose_shape_fill_color,
                                  icon='color', tip=get_str('shapeFillColorDetail'),
                                  enabled=False)

        labels = self.dock.toggleViewAction()
        labels.setText(get_str('showHide'))
        labels.setShortcut('Ctrl+Shift+L')

        files_toggle = self.file_dock.toggleViewAction()
        files_toggle.setText("显示/隐藏 文件列表")

        results_toggle = self.results_dock.toggleViewAction()
        results_toggle.setText("显示/隐藏 检测预览")
        results_toggle.setShortcut("Ctrl+1")

        # Keep the original dock toggle (no shortcut), and provide a fullscreen
        # toggle on Ctrl+2.
        video_toggle = self.video_dock.toggleViewAction()
        video_toggle.setText("显示/隐藏 视频工具(底部)")
        # Avoid shortcut conflict with fullscreen toggle.
        video_toggle.setShortcut("")

        # Ctrl+2: toggle video tool fullscreen (replaces the central canvas
        # temporarily).
        self.video_fullscreen_toggle = QAction("显示/隐藏 视频抽帧(全屏)", self)
        self.video_fullscreen_toggle.setShortcut("Ctrl+2")
        self.video_fullscreen_toggle.setShortcutContext(Qt.ApplicationShortcut)
        self.video_fullscreen_toggle.setCheckable(True)
        self.video_fullscreen_toggle.triggered.connect(
            self.toggle_video_fullscreen)
        self.addAction(self.video_fullscreen_toggle)

        # Ctrl+3: toggle right-side panels (labels + file list) so the canvas
        # can take full width.
        self.right_panel_toggle = QAction("显示/隐藏 右侧面板", self)
        self.right_panel_toggle.setShortcut("Ctrl+3")
        self.right_panel_toggle.triggered.connect(self.toggle_right_panels)
        # Ensure the shortcut works even if the action is not in focus.
        self.addAction(self.right_panel_toggle)

        # Label list context menu.
        label_menu = QMenu()
        add_actions(label_menu, (edit, delete))
        self.label_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.label_list.customContextMenuRequested.connect(
            self.pop_label_list_menu)

        # Draw squares/rectangles
        self.draw_squares_option = QAction(get_str('drawSquares'), self)
        self.draw_squares_option.setShortcut('Ctrl+Shift+R')
        self.draw_squares_option.setCheckable(True)
        self.draw_squares_option.setChecked(
            settings.get(SETTING_DRAW_SQUARE, False))
        self.draw_squares_option.triggered.connect(self.toggle_draw_square)
        self.draw_two_clicks_option = QAction("双点拉框模式", self)
        self.draw_two_clicks_option.setShortcut("Ctrl+Shift+D")
        self.draw_two_clicks_option.setCheckable(True)
        self.draw_two_clicks_option.setChecked(settings.get(SETTING_DRAW_TWO_CLICKS, False))
        self.draw_two_clicks_option.triggered.connect(self.toggle_draw_two_clicks)
        self.actions = Struct(save=save, save_format=save_format, saveAs=save_as, open=open, close=close,
                              resetAll=reset_all, deleteImg=delete_image,
                              lineColor=color1, create=create, createPolygon=create_polygon,
                              delete=delete, edit=edit, copy=copy,
                              createMode=create_mode, editMode=edit_mode, advancedMode=advanced_mode,
                              shapeLineColor=shape_line_color, shapeFillColor=shape_fill_color,
                              zoom=zoom, zoomIn=zoom_in, zoomOut=zoom_out, zoomOrg=zoom_org,
                              fitWindow=fit_window, fitWidth=fit_width,
                              batchDetectFromCurrent=batch_detect_from_current_action,
                              zoomActions=zoom_actions,
                              fileMenuActions=(
                                  open, open_dir, save, save_as, close, reset_all, quit),
                              beginner=(), advanced=(),
                              editMenu=(edit, copy, delete,
                                        None, color1, self.draw_squares_option, self.draw_two_clicks_option,
                                        None, sam_cursor_color_action),
                              beginnerContext=(
                                  create, create_polygon, edit, copy, delete),
                              advancedContext=(create_mode, edit_mode, edit, copy,
                                               delete, shape_line_color, shape_fill_color),
                              onLoadActive=(
                                  close, create, create_polygon, create_mode, edit_mode),
                              onShapesPresent=(save_as, hide_all, show_all))

        self._update_class_filter_items()

        self.menus = Struct(
            file=self.menu(get_str('menu_file')),
            edit=self.menu(get_str('menu_edit')),
            view=self.menu(get_str('menu_view')),
            help=self.menu(get_str('menu_help')),
            recentFiles=QMenu(get_str('menu_openRecent')),
            labelList=label_menu)

        # Auto saving : Enable auto saving if pressing next
        self.auto_saving = QAction(get_str('autoSaveMode'), self)
        self.auto_saving.setCheckable(True)
        self.auto_saving.setChecked(settings.get(SETTING_AUTO_SAVE, False))
        # Sync single class mode from PR#106
        self.single_class_mode = QAction(get_str('singleClsMode'), self)
        self.single_class_mode.setShortcut("Ctrl+Shift+S")
        self.single_class_mode.setCheckable(True)
        self.single_class_mode.setChecked(
            settings.get(SETTING_SINGLE_CLASS, False))
        self.lastLabel = None
        # Add option to enable/disable labels being displayed at the top of
        # bounding boxes
        self.display_label_option = QAction(get_str('displayLabel'), self)
        self.display_label_option.setShortcut("Ctrl+Shift+P")
        self.display_label_option.setCheckable(True)
        self.display_label_option.setChecked(
            settings.get(SETTING_PAINT_LABEL, False))
        self.display_label_option.triggered.connect(
            self.toggle_paint_labels_option)

        add_actions(self.menus.file,
                    (open, open_dir, open_video, change_save_dir, open_annotation, copy_prev_bounding,
                     self.menus.recentFiles, save,
                     save_format, save_as, close, reset_all, delete_image, quit,))
        add_actions(self.menus.help, (help_default, show_info, show_shortcut))
        add_actions(self.menus.view, (
            self.auto_saving,
            self.single_class_mode,
            self.display_label_option,
            self.actions.batchDetectFromCurrent,
            self.draw_squares_option,
            self.draw_two_clicks_option,
            labels, files_toggle, results_toggle, video_toggle, self.video_fullscreen_toggle, self.right_panel_toggle,
            advanced_mode, None,
            hide_all, show_all, None,
            zoom_in, zoom_out, zoom_org, None,
            fit_window, fit_width))

        self.menus.file.aboutToShow.connect(self.update_file_menu)

        # Custom context menu for the canvas widget:
        add_actions(self.canvas.menus[0], self.actions.beginnerContext)
        add_actions(self.canvas.menus[1], (
            action('&Copy here', self.copy_shape),
            action('&Move here', self.move_shape)))

        self.tools = self.toolbar('Tools')

        # 创建置信度输入框
        self.confidence_spinbox = QDoubleSpinBox()
        self.confidence_spinbox.setRange(0.0, 1.0)
        self.confidence_spinbox.setValue(0.25)
        self.confidence_spinbox.setSingleStep(0.01)
        self.confidence_spinbox.setDecimals(2)
        self.confidence_spinbox.setPrefix("conf: ")

        # 创建IoU输入框
        self.iou_spinbox = QDoubleSpinBox()
        self.iou_spinbox.setRange(0.0, 1.0)
        self.iou_spinbox.setValue(0.5)
        self.iou_spinbox.setSingleStep(0.01)
        self.iou_spinbox.setDecimals(2)
        self.iou_spinbox.setPrefix("IoU: ")

        # 新增：创建模型类型选择下拉框
        self.model_type_combobox = QComboBox()
        self.model_type_combobox.addItem("V8")  # 默认选项
        self.model_type_combobox.addItem("V8_Seg")
        self.model_type_combobox.addItem("V7")
        self.model_type_combobox.setCurrentText("V8")  # 显式设置默认值为 "V8"

        # YOLO 模型快速选择下拉 — model/ 目录下的文件
        self.model_select_combobox = QComboBox()
        self.model_select_combobox.setMinimumWidth(120)
        self.model_select_combobox.setToolTip("快速选择 model/ 目录下的模型文件")
        self.model_select_combobox.addItem("")  # 空白初始项
        _model_dir = os.path.join(os.path.dirname(__file__), "model")
        if os.path.isdir(_model_dir):
            for f in sorted(os.listdir(_model_dir)):
                # if f.endswith((".pt", ".onnx", ".engine", ".trt")):
                #     self.model_select_combobox.addItem(f)
                if f.endswith((".onnx")):
                    self.model_select_combobox.addItem(f)
        # 点击下拉框时热加载模型列表
        _orig_onnx_popup = self.model_select_combobox.showPopup

        def _onnx_hot_popup():
            self._refresh_onnx_models()
            _orig_onnx_popup()

        self.model_select_combobox.showPopup = _onnx_hot_popup

        # 可视化模式选择
        self.vis_mode_combobox = QComboBox()
        self.vis_mode_combobox.addItems(["目标框+Mask", "仅目标框", "仅Mask"])
        self.vis_mode_combobox.setCurrentText("目标框+Mask")

        # 保存模式选择
        self.save_mode_combobox = QComboBox()
        self.save_mode_combobox.addItems(["目标框标注", "分割标注", "两者"])
        self.save_mode_combobox.setCurrentText("目标框标注")

        # 切换图片筛选功能：勾选后选择保存目录，空格键保存当前图片到目录
        self._image_filter_save_dir = None
        self._image_filter_sync_txt = False
        self.box_contour_checkbox = QCheckBox("图片筛选模式")
        self.box_contour_checkbox.setToolTip(
            "勾选后选择保存目录；按空格键将当前图片保存到该目录（可选同步txt）")

        # ???????????????????????????????
        self._filter_space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self._filter_space_shortcut.setContext(Qt.ApplicationShortcut)
        self._filter_space_shortcut.activated.connect(self._on_filter_space_pressed)

        # ??? E/W??????/??????????
        self._polygon_shortcut = QShortcut(QKeySequence(Qt.Key_E), self)
        self._polygon_shortcut.setContext(Qt.ApplicationShortcut)
        self._polygon_shortcut.activated.connect(self.create_polygon_shape)

        self._rect_shortcut = QShortcut(QKeySequence(Qt.Key_W), self)
        self._rect_shortcut.setContext(Qt.ApplicationShortcut)
        self._rect_shortcut.activated.connect(self.create_shape)
        self._filter_move_combobox = QComboBox()
        self._filter_move_combobox.addItems(["复制到目录", "移动到目录"])
        self._filter_move_combobox.setCurrentIndex(0)
        self._filter_move_combobox.setToolTip("保存图片时复制还是移动到目标目录")

        def _on_image_filter_mode_toggled(checked):
            if not checked:
                return

            start_dir = self._image_filter_save_dir
            if not start_dir and getattr(self, "file_path", None):
                start_dir = os.path.dirname(self.file_path)

            save_dir = QFileDialog.getExistingDirectory(
                self,
                "选择筛选图片保存目录",
                start_dir or "",
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
            )

            if not save_dir:
                # 用户取消则自动取消勾选
                self.box_contour_checkbox.blockSignals(True)
                try:
                    self.box_contour_checkbox.setChecked(False)
                finally:
                    self.box_contour_checkbox.blockSignals(False)
                return

            self._image_filter_save_dir = save_dir
            reply = QMessageBox.question(
                self,
                "同步标注文件",
                "是否同步保存对应的 .txt 标注文件到该目录？\n（若没有对应txt则跳过）",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            self._image_filter_sync_txt = (reply == QMessageBox.Yes)
            sync_tip = "同步txt" if self._image_filter_sync_txt else "不同步txt"
            self.statusBar().showMessage(
                f"图片筛选保存目录: {save_dir}（{sync_tip}）", 5000)

        self.box_contour_checkbox.toggled.connect(
            _on_image_filter_mode_toggled)

        self.model_select_combobox.currentTextChanged.connect(
            self._on_model_select_changed)

        # SAM 信号连接
        self.sam_model_combo.currentTextChanged.connect(self._on_sam_model_selected)
        self.sam_checkbox.toggled.connect(self._on_sam_toggled)
        self.sam_prompt_combo.currentTextChanged.connect(self._on_sam_prompt_mode_changed)
        self.canvas.samResultReady.connect(self._on_sam_result_ready)
        self.sam_text_input.returnPressed.connect(self._on_sam_text_triggered)
        self.sam_text_btn.clicked.connect(self._on_sam_text_triggered)

        self.sam_cursor_color_btn = QToolButton()
        self.sam_cursor_color_btn.setToolTip("设置SAM光标颜色")
        self.sam_cursor_color_btn.clicked.connect(self._choose_sam_cursor_color)
        self.sam_cursor_color_btn.setStyleSheet(
            "QToolButton { background-color: #00ff00; border: 1px solid #999; min-width: 24px; min-height: 24px; border-radius: 4px; }"
        )
        self.canvas.samEncodeRequested.connect(self._sam_encode_current_image)
        self.canvas.shapeDoubleClicked.connect(self.edit_label)

        # 实时同步可视化模式到模型
        def _sync_vis_mode():
            if self.yolo_model is not None and hasattr(
                    self.yolo_model, 'vis_mode'):
                txt = self.vis_mode_combobox.currentText()
                if txt == "仅目标框":
                    self.yolo_model.vis_mode = "box"
                elif txt == "仅Mask":
                    self.yolo_model.vis_mode = "mask"
                else:
                    self.yolo_model.vis_mode = "both"

        self.vis_mode_combobox.currentTextChanged.connect(_sync_vis_mode)

        # 创建检测设置工具栏
        settings_toolbar = self.addToolBar('Detection Settings')
        settings_toolbar.addWidget(QLabel("                     "))
        settings_toolbar.addWidget(QLabel("检测参数: "))
        settings_toolbar.addWidget(self.confidence_spinbox)
        settings_toolbar.addWidget(self.iou_spinbox)
        settings_toolbar.addWidget(self.model_type_combobox)
        settings_toolbar.addWidget(QLabel(" 模型:"))
        settings_toolbar.addWidget(self.model_select_combobox)
        settings_toolbar.addWidget(QLabel(" 可视:"))
        settings_toolbar.addWidget(self.vis_mode_combobox)
        settings_toolbar.addWidget(QLabel(" 保存:"))
        settings_toolbar.addWidget(self.save_mode_combobox)
        settings_toolbar.addWidget(QLabel(" "))
        settings_toolbar.addWidget(QLabel(" | "))
        settings_toolbar.addWidget(self.box_contour_checkbox)
        settings_toolbar.addWidget(self._filter_move_combobox)
        settings_toolbar.addWidget(QLabel("  "))
        settings_toolbar.addWidget(QLabel(" | "))
        settings_toolbar.addWidget(self.resize_btn)
        settings_toolbar.addWidget(QLabel(" | "))
        settings_toolbar.addWidget(self.label_merge_btn)
        settings_toolbar.addWidget(QLabel(" | "))
        settings_toolbar.addWidget(self.detection_append_btn)
        settings_toolbar.addWidget(QLabel("  "))
        settings_toolbar.addWidget(QLabel(" | "))
        settings_toolbar.addWidget(self.sam_checkbox)
        settings_toolbar.addWidget(QLabel(" "))
        settings_toolbar.addWidget(self.sam_model_combo)
        settings_toolbar.addWidget(self.sam_output_combo)
        settings_toolbar.addWidget(self.sam_prompt_combo)
        # SAM 文字提示控件：默认隐藏，显示在状态栏（不被裁剪）
        self.sam_text_label.setVisible(False)
        self.sam_text_input.setVisible(False)
        self.sam_text_btn.setVisible(False)
        self._sam_needs_encode = False  # 切图后标记需编码

        self.yolo_model = None
        self.sam_client = SamClient()

        self.actions.beginner = (
            open_model_action,
            open_dir,
            detect_action,
            batch_detect_action,
            save_currentTxt_action,
            open, change_save_dir,
            open_next_image, open_prev_image,
            error_data,
            save_txt_action,
            load_data,
            sam_toggle_action,
            verify, save, save_format, None, create, create_polygon,
            copy, delete, None,
            zoom_in, zoom, zoom_out, fit_window, fit_width)

        self.actions.advanced = (
            open_model_action,
            open_dir,
            detect_action,
            save_currentTxt_action,
            open,
            change_save_dir,
            open_next_image,
            open_prev_image,
            error_data,
            save_txt_action,
            load_data,
            sam_toggle_action,
            save, save_format, None,
            create_mode, edit_mode, None,
            hide_all, show_all)

        self.statusBar().showMessage('%s started.' % __appname__)
        self.statusBar().show()

        # Application state.
        self.image = QImage()
        self.file_path = ustr(default_filename)
        self.last_open_dir = None
        self.recent_files = []
        self.max_recent = 7
        self.line_color = None
        self.fill_color = None
        self.zoom_level = 100
        self.fit_window = False
        # Add Chris
        self.difficult = False
        self._persistent_create_mode = 'rect'  # 'rect' or 'polygon'

        # Fix the compatible issue for qt4 and qt5. Convert the QStringList to
        # python list
        if settings.get(SETTING_RECENT_FILES):
            if have_qstring():
                recent_file_qstring_list = settings.get(SETTING_RECENT_FILES)
                self.recent_files = [ustr(i) for i in recent_file_qstring_list]
            else:
                self.recent_files = recent_file_qstring_list = settings.get(
                    SETTING_RECENT_FILES)

        size = settings.get(SETTING_WIN_SIZE, QSize(600, 500))
        position = QPoint(0, 0)
        saved_position = settings.get(SETTING_WIN_POSE, position)
        # Fix the multiple monitors issue
        for i in range(QApplication.desktop().screenCount()):
            if QApplication.desktop().availableGeometry(i).contains(saved_position):
                position = saved_position
                break
        self.resize(size)
        self.move(position)
        save_dir = ustr(settings.get(SETTING_SAVE_DIR, None))
        self.last_open_dir = ustr(settings.get(SETTING_LAST_OPEN_DIR, None))
        self.last_model_dir = ustr(settings.get(SETTING_LAST_MODEL_DIR, ""))
        if self.default_save_dir is None and save_dir is not None and os.path.exists(
                save_dir):
            self.default_save_dir = save_dir
            self.statusBar().showMessage('%s started. Annotation will be saved to %s' %
                                         (__appname__, self.default_save_dir))
            self.statusBar().show()

        self.restoreState(settings.get(SETTING_WIN_STATE, QByteArray()))
        Shape.line_color = self.line_color = QColor(
            settings.get(SETTING_LINE_COLOR, DEFAULT_LINE_COLOR))
        Shape.fill_color = self.fill_color = QColor(
            settings.get(SETTING_FILL_COLOR, DEFAULT_FILL_COLOR))
        self.canvas.set_drawing_color(self.line_color)
        self._sam_cursor_color = QColor(255, 0, 0)
        self.canvas.update_sam_cursor(self._sam_cursor_color)
        # Add chris
        Shape.difficult = self.difficult

        def xbool(x):
            if isinstance(x, QVariant):
                return x.toBool()
            return bool(x)

        if xbool(settings.get(SETTING_ADVANCE_MODE, False)):
            self.actions.advancedMode.setChecked(True)
            self.toggle_advanced_mode()

        # Populate the File menu dynamically.
        self.update_file_menu()

        # Since loading the file may take some time, make sure it runs in the
        # background.
        if self.file_path and os.path.isdir(self.file_path):
            self.queue_event(
                partial(
                    self.import_dir_images,
                    self.file_path or ""))
        elif self.file_path:
            self.queue_event(partial(self.load_file, self.file_path or ""))

        # Callbacks:
        self.zoom_widget.valueChanged.connect(self.paint_canvas)

        self.populate_mode_actions()

        # Display cursor coordinates at the right of status bar
        self.label_coordinates = QLabel('')
        self.statusBar().addPermanentWidget(self.label_coordinates)

        # SAM 文字提示控件固定在状态栏
        self.statusBar().addPermanentWidget(self.sam_text_label)
        self.statusBar().addPermanentWidget(self.sam_text_input)
        self.statusBar().addPermanentWidget(self.sam_text_btn)
        # 确保默认隐藏（addPermanentWidget 可能调用 show()）
        self.sam_text_label.setVisible(False)
        self.sam_text_input.setVisible(False)
        self.sam_text_btn.setVisible(False)

        # Open Dir if default file
        if self.file_path and os.path.isdir(self.file_path):
            self.open_dir_dialog(dir_path=self.file_path, silent=True)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.canvas.set_drawing_shape_to_square(False)
        elif event.key() == Qt.Key_Space:
            if getattr(self.canvas, "sam_passthrough", False):
                self.canvas.sam_passthrough = False
                self.canvas.unsetCursor()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.canvas.set_drawing_shape_to_square(True)

    # Support Functions #
    def set_format(self, save_format):
        if save_format == FORMAT_PASCALVOC:
            self.actions.save_format.setText(FORMAT_PASCALVOC)
            self.actions.save_format.setIcon(new_icon("format_voc"))
            self.label_file_format = LabelFileFormat.PASCAL_VOC
            LabelFile.suffix = XML_EXT

        elif save_format == FORMAT_YOLO:
            self.actions.save_format.setText(FORMAT_YOLO)
            self.actions.save_format.setIcon(new_icon("format_yolo"))
            self.label_file_format = LabelFileFormat.YOLO
            LabelFile.suffix = TXT_EXT

        elif save_format == FORMAT_CREATEML:
            self.actions.save_format.setText(FORMAT_CREATEML)
            self.actions.save_format.setIcon(new_icon("format_createml"))
            self.label_file_format = LabelFileFormat.CREATE_ML
            LabelFile.suffix = JSON_EXT

    def change_format(self):
        if self.label_file_format == LabelFileFormat.PASCAL_VOC:
            self.set_format(FORMAT_YOLO)
        elif self.label_file_format == LabelFileFormat.YOLO:
            self.set_format(FORMAT_CREATEML)
        elif self.label_file_format == LabelFileFormat.CREATE_ML:
            self.set_format(FORMAT_PASCALVOC)
        else:
            raise ValueError('Unknown label file format.')
        self.set_dirty()

    def no_shapes(self):
        return not self.items_to_shapes

    def toggle_advanced_mode(self, value=True):
        self._beginner = not value
        self.canvas.set_editing(True)
        self.populate_mode_actions()
        self.edit_button.setVisible(not value)
        if value:
            self.actions.createMode.setEnabled(True)
            self.actions.editMode.setEnabled(False)
        else:
            pass

    def toggle_right_panels(self):
        """
        Toggle the right-side dock widgets (label panel + file list) together.
        Bound to Ctrl+3 so Ctrl+1 (left) + Ctrl+3 (right) can free up the canvas.
        """
        right_docks = (self.dock, self.file_dock)
        any_visible = any(dock.isVisible() for dock in right_docks)
        target_visible = not any_visible
        for dock in right_docks:
            dock.setVisible(target_visible)

    def toggle_video_fullscreen(self):
        """
        Toggle the video frame-extraction/detection UI into the central area.
        When enabled, temporarily hides other docks so the video UI can use the full window.
        """
        # If video UI is currently shown: restore previous central widget and
        # dock visibilities.
        if self.centralWidget() is self.video_splitter:
            video_widget = self.takeCentralWidget()
            if video_widget is not None:
                video_widget.setParent(None)
            if hasattr(
                    self,
                    "_central_widget_before_video") and self._central_widget_before_video is not None:
                self.setCentralWidget(self._central_widget_before_video)
                self._central_widget_before_video = None
            if hasattr(self, "video_fullscreen_toggle"):
                self.video_fullscreen_toggle.setChecked(False)
            if hasattr(self, "_dock_visibility_before_video"):
                for dock, visible in self._dock_visibility_before_video.items():
                    dock.setVisible(visible)
                delattr(self, "_dock_visibility_before_video")
            QTimer.singleShot(0, self.apply_main_layout_ratio)

            # 手动切换回标注界面时，自动加载抽帧目录
            ext = getattr(self, "_last_extract_result", None)
            if ext:
                frames_dir = ext.get("frames_dir", "")
                frames = ext.get("frames", [])
                if frames_dir and os.path.isdir(frames_dir):
                    try:
                        self.open_dir_dialog(dir_path=frames_dir, silent=True)
                    except Exception:
                        pass
                if frames:
                    try:
                        self.load_file(frames[0])
                    except Exception:
                        pass
            return

        # Enter fullscreen video UI: store current dock visibilities and hide
        # them.
        self._dock_visibility_before_video = {
            self.dock: self.dock.isVisible(),
            self.file_dock: self.file_dock.isVisible(),
            self.results_dock: self.results_dock.isVisible(),
            self.video_dock: self.video_dock.isVisible(),
        }
        self.dock.hide()
        self.file_dock.hide()
        self.results_dock.hide()
        self.video_dock.hide()

        # Detach current central widget so it doesn't get deleted by Qt when we
        # swap.
        self._central_widget_before_video = self.takeCentralWidget()
        if self._central_widget_before_video is not None:
            self._central_widget_before_video.setParent(None)
        self.setCentralWidget(self.video_splitter)
        if hasattr(self, "video_fullscreen_toggle"):
            self.video_fullscreen_toggle.setChecked(True)

    def apply_main_layout_ratio(self):
        """
        Initialize/restore dock widths so the main labeling UI roughly matches:
        left : center : right = 2 : 4 : 1
        where center is the central widget (canvas scroll area).
        """
        try:
            total_w = max(1, int(self.width()))
            unit = max(60, total_w // 7)
            left_w = unit * 2
            right_w = unit * 1

            # Left dock (detection preview).
            if hasattr(self, "results_dock") and self.results_dock.isVisible():
                self.resizeDocks([self.results_dock], [left_w], Qt.Horizontal)

            # Right docks (labels + file list) share the same width.
            right_docks = []
            if hasattr(self, "dock") and self.dock.isVisible():
                right_docks.append(self.dock)
            if hasattr(self, "file_dock") and self.file_dock.isVisible():
                right_docks.append(self.file_dock)
            if right_docks:
                self.resizeDocks(
                    right_docks,
                    [right_w] *
                    len(right_docks),
                    Qt.Horizontal)

            # Right panel internal split: labels : files = 1 : 1 (vertical).
            if hasattr(
                    self,
                    "dock") and hasattr(
                self,
                "file_dock") and self.dock.isVisible() and self.file_dock.isVisible():
                total_h = max(1, int(self.height()))
                half_h = max(120, total_h // 2)
                self.resizeDocks([self.dock, self.file_dock], [
                    half_h, half_h], Qt.Vertical)
        except Exception:
            pass

    def populate_mode_actions(self):
        if self.beginner():
            tool, menu = self.actions.beginner, self.actions.beginnerContext
        else:
            tool, menu = self.actions.advanced, self.actions.advancedContext
        self.tools.clear()
        add_actions(self.tools, tool)
        self.canvas.menus[0].clear()
        add_actions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (self.actions.create,) if self.beginner() \
            else (self.actions.createMode, self.actions.editMode)
        add_actions(self.menus.edit, actions + self.actions.editMenu)

    def set_beginner(self):
        self.tools.clear()
        add_actions(self.tools, self.actions.beginner)

    def set_advanced(self):
        self.tools.clear()
        add_actions(self.tools, self.actions.advanced)

    def set_dirty(self):
        self.dirty = True
        self.actions.save.setEnabled(True)

    def set_clean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.create.setEnabled(True)

    def toggle_actions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def queue_event(self, function):
        QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def reset_state(self):
        self.items_to_shapes.clear()
        self.shapes_to_items.clear()
        self.label_list.clear()
        self.file_path = None
        self.image_data = None
        self.label_file = None
        self.canvas.reset_state()
        self.label_coordinates.clear()
        self.combo_box.cb.clear()

    def current_item(self):
        items = self.label_list.selectedItems()
        if items:
            return items[0]
        return None

    def add_recent_file(self, file_path):
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)
        elif len(self.recent_files) >= self.max_recent:
            self.recent_files.pop()
        self.recent_files.insert(0, file_path)

    def beginner(self):
        return self._beginner

    def advanced(self):
        return not self.beginner()

    def show_tutorial_dialog(self, browser='default', link=None):
        if link is None:
            link = self.screencast

        if browser.lower() == 'default':
            wb.open(link, new=2)
        elif browser.lower() == 'chrome' and self.os_name == 'Windows':
            if shutil.which(
                    browser.lower()):  # 'chrome' not in wb._browsers in windows
                wb.register('chrome', None, wb.BackgroundBrowser('chrome'))
            else:
                chrome_path = "D:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"
                if os.path.isfile(chrome_path):
                    wb.register(
                        'chrome', None, wb.BackgroundBrowser(chrome_path))
            try:
                wb.get('chrome').open(link, new=2)
            except:
                wb.open(link, new=2)
        elif browser.lower() in wb._browsers:
            wb.get(browser.lower()).open(link, new=2)

    def show_default_tutorial_dialog(self):
        self.show_tutorial_dialog(browser='default')

    def show_info_dialog(self):
        from libs.__init__ import __version__
        msg = u'Name:{0} \nApp Version:{1} \n{2} '.format(
            __appname__, __version__, sys.version_info)
        QMessageBox.information(self, u'Information', msg)

    def show_shortcuts_dialog(self):
        self.show_tutorial_dialog(
            browser='default',
            link='https://github.com/tzutalin/labelImg#Hotkeys')

    def _apply_expected_label_drawing_color(self):
        """Set canvas drawing color to match the expected label's class color."""
        if self.use_default_label_checkbox.isChecked(
        ) and self.default_label_text_line.text():
            color = generate_color_by_text(self.default_label_text_line.text())
            self.canvas.set_drawing_color(color)
        elif hasattr(self, 'lastLabel') and self.lastLabel:
            color = generate_color_by_text(self.lastLabel)
            self.canvas.set_drawing_color(color)

    def _show_mode_popup(self, text):
        """Show a brief auto-hiding mode indicator popup (like labelme)."""
        self.mode_popup.setText(text)
        self.mode_popup.adjustSize()
        # Center on the main window
        pp = self.mode_popup.sizeHint()
        x = (self.width() - pp.width()) // 2
        y = self.height() // 5
        self.mode_popup.move(x, y)
        self.mode_popup.show()
        self.mode_popup.raise_()
        QTimer.singleShot(800, self.mode_popup.hide)

    def _enter_create_mode(self, mode):
        """Enter persistent create mode for 'rect' or 'polygon'."""
        self._show_mode_popup("⬜ 矩形模式" if mode == 'rect' else "⬠ 多边形模式")
        self._persistent_create_mode = mode
        self._apply_expected_label_drawing_color()
        self.canvas.set_drawing_mode(mode)
        self.canvas.set_editing(False)
        self.actions.create.setEnabled(False)
        self.actions.createPolygon.setEnabled(False)

    def create_shape(self):
        assert self.beginner()
        if isinstance(QApplication.focusWidget(), (QLineEdit, QTextEdit)):
            return
        if self.canvas.drawing() and self.canvas.current is not None:
            self.canvas.current = None
            self.canvas.set_hiding(False)
            self.canvas.update()
        self._enter_create_mode('rect')

    def create_polygon_shape(self):
        assert self.beginner()
        if isinstance(QApplication.focusWidget(), (QLineEdit, QTextEdit)):
            return
        if self.canvas.drawing() and self.canvas.current is not None:
            self.canvas.current = None
            self.canvas.set_hiding(False)
            self.canvas.update()
        self._enter_create_mode('polygon')

    def toggle_drawing_sensitive(self, drawing=True):
        """In the middle of drawing, toggling between modes should be disabled."""
        self.actions.editMode.setEnabled(not drawing)
        if not drawing and self.beginner():
            # ESC or label-dialog cancel → go back to edit mode
            self.canvas.set_editing(True)
            self.canvas.restore_cursor()
            self.actions.create.setEnabled(True)
            self.actions.createPolygon.setEnabled(True)

    def toggle_draw_mode(self, edit=True):
        self.canvas.set_editing(edit)
        self.actions.createMode.setEnabled(edit)
        self.actions.editMode.setEnabled(not edit)

    def set_create_mode(self):
        assert self.advanced()
        self.toggle_draw_mode(False)

    def set_edit_mode(self):
        assert self.advanced()
        self.toggle_draw_mode(True)
        self.label_selection_changed()

    def update_file_menu(self):
        curr_file_path = self.file_path

        def exists(filename):
            return os.path.exists(filename)

        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recent_files if f !=
                 curr_file_path and exists(f)]
        for i, f in enumerate(files):
            icon = new_icon('labels')
            action = QAction(
                icon, '&%d %s' % (i + 1, QFileInfo(f).fileName()), self)
            action.triggered.connect(partial(self.load_recent, f))
            menu.addAction(action)

    def pop_label_list_menu(self, point):
        self.menus.labelList.exec_(self.label_list.mapToGlobal(point))

    def edit_label(self):
        if not self.canvas.editing():
            return
        item = self.current_item()
        if not item:
            return
        text = self.label_dialog.pop_up(item.text())
        if text is not None:
            item.setText(text)
            item.setBackground(generate_color_by_text(text))
            self.set_dirty()
            self.update_combo_box()

    # Tzutalin 20160906 : Add file list and dock to move faster
    def file_item_double_clicked(self, item=None):
        self.cur_img_idx = self.m_img_list.index(ustr(item.text()))
        filename = self.m_img_list[self.cur_img_idx]
        if filename:
            self.load_file(filename)

    def _on_file_list_context_menu(self, pos):
        item = self.file_list_widget.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        copy_path_action = menu.addAction("复制完整路径")
        copy_dir_action = menu.addAction("复制目录路径")
        menu.addSeparator()
        copy_name_action = menu.addAction("复制图片名")
        action = menu.exec_(self.file_list_widget.mapToGlobal(pos))
        if action == copy_path_action:
            QApplication.clipboard().setText(item.text())
            self.statusBar().showMessage("已复制: " + item.text(), 3000)
        elif action == copy_dir_action:
            dir_path = os.path.dirname(item.text())
            QApplication.clipboard().setText(dir_path)
            self.statusBar().showMessage("已复制: " + dir_path, 3000)
        elif action == copy_name_action:
            name = os.path.basename(item.text())
            QApplication.clipboard().setText(name)
            self.statusBar().showMessage("已复制: " + name, 3000)

    def file_search_changed(self):
        text = self.file_search.text().strip().lower()
        for i in range(self.file_list_widget.count()):
            item = self.file_list_widget.item(i)
            if text in item.text().lower():
                item.setHidden(False)
            else:
                item.setHidden(True)

    def file_search_jump(self):
        text = self.file_search.text().strip().lower()
        if not text:
            return
        for i in range(self.file_list_widget.count()):
            item = self.file_list_widget.item(i)
            if text in item.text().lower():
                self.file_list_widget.setCurrentItem(item)
                self.file_item_double_clicked(item)
                return

    # Add chris
    def button_state(self, item=None):
        """ Function to handle difficult examples
        Update on each object """
        if not self.canvas.editing():
            return

        item = self.current_item()
        if not item:  # If not selected Item, take the first one
            item = self.label_list.item(self.label_list.count() - 1)

        difficult = self.diffc_button.isChecked()

        try:
            shape = self.items_to_shapes[item]
        except:
            pass
        # Checked and Update
        try:
            if difficult != shape.difficult:
                shape.difficult = difficult
                self.set_dirty()
            else:  # User probably changed item visibility
                self.canvas.set_shape_visible(
                    shape, item.checkState() == Qt.Checked)
        except:
            pass

    # React to canvas signals.
    def shape_selection_changed(self, selected=False):
        if self._no_selection_slot:
            self._no_selection_slot = False
        else:
            shape = self.canvas.selected_shape
            if shape:
                self.shapes_to_items[shape].setSelected(True)
            else:
                self.label_list.clearSelection()
        self.actions.delete.setEnabled(selected)
        self.actions.copy.setEnabled(selected)
        self.actions.edit.setEnabled(selected)
        self.actions.shapeLineColor.setEnabled(selected)
        self.actions.shapeFillColor.setEnabled(selected)

    def add_label(self, shape):
        shape.paint_label = self.display_label_option.isChecked()
        if shape.label and shape.label not in self.label_hist:
            self.label_hist.append(shape.label)
            self._update_class_filter_items()
        item = HashableQListWidgetItem(shape.label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        item.setBackground(generate_color_by_text(shape.label))
        self.items_to_shapes[item] = shape
        self.shapes_to_items[shape] = item
        self.label_list.addItem(item)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)
        self.update_combo_box()
        if self._only_show_selected_class_labels and self._active_class_filter_set():
            self._apply_label_visibility_filter()
        if getattr(self, "_label_uncheck_filter_set", None):
            self._apply_label_uncheck_filter()
        # 保持按类别排序
        if self.label_list.count() > 1:
            self._sort_label_list_by_class()

    def remove_label(self, shape):
        if shape is None:
            # #print('rm empty label')
            return
        item = self.shapes_to_items.get(shape)
        if item is None:
            return
        self.label_list.takeItem(self.label_list.row(item))
        del self.shapes_to_items[shape]
        del self.items_to_shapes[item]
        self.update_combo_box()

    def load_labels(self, shapes):
        s = []
        for shape_tuple in shapes:
            label = shape_tuple[0]
            points = shape_tuple[1]
            line_color = shape_tuple[2] if len(shape_tuple) > 2 else None
            fill_color = shape_tuple[3] if len(shape_tuple) > 3 else None
            difficult = shape_tuple[4] if len(shape_tuple) > 4 else False
            shape_type = shape_tuple[5] if len(
                shape_tuple) > 5 else 'rectangle'

            shape = Shape(label=label, shape_type=shape_type)
            for x, y in points:

                # Ensure the labels are within the bounds of the image. If not,
                # fix them.
                x, y, snapped = self.canvas.snap_point_to_canvas(x, y)
                if snapped:
                    self.set_dirty()

                shape.add_point(QPointF(x, y))
            shape.difficult = difficult
            shape.close()
            s.append(shape)

            if line_color:
                shape.line_color = QColor(*line_color)
            else:
                shape.line_color = generate_color_by_text(label)

            if fill_color:
                shape.fill_color = QColor(*fill_color)
            else:
                shape.fill_color = generate_color_by_text(label)

            self.add_label(shape)
        self.update_combo_box()
        self.canvas.load_shapes(s)
        # 按类别名排序标签列表
        self._sort_label_list_by_class()

    def update_combo_box(self):
        # Get the unique labels and add them to the Combobox.
        items_text_list = [str(self.label_list.item(i).text())
                           for i in range(self.label_list.count())]

        unique_text_list = list(set(items_text_list))
        # Add a null row for showing all the labels
        unique_text_list.append("")
        unique_text_list.sort()

        self.combo_box.update_items(unique_text_list)

    def _sort_label_list_by_class(self):
        """按类别名排序 label_list，同类别标签集中显示"""
        count = self.label_list.count()
        if count <= 1:
            return
        items_with_text = []
        for i in range(count):
            item = self.label_list.takeItem(0)  # 从头部逐个取出
            items_with_text.append((item.text().lower(), item))
        # 按类别名字母序排列
        items_with_text.sort(key=lambda x: x[0])
        for _, item in items_with_text:
            self.label_list.addItem(item)

    def _update_class_filter_items(self):
        if not hasattr(self, "class_filter_combo"):
            return

        multi_item = "多选(自定义)…"
        current_text = self.class_filter_combo.currentText()
        base_labels = self._dataset_classes_from_file if self._dataset_classes_from_file else self.label_hist
        seen = set()
        items = ["全部", multi_item]
        for label in base_labels:
            if not label:
                continue
            if label in seen:
                continue
            seen.add(label)
            items.append(label)

        # Append any extra labels encountered at runtime (e.g. manual new
        # labels)
        for label in self.label_hist:
            if not label or label in seen:
                continue
            seen.add(label)
            items.append(label)

        self.class_filter_combo.blockSignals(True)
        try:
            self.class_filter_combo.clear()
            self.class_filter_combo.addItems(items)
            if current_text in items:
                self.class_filter_combo.setCurrentText(current_text)
            else:
                self.class_filter_combo.setCurrentIndex(0)
        finally:
            self.class_filter_combo.blockSignals(False)

        selected_text = ustr(self.class_filter_combo.currentText()).strip()
        if selected_text in ("", "全部"):
            self._selected_class_filter = None
            self._selected_class_filter_set = None
        elif selected_text == multi_item:
            self._selected_class_filter = None
        else:
            self._selected_class_filter_set = None
            self._selected_class_filter = selected_text

        # Keep UI consistent when multi-filter is active
        if self._selected_class_filter_set:
            try:
                self.class_filter_combo.blockSignals(True)
                self.class_filter_combo.setCurrentText(multi_item)
            finally:
                self.class_filter_combo.blockSignals(False)

    def on_class_filter_changed(self, _index=None):
        if not hasattr(self, "class_filter_combo"):
            return

        multi_item = "多选(自定义)…"
        prev_filter = self._selected_class_filter
        prev_filter_set = set(
            self._selected_class_filter_set) if self._selected_class_filter_set else None
        prev_text = "全部"
        if prev_filter_set:
            prev_text = multi_item
        elif prev_filter is not None:
            prev_text = prev_filter

        text = ustr(self.class_filter_combo.currentText()).strip()
        if text in ("", "全部"):
            self._selected_class_filter = None
            self._selected_class_filter_set = None
            self._apply_image_list_filter(
                keep_current=True,
                selected_filter=None,
                allow_empty=True)
            if self._only_show_selected_class_labels:
                self._apply_label_visibility_filter()
            return

        if text == multi_item:
            result = self._prompt_multi_class_filter_set(
                initial_checked=(
                        prev_filter_set or (
                    {prev_filter} if prev_filter else None))
            )
            if result is None:
                # cancelled
                try:
                    self.class_filter_combo.blockSignals(True)
                    self.class_filter_combo.setCurrentText(prev_text)
                finally:
                    self.class_filter_combo.blockSignals(False)
                return
            selected_set, mode = result

            if mode == "过滤":
                self._label_uncheck_filter_set = set(
                    selected_set) if selected_set else None
                self._apply_label_uncheck_filter()
                if hasattr(self, "class_filter_restore_button"):
                    self.class_filter_restore_button.setVisible(
                        bool(self._label_uncheck_filter_set))
                self.statusBar().showMessage("已应用标签过滤：取消勾选所选类别", 3000)
                # 不改变图片筛选状态
                try:
                    self.class_filter_combo.blockSignals(True)
                    self.class_filter_combo.setCurrentText(prev_text)
                finally:
                    self.class_filter_combo.blockSignals(False)
                return

            self._selected_class_filter = None
            self._selected_class_filter_set = set(
                selected_set) if selected_set else None
            if self._selected_class_filter_set and hasattr(
                    self, "only_show_selected_class_checkbox"):
                self.only_show_selected_class_checkbox.setChecked(True)
            if not self._apply_image_list_filter(
                    keep_current=True,
                    selected_filter=self._selected_class_filter_set,
                    allow_empty=False):
                if self._last_filter_cancelled:
                    QMessageBox.information(self, "提示", "已取消筛选")
                else:
                    QMessageBox.information(self, "提示", "筛选结果为空")
                self._selected_class_filter_set = prev_filter_set
                self._selected_class_filter = prev_filter
                try:
                    self.class_filter_combo.blockSignals(True)
                    self.class_filter_combo.setCurrentText(prev_text)
                finally:
                    self.class_filter_combo.blockSignals(False)
                return

            if self._only_show_selected_class_labels:
                self._apply_label_visibility_filter()
            return

        # single selection
        self._selected_class_filter_set = None
        if not self._apply_image_list_filter(
                keep_current=True,
                selected_filter=text,
                allow_empty=False):
            if self._last_filter_cancelled:
                QMessageBox.information(self, "提示", "已取消筛选")
            else:
                QMessageBox.information(self, "提示", f"数据集中没有类别：{text}")
            try:
                self.class_filter_combo.blockSignals(True)
                self.class_filter_combo.setCurrentText(prev_text)
            finally:
                self.class_filter_combo.blockSignals(False)
            self._selected_class_filter_set = prev_filter_set
            self._selected_class_filter = prev_filter
            return

        self._selected_class_filter = text
        if self._only_show_selected_class_labels:
            self._apply_label_visibility_filter()

        # If current image does not match the new filter, don't force-jump;
        # just ensure next/prev starts correctly.
        if self.file_path and self.file_path not in self.m_img_list:
            self.cur_img_idx = -1

    def on_only_show_selected_class_changed(self, _state):
        if not hasattr(self, "only_show_selected_class_checkbox"):
            return
        self._only_show_selected_class_labels = self.only_show_selected_class_checkbox.isChecked()
        self._apply_label_visibility_filter()

    def _effective_label_visibility_filter(self):
        if not self._only_show_selected_class_labels:
            return None
        selected = self._active_class_filter_set()
        return selected if selected else None

    def _normalize_class_filter_to_set(self, selected_filter):
        if selected_filter is None:
            return None
        if isinstance(selected_filter, set):
            values = selected_filter
        elif isinstance(selected_filter, (list, tuple)):
            values = set(selected_filter)
        else:
            values = {selected_filter}
        normalized = set()
        for v in values:
            try:
                s = ustr(v).strip()
            except Exception:
                s = str(v).strip()
            if s:
                normalized.add(s)
        return normalized if normalized else None

    def _active_class_filter_set(self, selected_filter=None):
        if selected_filter is None:
            selected_filter = self._selected_class_filter_set if self._selected_class_filter_set is not None else self._selected_class_filter
        if selected_filter is None:
            return None
        if isinstance(selected_filter, str):
            s = ustr(selected_filter).strip()
            return {s} if s else None
        return self._normalize_class_filter_to_set(selected_filter)

    def _all_available_filter_labels(self):
        base_labels = self._dataset_classes_from_file if self._dataset_classes_from_file else self.label_hist
        seen = set()
        labels = []
        for label in list(base_labels) + list(self.label_hist):
            if not label:
                continue
            if label in seen:
                continue
            seen.add(label)
            labels.append(label)
        return labels

    def _prompt_multi_class_filter_set(
            self, initial_checked=None, default_mode="筛选"):
        labels = self._all_available_filter_labels()
        if not labels:
            QMessageBox.information(self, "提示", "当前没有可选择的类别")
            return None

        initial_checked = set(initial_checked) if initial_checked else set()

        dlg = QDialog(self)
        dlg.setWindowTitle("多选类别")
        dlg.resize(420, 520)
        layout = QVBoxLayout(dlg)

        mode_row = QHBoxLayout()
        mode_group = QGroupBox("模式", dlg)
        mode_group_layout = QHBoxLayout(mode_group)
        mode_filter_images = QRadioButton("筛选", mode_group)  # 维持现有：筛选图片列表
        mode_filter_labels = QRadioButton(
            "过滤", mode_group)  # 新增：仅取消勾选 Box labels
        mode_group_layout.addWidget(mode_filter_images)
        mode_group_layout.addWidget(mode_filter_labels)
        mode_group_layout.addStretch(1)
        if default_mode == "过滤":
            mode_filter_labels.setChecked(True)
        else:
            mode_filter_images.setChecked(True)
        mode_row.addWidget(mode_group)
        layout.addLayout(mode_row)

        search = QLineEdit(dlg)
        search.setPlaceholderText("搜索类别...")
        layout.addWidget(search)

        listw = QListWidget(dlg)
        listw.setSelectionMode(QAbstractItemView.NoSelection)
        for name in labels:
            item = QListWidgetItem(name, listw)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(
                Qt.Checked if name in initial_checked else Qt.Unchecked)
        layout.addWidget(listw, 1)

        btn_row = QHBoxLayout()
        btn_all = QPushButton("全选", dlg)
        btn_none = QPushButton("全不选", dlg)
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
        buttons.button(QDialogButtonBox.Ok).setText("确定")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        layout.addWidget(buttons)

        def apply_filter_text(txt):
            t = (txt or "").strip().lower()
            for i in range(listw.count()):
                it = listw.item(i)
                it.setHidden(t not in it.text().lower())

        search.textChanged.connect(apply_filter_text)
        btn_all.clicked.connect(
            lambda: [
                listw.item(i).setCheckState(
                    Qt.Checked) for i in range(
                    listw.count())])
        btn_none.clicked.connect(
            lambda: [
                listw.item(i).setCheckState(
                    Qt.Unchecked) for i in range(
                    listw.count())])
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        if dlg.exec_() != QDialog.Accepted:
            return None

        selected = set()
        for i in range(listw.count()):
            it = listw.item(i)
            if it.checkState() == Qt.Checked:
                selected.add(ustr(it.text()).strip())
        mode = "过滤" if mode_filter_labels.isChecked() else "筛选"
        return selected, mode

    def open_multi_class_filter_dialog(self, _value=False):
        prev_filter = self._selected_class_filter
        prev_filter_set = set(
            self._selected_class_filter_set) if self._selected_class_filter_set else None

        result = self._prompt_multi_class_filter_set(
            initial_checked=self._active_class_filter_set() or None)
        if result is None:
            return
        selected_set, mode = result

        if mode == "过滤":
            self._label_uncheck_filter_set = set(
                selected_set) if selected_set else None
            self._apply_label_uncheck_filter()
            if hasattr(self, "class_filter_restore_button"):
                self.class_filter_restore_button.setVisible(
                    bool(self._label_uncheck_filter_set))
            self.statusBar().showMessage("已应用标签过滤：取消勾选所选类别", 3000)
            return

        self._selected_class_filter = None
        self._selected_class_filter_set = set(
            selected_set) if selected_set else None
        if self._selected_class_filter_set and hasattr(
                self, "only_show_selected_class_checkbox"):
            self.only_show_selected_class_checkbox.setChecked(True)
        try:
            if hasattr(self, "class_filter_combo"):
                self.class_filter_combo.blockSignals(True)
                self.class_filter_combo.setCurrentText(
                    "多选(自定义)…" if self._selected_class_filter_set else "全部")
        finally:
            if hasattr(self, "class_filter_combo"):
                self.class_filter_combo.blockSignals(False)

        ok = self._apply_image_list_filter(
            keep_current=True,
            selected_filter=self._selected_class_filter_set,
            allow_empty=(self._selected_class_filter_set is None),
        )
        if not ok:
            if self._last_filter_cancelled:
                QMessageBox.information(self, "提示", "已取消筛选")
            else:
                QMessageBox.information(self, "提示", "筛选结果为空")
            self._selected_class_filter = prev_filter
            self._selected_class_filter_set = prev_filter_set
            try:
                if hasattr(self, "class_filter_combo"):
                    self.class_filter_combo.blockSignals(True)
                    if prev_filter_set:
                        self.class_filter_combo.setCurrentText("多选(自定义)…")
                    elif prev_filter:
                        self.class_filter_combo.setCurrentText(prev_filter)
                    else:
                        self.class_filter_combo.setCurrentText("全部")
            finally:
                if hasattr(self, "class_filter_combo"):
                    self.class_filter_combo.blockSignals(False)
            return

        if self._only_show_selected_class_labels:
            self._apply_label_visibility_filter()

    def _apply_image_list_filter(
            self,
            keep_current=True,
            selected_filter=None,
            allow_empty=True):
        if not self.m_img_list_all:
            self.m_img_list_all = list(self.m_img_list)

        selected_filter = (
            self._selected_class_filter_set if self._selected_class_filter_set is not None else self._selected_class_filter) if selected_filter is None else selected_filter
        selected_filter_set = self._active_class_filter_set(selected_filter)

        if selected_filter_set is None:
            filtered_list = list(self.m_img_list_all)
        else:
            filtered_list = []
            self._last_filter_cancelled = False
            use_progress = len(self.m_img_list_all) >= 200
            progress = None
            if use_progress:
                display = ",".join(sorted(selected_filter_set))
                progress = QProgressDialog(
                    f"正在筛选类别：{display}", "取消", 0, len(
                        self.m_img_list_all), self)
                progress.setWindowModality(Qt.WindowModal)
                progress.show()

            for idx, image_path in enumerate(self.m_img_list_all):
                if progress is not None and (idx % 20 == 0):
                    progress.setValue(idx)
                    QApplication.processEvents()
                    if progress.wasCanceled():
                        self._last_filter_cancelled = True
                        progress.close()
                        return False
                if self._image_contains_any_labels(
                        image_path, selected_filter_set):
                    filtered_list.append(image_path)

            if progress is not None:
                progress.setValue(len(self.m_img_list_all))
                progress.close()

        if not allow_empty and selected_filter_set is not None and len(
                filtered_list) == 0:
            return False

        self.m_img_list = filtered_list
        self.img_count = len(self.m_img_list)

        self.file_list_widget.clear()
        for img_path in self.m_img_list:
            self.file_list_widget.addItem(QListWidgetItem(img_path))

        if self.img_count <= 0:
            self.cur_img_idx = 0
            self.statusBar().showMessage("筛选后无匹配图片")
            return True

        current_path = os.path.abspath(
            self.file_path) if self.file_path else None
        if keep_current and current_path and current_path in self.m_img_list:
            self.cur_img_idx = self.m_img_list.index(current_path)
            item = self.file_list_widget.item(self.cur_img_idx)
            if item:
                item.setSelected(True)
            self.setWindowTitle(
                __appname__ +
                ' ' +
                current_path +
                ' ' +
                self.counter_str())
            return True

        self.cur_img_idx = 0
        self.load_file(self.m_img_list[0])
        return True

    def _annotation_file_candidates(self, image_path):
        if not image_path:
            return None, None, None

        if self.default_save_dir is not None:
            basename = os.path.basename(os.path.splitext(image_path)[0])
            xml_path = os.path.join(self.default_save_dir, basename + XML_EXT)
            txt_path = os.path.join(self.default_save_dir, basename + TXT_EXT)
            json_path = os.path.join(
                self.default_save_dir, basename + JSON_EXT)
            return xml_path, txt_path, json_path

        xml_path = os.path.splitext(image_path)[0] + XML_EXT
        txt_path = os.path.splitext(image_path)[0] + TXT_EXT
        json_path = os.path.splitext(image_path)[0] + JSON_EXT
        return xml_path, txt_path, json_path

    def _read_yolo_classes(self, classes_file_path):
        try:
            if classes_file_path and os.path.exists(classes_file_path):
                with open(classes_file_path, "r", encoding="utf-8", errors="replace") as f:
                    return [line.strip() for line in f if line.strip()]
        except Exception:
            pass
        return list(self.label_hist) if self.label_hist else []

    def _scan_yolo_annotation_class_ids_in_dir(self, dir_path):
        class_ids = set()
        has_yolo = False
        if not dir_path or not os.path.exists(dir_path):
            return False, class_ids

        for root, _dirs, files in os.walk(dir_path):
            for filename in files:
                if not filename.lower().endswith(".txt"):
                    continue
                if filename.lower() == "classes.txt":
                    continue
                txt_path = os.path.join(root, filename)
                try:
                    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                        for _ in range(5):  # sample a few lines only
                            line = f.readline()
                            if not line:
                                break
                            parts = line.strip().split()
                            if len(parts) < 5:
                                continue
                            class_id = parts[0]
                            if class_id.lstrip("-").isdigit():
                                has_yolo = True
                                class_ids.add(class_id)
                except Exception:
                    continue
        return has_yolo, class_ids

    def _refresh_dataset_class_source(self, dir_path, show_warnings=True):
        self._asked_generate_classes_txt = False
        self.txt_path = os.path.join(
            dir_path, "classes.txt") if dir_path else ""
        self._dataset_has_classes_txt = bool(
            self.txt_path) and os.path.exists(
            self.txt_path)

        has_yolo, class_ids = self._scan_yolo_annotation_class_ids_in_dir(
            dir_path)
        self._dataset_has_yolo_annotations = has_yolo
        self._dataset_numeric_label_ids = set(class_ids)
        self._dataset_uses_numeric_labels = bool(
            has_yolo and not self._dataset_has_classes_txt)
        self._dataset_can_generate_classes_txt = not self._dataset_uses_numeric_labels
        self._dataset_classes_from_file = []

        self.label_hist = []
        if self._dataset_has_classes_txt:
            self.load_predefined_classes(self.txt_path)
            self._dataset_classes_from_file = list(self.label_hist)
        elif self._dataset_uses_numeric_labels:
            def sort_key(value):
                return int(value) if value.lstrip("-").isdigit() else value

            self.label_hist = sorted(class_ids, key=sort_key)
            if show_warnings:
                QMessageBox.warning(
                    self,
                    "提示",
                    "当前文件夹存在标注 txt，但缺少 classes.txt。\n"
                    "为避免标签混乱，将使用数字(0,1,2,...)作为类别名，且不会自动生成 classes.txt。\n"
                    "如需类别名，请手动补齐 classes.txt 后重新打开文件夹。"
                )

        self._update_class_filter_items()

    def _load_class_link_groups(self, dir_path):
        self._class_link_groups = []
        if not dir_path:
            return
        linkage_path = os.path.join(dir_path, "class_linkage.txt")
        if not os.path.exists(linkage_path):
            return
        groups = []
        try:
            with open(linkage_path, "r", encoding="utf-8", errors="replace") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = [p.strip() for p in line.replace(",", " ").split()]
                    parts = [p for p in parts if p]
                    if len(parts) >= 2:
                        groups.append(set(parts))
        except Exception:
            return

        merged = []
        for group in groups:
            placed = False
            for existing in merged:
                if existing.intersection(group):
                    existing.update(group)
                    placed = True
                    break
            if not placed:
                merged.append(set(group))
        self._class_link_groups = merged

    def _expand_linked_labels(self, label):
        if label is None:
            return set()
        # Legacy linkage-file behavior removed; treat as a single label.
        return {ustr(label)}

    def _selected_filter_label_set(self):
        return self._active_class_filter_set()

    def _ensure_classes_txt_for_detection(self, image_path):
        # Never overwrite existing dataset classes.txt; only optionally create
        # one for pure-image folders.
        if not self._dataset_can_generate_classes_txt:
            return
        if self.yolo_model is None or not hasattr(self.yolo_model, "classes"):
            return
        if not self.last_open_dir:
            return

        if self._dataset_has_yolo_annotations:
            return

        classes_path = os.path.join(self.last_open_dir, "classes.txt")
        if os.path.exists(classes_path):
            return

        if self._asked_generate_classes_txt:
            return
        self._asked_generate_classes_txt = True

        reply = QMessageBox.question(
            self,
            "提示",
            "当前文件夹未发现标注文件且缺少 classes.txt。\n"
            "是否根据当前模型类别创建 classes.txt？\n"
            "（不会覆盖已有 classes.txt）",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            with open(classes_path, "w", encoding="utf-8") as f:
                for name in self.yolo_model.classes:
                    f.write(str(name) + "\n")
        except Exception as e:
            self.error_message("生成 classes.txt 失败", str(e))
            return

        # Keep dropdown consistent after generation
        if self.last_open_dir:
            self._refresh_dataset_class_source(
                self.last_open_dir, show_warnings=False)

    def replace_classes_txt_with_model(self, _value=False):
        if self.yolo_model is None or not hasattr(self.yolo_model, "classes"):
            self.error_message("未加载模型", "请先加载 YOLO 模型")
            return

        model_classes = [
            str(x).strip() for x in list(
                getattr(
                    self.yolo_model,
                    "classes",
                    [])) if str(x).strip()]
        if not model_classes:
            self.error_message("模型类别为空", "当前模型未提供可用的类别列表")
            return

        target_dir = None

        if self.default_save_dir and os.path.isdir(self.default_save_dir):
            target_dir = self.default_save_dir
        elif self.last_open_dir and os.path.isdir(self.last_open_dir):
            target_dir = self.last_open_dir
        elif self.file_path:
            try:
                target_dir = os.path.dirname(os.path.abspath(self.file_path))
            except Exception:
                target_dir = None

        if not target_dir or not os.path.isdir(target_dir):
            self.error_message("未打开文件夹", "请先打开图片或文件夹，再写入 classes.txt")
            return

        classes_path = os.path.join(target_dir, "classes.txt")
        exists = os.path.exists(classes_path)

        warning_lines = []
        if bool(getattr(self, "_dataset_has_yolo_annotations", False)):
            warning_lines.append("注意：YOLO 标注中的类别 ID 依赖 classes.txt 的顺序。")
            warning_lines.append("请确认“模型类别顺序”与数据集标注 ID 完全一致，否则会导致标签错乱。")
        if exists:
            warning_lines.append("将覆盖已有 classes.txt。")

        title = "覆盖 classes.txt" if exists else "生成 classes.txt"
        msg = f"目标文件：\n{classes_path}\n\n将写入 {len(model_classes)} 个类别。\n"
        if warning_lines:
            msg += "\n" + "\n".join(warning_lines) + "\n"
        msg += "\n是否继续？"

        reply = QMessageBox.question(
            self,
            title,
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            with open(classes_path, "w", encoding="utf-8") as f:
                for name in model_classes:
                    f.write(name + "\n")
        except Exception as e:
            self.error_message("写入 classes.txt 失败", str(e))
            return

        # Refresh UI/state after replacement
        try:
            self._image_label_cache = {}
        except Exception:
            pass

        try:
            if self.last_open_dir and os.path.abspath(
                    target_dir) == os.path.abspath(self.last_open_dir):
                self._refresh_dataset_class_source(
                    self.last_open_dir, show_warnings=False)
            else:
                self.txt_path = classes_path
                self._dataset_has_classes_txt = True
                self._dataset_uses_numeric_labels = False
                self._dataset_can_generate_classes_txt = True
                self.label_hist = []
                self.load_predefined_classes(classes_path)
                self._dataset_classes_from_file = list(self.label_hist)
                self._update_class_filter_items()
        except Exception:
            pass

        try:
            self.statusBar().showMessage(f"classes.txt 已更新：{classes_path}")
            self.statusBar().show()
        except Exception:
            pass

    def _read_annotation_labels(self, image_path):
        xml_path, txt_path, json_path = self._annotation_file_candidates(
            image_path)
        labels = set()

        if xml_path and os.path.isfile(xml_path):
            try:
                from xml.etree import ElementTree
                root = ElementTree.parse(xml_path).getroot()
                for node in root.findall(".//object/name"):
                    if node.text:
                        labels.add(node.text.strip())
            except Exception:
                return set()
            return labels

        if txt_path and os.path.isfile(txt_path):
            classes_path = self.txt_path if getattr(self, "_dataset_has_classes_txt",
                                                    False) and self.txt_path and os.path.exists(self.txt_path) \
                else os.path.join(os.path.dirname(os.path.abspath(txt_path)), "classes.txt")
            classes_exists = bool(
                classes_path) and os.path.exists(classes_path)
            classes = self._read_yolo_classes(
                classes_path) if classes_exists else []
            try:
                with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        class_index = parts[0]
                        if class_index.isdigit():
                            if classes_exists:
                                idx = int(class_index)
                                if 0 <= idx < len(classes):
                                    labels.add(classes[idx])
                                else:
                                    labels.add(class_index)
                            else:
                                labels.add(class_index)
                        else:
                            labels.add(class_index)
            except Exception:
                return set()
            return labels

        if json_path and os.path.isfile(json_path):
            try:
                with open(json_path, "r", encoding="utf-8", errors="replace") as f:
                    data = json.load(f)
                filename = os.path.basename(image_path)
                if isinstance(data, list):
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        if item.get("image") != filename:
                            continue
                        for ann in item.get("annotations", []):
                            if isinstance(ann, dict) and ann.get("label"):
                                labels.add(str(ann["label"]))
            except Exception:
                return set()
            return labels

        return labels

    def _image_contains_label(self, image_path, label):
        if not label:
            return True

        image_path = os.path.abspath(ustr(image_path))
        if self.file_path and os.path.abspath(
                self.file_path) == image_path and self.canvas and self.canvas.shapes:
            return any(getattr(shape, "label", None) ==
                       label for shape in self.canvas.shapes)

        if image_path in self._image_label_cache:
            labels = self._image_label_cache[image_path]
        else:
            labels = self._read_annotation_labels(image_path)
            self._image_label_cache[image_path] = labels
        return label in labels

    def _image_contains_any_labels(self, image_path, labels_set):
        if not labels_set:
            return True

        image_path = os.path.abspath(ustr(image_path))
        if self.file_path and os.path.abspath(
                self.file_path) == image_path and self.canvas and self.canvas.shapes:
            for shape in self.canvas.shapes:
                if getattr(shape, "label", None) in labels_set:
                    return True
            return False

        if image_path in self._image_label_cache:
            labels = self._image_label_cache[image_path]
        else:
            labels = self._read_annotation_labels(image_path)
            self._image_label_cache[image_path] = labels
        return any(lbl in labels for lbl in labels_set)

    def _apply_label_visibility_filter(self):
        selected = self._effective_label_visibility_filter()

        # Prune visibility map to current shapes to avoid stale refs
        try:
            current_shapes = set(self.canvas.shapes) if self.canvas else set()
            if hasattr(self.canvas, "visible"):
                for shape in list(self.canvas.visible.keys()):
                    if shape not in current_shapes:
                        del self.canvas.visible[shape]
        except Exception:
            pass

        if selected is None:
            self.label_list.blockSignals(True)
            try:
                for i in range(self.label_list.count()):
                    item = self.label_list.item(i)
                    item.setHidden(False)
                    item.setCheckState(Qt.Checked)
                    shape = self.items_to_shapes.get(item)
                    if shape is not None:
                        self.canvas.visible[shape] = True
            finally:
                self.label_list.blockSignals(False)
            if self.canvas and self.canvas.updatesEnabled():
                self.canvas.update()
            return

        self.label_list.blockSignals(True)
        try:
            for i in range(self.label_list.count()):
                item = self.label_list.item(i)
                is_match = (ustr(item.text()) in selected)
                item.setHidden(not is_match)
                item.setCheckState(Qt.Checked if is_match else Qt.Unchecked)
                shape = self.items_to_shapes.get(item)
                if shape is not None:
                    self.canvas.visible[shape] = is_match
        finally:
            self.label_list.blockSignals(False)

        if self.canvas.selected_shape and not self.canvas.isVisible(
                self.canvas.selected_shape):
            self.canvas.de_select_shape()
        if self.canvas and self.canvas.updatesEnabled():
            self.canvas.update()

    def _apply_label_uncheck_filter(self):
        """应用或清除过滤。selected 为空时恢复全部勾选。"""
        selected = getattr(self, "_label_uncheck_filter_set", None)
        if not hasattr(self, "label_list"):
            return

        self.label_list.blockSignals(True)
        try:
            if selected:
                # 过滤：取消勾选命中类别
                for i in range(self.label_list.count()):
                    item = self.label_list.item(i)
                    if ustr(item.text()) in selected:
                        item.setCheckState(Qt.Unchecked)
                        shape = self.items_to_shapes.get(item)
                        if shape is not None:
                            self.canvas.visible[shape] = False
            else:
                # 恢复：全部勾选
                for i in range(self.label_list.count()):
                    item = self.label_list.item(i)
                    if item.checkState() == Qt.Unchecked:
                        item.setCheckState(Qt.Checked)
                        shape = self.items_to_shapes.get(item)
                        if shape is not None:
                            self.canvas.visible[shape] = True
        finally:
            self.label_list.blockSignals(False)

        if self.canvas and self.canvas.updatesEnabled():
            self.canvas.update()

    def _restore_label_uncheck_filter(self):
        """清除过滤集并恢复全部标签勾选状态。"""
        self._label_uncheck_filter_set = None
        self._apply_label_uncheck_filter()
        if hasattr(self, "class_filter_restore_button"):
            self.class_filter_restore_button.setVisible(False)
        self.statusBar().showMessage("已恢复全部标签勾选", 3000)

    def _on_current_image_labels_changed(self):
        if self.file_path is None:
            return
        current_path = os.path.abspath(self.file_path)
        if current_path in self._image_label_cache:
            del self._image_label_cache[current_path]

        selected_set = self._active_class_filter_set()
        if not selected_set:
            return

        still_matches = False
        if self.canvas and self.canvas.shapes:
            for shape in self.canvas.shapes:
                if getattr(shape, "label", None) in selected_set:
                    still_matches = True
                    break

        if still_matches:
            return

        # Current image no longer matches the active filter; remove it from the
        # filtered list.
        if current_path in self.m_img_list:
            remove_index = self.m_img_list.index(current_path)
            self.m_img_list.pop(remove_index)
            self.img_count = len(self.m_img_list)

            self.file_list_widget.blockSignals(True)
            try:
                self.file_list_widget.takeItem(remove_index)
            finally:
                self.file_list_widget.blockSignals(False)

            if self.img_count <= 0:
                QMessageBox.information(
                    self,
                    "提示",
                    f"当前图片已不包含筛选类别：{','.join(sorted(selected_set))}\n筛选结果为空，已切换为全部。"
                )
                self._selected_class_filter = None
                self._selected_class_filter_set = None
                if hasattr(self, "class_filter_combo"):
                    self.class_filter_combo.blockSignals(True)
                    try:
                        self.class_filter_combo.setCurrentText("全部")
                    finally:
                        self.class_filter_combo.blockSignals(False)
                # Restore full list but keep current image displayed.
                self.m_img_list = list(
                    self.m_img_list_all) if self.m_img_list_all else []
                self.img_count = len(self.m_img_list)
                self.file_list_widget.clear()
                for img_path in self.m_img_list:
                    self.file_list_widget.addItem(QListWidgetItem(img_path))
                if current_path in self.m_img_list:
                    self.cur_img_idx = self.m_img_list.index(current_path)
                    item = self.file_list_widget.item(self.cur_img_idx)
                    if item:
                        item.setSelected(True)
                return

            # Do not auto-navigate here (avoid re-entrant load during delete).
            # Make next/prev continue from the correct position.
            self.cur_img_idx = remove_index - 1
            self.statusBar().showMessage(f"当前图片已不包含筛选类别：{','.join(sorted(selected_set))}，下次切换将跳过")

    def save_labels(self, annotation_file_path):
        annotation_file_path = ustr(annotation_file_path)
        if self.label_file is None:
            self.label_file = LabelFile()
            self.label_file.verified = self.canvas.verified

        def format_shape(s):
            return dict(label=s.label,
                        line_color=s.line_color.getRgb(),
                        fill_color=s.fill_color.getRgb(),
                        points=[(p.x(), p.y()) for p in s.points],
                        shape_type=s.shape_type,
                        # add chris
                        difficult=s.difficult)

        shapes = [format_shape(shape) for shape in self.canvas.shapes]
        # Can add different annotation formats here
        try:
            if self.label_file_format == LabelFileFormat.PASCAL_VOC:
                if annotation_file_path[-4:].lower() != ".xml":
                    annotation_file_path += XML_EXT
                self.label_file.save_pascal_voc_format(annotation_file_path, shapes, self.file_path, self.image_data,
                                                       self.line_color.getRgb(), self.fill_color.getRgb())
            elif self.label_file_format == LabelFileFormat.YOLO:
                if annotation_file_path[-4:].lower() != ".txt":
                    annotation_file_path += TXT_EXT
                self.label_file.save_yolo_format(annotation_file_path, shapes, self.file_path, self.image_data,
                                                 self.label_hist,
                                                 self.line_color.getRgb(), self.fill_color.getRgb())
            elif self.label_file_format == LabelFileFormat.CREATE_ML:
                if annotation_file_path[-5:].lower() != ".json":
                    annotation_file_path += JSON_EXT
                self.label_file.save_create_ml_format(annotation_file_path, shapes, self.file_path, self.image_data,
                                                      self.label_hist, self.line_color.getRgb(),
                                                      self.fill_color.getRgb())
            else:
                self.label_file.save(annotation_file_path, shapes, self.file_path, self.image_data,
                                     self.line_color.getRgb(), self.fill_color.getRgb())
            # print('Image:{0} -> Annotation:{1}'.format(self.file_path,
            # annotation_file_path))
            return True
        except LabelFileError as e:
            self.error_message(u'Error saving label data', u'<b>%s</b>' % e)
            return False

    def copy_selected_shape(self):
        self.add_label(self.canvas.copy_selected_shape())
        # fix copy and delete
        self.shape_selection_changed(True)

    def combo_selection_changed(self, index):
        text = self.combo_box.cb.itemText(index)
        for i in range(self.label_list.count()):
            if text == "":
                self.label_list.item(i).setCheckState(2)
            elif text != self.label_list.item(i).text():
                self.label_list.item(i).setCheckState(0)
            else:
                self.label_list.item(i).setCheckState(2)

    def label_selection_changed(self):
        item = self.current_item()
        if item and self.canvas.editing():
            shape = self.items_to_shapes.get(item)
            if shape is None:
                return
            self._no_selection_slot = True
            self.canvas.select_shape(shape)
            # Add Chris
            self.diffc_button.setChecked(shape.difficult)

    def label_item_changed(self, item):
        shape = self.items_to_shapes.get(item)
        if shape is None:
            return
        label = item.text()
        if label != shape.label:
            shape.label = item.text()
            shape.line_color = generate_color_by_text(shape.label)
            self.set_dirty()
        else:  # User probably changed item visibility
            self.canvas.set_shape_visible(
                shape, item.checkState() == Qt.Checked)

    # Callback functions:
    def new_shape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        if not self.use_default_label_checkbox.isChecked(
        ) or not self.default_label_text_line.text():
            if len(self.label_hist) > 0:
                self.label_dialog = LabelDialog(
                    parent=self, list_item=self.label_hist)

            # Sync single class mode from PR#106
            if self.single_class_mode.isChecked() and self.lastLabel:
                text = self.lastLabel
            else:
                text = self.label_dialog.pop_up(text=self.prev_label_text)
                self.lastLabel = text
        else:
            text = self.default_label_text_line.text()

        # Add Chris
        self.diffc_button.setChecked(False)
        if text is not None:
            self.prev_label_text = text
            generate_color = generate_color_by_text(text)
            shape = self.canvas.set_last_label(
                text, generate_color, generate_color)
            self.add_label(shape)
            if self.beginner():  # Stay in create mode for continuous annotation
                self._apply_expected_label_drawing_color()
                self.canvas.set_drawing_mode(self._persistent_create_mode)
                self.canvas.set_editing(False)
                self.actions.create.setEnabled(False)
                self.actions.createPolygon.setEnabled(False)
            else:
                self.actions.editMode.setEnabled(True)
            self.set_dirty()

            if text not in self.label_hist:
                self.label_hist.append(text)
        else:
            # self.canvas.undoLastLine()
            self.canvas.reset_all_lines()

    def scroll_request(self, delta, orientation):
        bar = self.scroll_bars[orientation]
        bar.setValue(int(bar.value() + delta))

    def set_zoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoom_mode = self.MANUAL_ZOOM
        self.zoom_widget.setValue(int(value))

    def add_zoom(self, increment=10):
        self.set_zoom(self.zoom_widget.value() + increment)

    def zoom_request(self, delta):
        # get the current scrollbar positions
        # calculate the percentages ~ coordinates
        h_bar = self.scroll_bars[Qt.Horizontal]
        v_bar = self.scroll_bars[Qt.Vertical]

        # get the current maximum, to know the difference after zooming
        h_bar_max = h_bar.maximum()
        v_bar_max = v_bar.maximum()

        # get the cursor position and canvas size
        # calculate the desired movement from 0 to 1
        # where 0 = move left
        #       1 = move right
        # up and down analogous
        cursor = QCursor()
        pos = cursor.pos()
        relative_pos = QWidget.mapFromGlobal(self, pos)

        cursor_x = relative_pos.x()
        cursor_y = relative_pos.y()

        w = self.scroll_area.width()
        h = self.scroll_area.height()

        # the scaling from 0 to 1 has some padding
        # you don't have to hit the very leftmost pixel for a maximum-left
        # movement
        margin = 0.1
        move_x = (cursor_x - margin * w) / (w - 2 * margin * w)
        move_y = (cursor_y - margin * h) / (h - 2 * margin * h)

        # clamp the values from 0 to 1
        move_x = min(max(move_x, 0), 1)
        move_y = min(max(move_y, 0), 1)

        # zoom in
        units = delta / (8 * 15)
        scale = 10
        self.add_zoom(scale * units)

        # get the difference in scrollbar values
        # this is how far we can move
        d_h_bar_max = h_bar.maximum() - h_bar_max
        d_v_bar_max = v_bar.maximum() - v_bar_max

        # get the new scrollbar values
        new_h_bar_value = h_bar.value() + move_x * d_h_bar_max
        new_v_bar_value = v_bar.value() + move_y * d_v_bar_max

        h_bar.setValue(int(new_h_bar_value))
        v_bar.setValue(int(new_v_bar_value))

    def set_fit_window(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoom_mode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjust_scale()

    def set_fit_width(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoom_mode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjust_scale()

    def toggle_polygons(self, value):
        for item, shape in self.items_to_shapes.items():
            item.setCheckState(Qt.Checked if value else Qt.Unchecked)

    def load_file(self, file_path=None):
        """Load the specified file, or the last opened file if None."""
        self.reset_state()
        self.canvas.setEnabled(False)
        if file_path is None:
            file_path = self.settings.get(SETTING_FILENAME)

        # Make sure that filePath is a regular python string, rather than
        # QString
        file_path = ustr(file_path)

        # Fix bug: An  index error after select a directory when open a new
        # file.
        unicode_file_path = ustr(file_path)
        unicode_file_path = os.path.abspath(unicode_file_path)
        # Tzutalin 20160906 : Add file list and dock to move faster
        # Highlight the file item
        if unicode_file_path and self.file_list_widget.count() > 0:
            if unicode_file_path in self.m_img_list:
                index = self.m_img_list.index(unicode_file_path)
                file_widget_item = self.file_list_widget.item(index)
                file_widget_item.setSelected(True)
            else:
                if not (
                        self.m_img_list_all and unicode_file_path in self.m_img_list_all):
                    self.file_list_widget.clear()
                    self.m_img_list.clear()

        if unicode_file_path and os.path.exists(unicode_file_path):
            if LabelFile.is_label_file(unicode_file_path):
                try:
                    self.label_file = LabelFile(unicode_file_path)
                except LabelFileError as e:
                    self.error_message(u'Error opening file',
                                       (u"<p><b>%s</b></p>"
                                        u"<p>Make sure <i>%s</i> is a valid label file.")
                                       % (e, unicode_file_path))
                    self.status("Error reading %s" % unicode_file_path)
                    return False
                self.image_data = self.label_file.image_data
                self.line_color = QColor(*self.label_file.lineColor)
                self.fill_color = QColor(*self.label_file.fillColor)
                self.canvas.verified = self.label_file.verified
            else:
                # Load image:
                # read data first and store for saving into label file.
                self.image_data = read(unicode_file_path, None)
                self.label_file = None
                self.canvas.verified = False

            if isinstance(self.image_data, QImage):
                image = self.image_data
            else:
                image = QImage.fromData(self.image_data)
            if image.isNull():
                self.error_message(u'Error opening file',
                                   u"<p>Make sure <i>%s</i> is a valid image file." % unicode_file_path)
                self.status("Error reading %s" % unicode_file_path)
                return False
            self.status("Loaded %s" % os.path.basename(unicode_file_path))
            self.image = image
            self.file_path = unicode_file_path
            self.canvas.load_pixmap(QPixmap.fromImage(image))
            # SAM: 切图时标记需重编码（延迟执行，不阻塞切换）
            self._sam_needs_encode = True
            if self.canvas.sam_enabled and self.sam_client.is_loaded:
                QTimer.singleShot(200, self._sam_encode_if_needed)
            if self.label_file:
                self.load_labels(self.label_file.shapes)
            self.set_clean()
            self.canvas.setEnabled(True)
            self.adjust_scale(initial=True)
            self.paint_canvas()
            self.add_recent_file(self.file_path)
            self.toggle_actions(True)
            self.canvas.setUpdatesEnabled(False)
            try:
                self.show_bounding_box_from_annotation_file(file_path)
                if self._only_show_selected_class_labels:
                    self._apply_label_visibility_filter()
                if getattr(self, "_label_uncheck_filter_set", None):
                    self._apply_label_uncheck_filter()
            finally:
                self.canvas.setUpdatesEnabled(True)
            self.canvas.update()

            counter = self.counter_str()
            self.setWindowTitle(__appname__ + ' ' + file_path + ' ' + counter)

            # Default : select last item if there is at least one item
            if self.label_list.count():
                for i in range(self.label_list.count() - 1, -1, -1):
                    item = self.label_list.item(i)
                    if not item.isHidden():
                        self.label_list.setCurrentItem(item)
                        item.setSelected(True)
                        break

            self.canvas.setFocus(True)

            # Update left preview
            self.update_preview_display()

            # 仅视频检测后：无标注文件但有检测结果时，加载检测框到画布
            if (getattr(self, '_video_detect_context', False)
                    and self.label_list.count() == 0
                    and hasattr(self, 'detection_results')
                    and self.file_path in self.detection_results):
                self._load_shapes_from_detection_result(self.file_path)

            return True
        return False

    def counter_str(self):
        """
        Converts image counter to string representation.
        """
        return '[{} / {}]'.format(self.cur_img_idx + 1, self.img_count)

    def show_bounding_box_from_annotation_file(self, file_path):
        if self.default_save_dir is not None:
            basename = os.path.basename(os.path.splitext(file_path)[0])
            xml_path = os.path.join(self.default_save_dir, basename + XML_EXT)
            txt_path = os.path.join(self.default_save_dir, basename + TXT_EXT)
            json_path = os.path.join(
                self.default_save_dir, basename + JSON_EXT)

            """Annotation file priority:
            PascalXML > YOLO
            """
            if os.path.isfile(xml_path):
                self.load_pascal_xml_by_filename(xml_path)
            elif os.path.isfile(txt_path):
                self.load_yolo_txt_by_filename(txt_path)
            elif os.path.isfile(json_path):
                self.load_create_ml_json_by_filename(json_path, file_path)

        else:
            xml_path = os.path.splitext(file_path)[0] + XML_EXT
            txt_path = os.path.splitext(file_path)[0] + TXT_EXT
            if os.path.isfile(xml_path):
                self.load_pascal_xml_by_filename(xml_path)
            elif os.path.isfile(txt_path):
                self.load_yolo_txt_by_filename(txt_path)

    def resizeEvent(self, event):
        if self.canvas and not self.image.isNull() \
                and self.zoom_mode != self.MANUAL_ZOOM:
            self.adjust_scale()
        # Update preview display when window is resized
        self.update_preview_display()
        super(MainWindow, self).resizeEvent(event)

    def paint_canvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoom_widget.value()
        self.canvas.label_font_size = int(
            0.02 * max(self.image.width(), self.image.height()))
        self.canvas.adjustSize()
        self.canvas.update()

    def adjust_scale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoom_mode]()
        self.zoom_widget.setValue(int(100 * value))

    def scale_fit_window(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scale_fit_width(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def closeEvent(self, event):
        if not self.may_continue():
            event.ignore()
        settings = self.settings
        # If it loads images from dir, don't load it at the beginning
        if self.dir_name is None:
            settings[SETTING_FILENAME] = self.file_path if self.file_path else ''
        else:
            settings[SETTING_FILENAME] = ''

        settings[SETTING_WIN_SIZE] = self.size()
        settings[SETTING_WIN_POSE] = self.pos()
        settings[SETTING_WIN_STATE] = self.saveState()
        settings[SETTING_LINE_COLOR] = self.line_color
        settings[SETTING_FILL_COLOR] = self.fill_color
        settings[SETTING_RECENT_FILES] = self.recent_files
        settings[SETTING_ADVANCE_MODE] = not self._beginner
        # if self.default_save_dir and os.path.exists(self.default_save_dir):
        #     settings[SETTING_SAVE_DIR] = ustr(self.default_save_dir)
        # else:
        settings[SETTING_SAVE_DIR] = ''

        if self.last_open_dir and os.path.exists(self.last_open_dir):
            settings[SETTING_LAST_OPEN_DIR] = self.last_open_dir
        else:
            settings[SETTING_LAST_OPEN_DIR] = ''

        if self.last_model_dir and os.path.exists(self.last_model_dir):
            settings[SETTING_LAST_MODEL_DIR] = self.last_model_dir
        else:
            settings[SETTING_LAST_MODEL_DIR] = ''

        settings[SETTING_AUTO_SAVE] = self.auto_saving.isChecked()
        settings[SETTING_SINGLE_CLASS] = self.single_class_mode.isChecked()
        settings[SETTING_PAINT_LABEL] = self.display_label_option.isChecked()
        settings[SETTING_DRAW_SQUARE] = self.draw_squares_option.isChecked()
        settings[SETTING_DRAW_TWO_CLICKS] = self.draw_two_clicks_option.isChecked()
        settings[SETTING_LABEL_FILE_FORMAT] = self.label_file_format
        settings.save()

    def load_recent(self, filename):
        if self.may_continue():
            self.load_file(filename)

    def scan_all_images(self, folder_path):
        extensions = ['.%s' % fmt.data().decode("ascii").lower()
                      for fmt in QImageReader.supportedImageFormats()]
        images = []

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relative_path = os.path.join(root, file)
                    path = ustr(os.path.abspath(relative_path))
                    images.append(path)
        natural_sort(images, key=lambda x: x.lower())
        return images

    def change_save_dir_dialog(self, _value=False):
        if self.default_save_dir is not None:
            path = ustr(self.default_save_dir)
        else:
            path = '.'

        dir_path = ustr(QFileDialog.getExistingDirectory(self,
                                                         '%s - Save annotations to the directory' % __appname__, path,
                                                         QFileDialog.ShowDirsOnly
                                                         | QFileDialog.DontResolveSymlinks))

        if dir_path is not None and len(dir_path) > 1:
            self.default_save_dir = dir_path
            try:
                if hasattr(self, "video_output_dir_edit"):
                    self.video_output_dir_edit.setText(self.default_save_dir)
            except Exception:
                pass

        self.statusBar().showMessage('%s . Annotation will be saved to %s' %
                                     ('Change saved folder', self.default_save_dir))
        self.statusBar().show()

    def open_annotation_dialog(self, _value=False):
        if self.file_path is None:
            self.statusBar().showMessage('Please select image first')
            self.statusBar().show()
            return

        path = os.path.dirname(ustr(self.file_path)) \
            if self.file_path else '.'
        if self.label_file_format == LabelFileFormat.PASCAL_VOC:
            filters = "Open Annotation XML file (%s)" % ' '.join(['*.xml'])
            filename = ustr(
                QFileDialog.getOpenFileName(
                    self, '%s - Choose a xml file' %
                          __appname__, path, filters))
            if filename:
                if isinstance(filename, (tuple, list)):
                    filename = filename[0]
            self.load_pascal_xml_by_filename(filename)

    def open_dir_dialog(self, _value=False, dir_path=None, silent=False):
        if not self.may_continue():
            return

        prev_last_open_dir = self.last_open_dir
        default_open_dir_path = dir_path if dir_path else '.'
        if self.last_open_dir and os.path.exists(self.last_open_dir):
            default_open_dir_path = self.last_open_dir
        else:
            default_open_dir_path = os.path.dirname(
                self.file_path) if self.file_path else '.'

        if silent != True:
            target_dir_path = ustr(QFileDialog.getExistingDirectory(self,
                                                                    '%s - Open Directory' % __appname__,
                                                                    default_open_dir_path,
                                                                    QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))
        else:
            target_dir_path = ustr(default_open_dir_path)

        # If save dir was following the previous opened dir, keep it in sync to avoid
        # reading labels from the wrong folder during filtering.
        try:
            if self.default_save_dir and prev_last_open_dir \
                    and os.path.abspath(self.default_save_dir) == os.path.abspath(prev_last_open_dir):
                self.default_save_dir = target_dir_path
        except Exception:
            pass

        self._sync_video_dirs_to_data_dir(target_dir_path)

        self._refresh_dataset_class_source(
            target_dir_path, show_warnings=(
                    silent != True))

        self.import_dir_images(target_dir_path)
        # #print(1)

    def import_dir_images(self, dir_path):
        if not self.may_continue() or not dir_path:
            return

        self.last_open_dir = dir_path
        self.dir_name = dir_path
        self.file_path = None
        self.file_list_widget.clear()
        self._image_label_cache = {}
        self.pre_img_txt = []
        self.pre_img_seg = []
        self.pre_error_img_txt = []
        self.detection_results = {}
        self.m_img_list_all = self.scan_all_images(dir_path)
        self.m_img_list = list(self.m_img_list_all)
        active_filter = self._selected_class_filter_set if self._selected_class_filter_set is not None else self._selected_class_filter
        ok = self._apply_image_list_filter(
            keep_current=False,
            selected_filter=active_filter,
            allow_empty=(active_filter is None)
        )
        if not ok:
            if self._last_filter_cancelled:
                QMessageBox.information(self, "提示", "已取消筛选")
            else:
                display = ",".join(
                    sorted(
                        self._active_class_filter_set(active_filter) or set()))
                QMessageBox.information(
                    self, "提示", f"数据集中没有类别：{display or '（空）'}")
            self._selected_class_filter = None
            self._selected_class_filter_set = None
            if hasattr(self, "class_filter_combo"):
                self.class_filter_combo.blockSignals(True)
                try:
                    self.class_filter_combo.setCurrentText("全部")
                finally:
                    self.class_filter_combo.blockSignals(False)
            self._apply_image_list_filter(
                keep_current=False,
                selected_filter=None,
                allow_empty=True)
        # #print(1)

    def _save_current_image_to_filter_dir(self):
        if not getattr(
                self,
                "box_contour_checkbox",
                None) or not self.box_contour_checkbox.isChecked():
            return False

        save_dir = getattr(self, "_image_filter_save_dir", None)
        if not save_dir:
            QMessageBox.information(self, "提示", "请先在“图片筛选模式”中选择保存目录")
            return False

        src_path = getattr(self, "file_path", None)
        if not src_path or not os.path.isfile(src_path):
            QMessageBox.information(self, "提示", "当前没有可保存的图片文件")
            return False

        base_name = os.path.basename(src_path)
        name, ext = os.path.splitext(base_name)
        dst_path = os.path.join(save_dir, base_name)
        suffix = 1
        while os.path.exists(dst_path):
            dst_path = os.path.join(save_dir, f"{name}_{suffix}{ext}")
            suffix += 1

        try:
            _is_move = getattr(self, "_filter_move_combobox",
                               None) and self._filter_move_combobox.currentText() == "移动到目录"
            if _is_move:
                shutil.move(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败：{e}")
            return False

        if getattr(self, "_image_filter_sync_txt", False):
            src_txt = os.path.splitext(src_path)[0] + ".txt"
            if os.path.isfile(src_txt):
                dst_txt = os.path.splitext(dst_path)[0] + ".txt"
                try:
                    if _is_move:
                        shutil.move(src_txt, dst_txt)
                    else:
                        shutil.copy2(src_txt, dst_txt)
                except Exception as e:
                    # 仅提示，不影响图片已保存的结果
                    QMessageBox.warning(self, "提示", f"图片已保存，但同步txt失败：{e}")

        self.statusBar().showMessage(f"已保存图片: {dst_path}", 3000)
        return True

    def _on_filter_space_pressed(self):
        """空格键处理：SAM穿透模式 / 图片筛选保存。"""
        if isinstance(QApplication.focusWidget(), (QLineEdit, QTextEdit)):
            return
        if getattr(self.canvas, "sam_enabled", False):
            self.canvas.sam_passthrough = True
            self.canvas.update_sam_cursor(self._sam_cursor_color)
            self.canvas.setCursor(self.canvas._sam_cross_cursor)
            self.canvas.update()
            self.statusBar().showMessage("🔓 穿透模式：可穿透已有框触发 SAM", 3000)
            return
        if getattr(self, "box_contour_checkbox", None) and self.box_contour_checkbox.isChecked():
            if self._save_current_image_to_filter_dir():
                self.open_next_image(True)

    def verify_image(self, _value=False):
        # 图片筛选模式：空格键用于保存当前图片到选择目录
        if getattr(
                self,
                "box_contour_checkbox",
                None) and self.box_contour_checkbox.isChecked():
            self._save_current_image_to_filter_dir()
            return

        # Proceeding next image without dialog if having any label
        if self.file_path is not None:
            try:
                self.label_file.toggle_verify()
            except AttributeError:
                # If the labelling file does not exist yet, create if and
                # re-save it with the verified attribute.
                self.save_file()
                if self.label_file is not None:
                    self.label_file.toggle_verify()
                else:
                    return

            self.canvas.verified = self.label_file.verified
            self.paint_canvas()
            self.save_file()

    def open_prev_image(self, _value=False):
        # Proceeding prev image without dialog if having any label
        if self.auto_saving.isChecked():
            if self.default_save_dir is not None:
                if self.dirty is True:
                    self.save_file()
            else:
                self.change_save_dir_dialog()
                return

        if not self.may_continue():
            return

        if self.img_count <= 0:
            return

        if self.file_path is None:
            if self.m_img_list and self.img_count > 0:
                self.cur_img_idx = self.img_count - 1
                self.load_file(self.m_img_list[self.cur_img_idx])
            return

        if self.file_path not in self.m_img_list:
            if self.m_img_list and self.img_count > 0:
                self.cur_img_idx = self.img_count - 1
                self.load_file(self.m_img_list[self.cur_img_idx])
            return

        if self.cur_img_idx - 1 >= 0:
            self.cur_img_idx -= 1
            filename = self.m_img_list[self.cur_img_idx]
            if filename:
                self.load_file(filename)

    def open_next_image(self, _value=False):
        # Proceeding prev image without dialog if having any label
        if self.auto_saving.isChecked():
            if self.default_save_dir is not None:
                if self.dirty is True:
                    self.save_file()
            else:
                self.change_save_dir_dialog()
                return

        if not self.may_continue():
            return

        if self.img_count <= 0:
            return

        filename = None
        if self.file_path is None or (self.file_path not in self.m_img_list):
            filename = self.m_img_list[0]
            self.cur_img_idx = 0
        else:
            if self.cur_img_idx + 1 < self.img_count:
                self.cur_img_idx += 1
                filename = self.m_img_list[self.cur_img_idx]

        if filename:
            self.load_file(filename)

    def open_video_dialog(self, _value=False):
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            self.last_open_dir if getattr(
                self, "last_open_dir", None) else ".",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv);;All Files (*.*)",
        )
        if not video_path:
            return
        ok = self.video_player.openVideo(video_path)
        if not ok:
            self.error_message("打开视频失败", f"无法打开: {video_path}")
            return
        self.video_path_label.setText(video_path)
        try:
            self.last_open_dir = os.path.dirname(video_path)
        except Exception:
            pass
        try:
            self._sync_video_dirs_to_data_dir(self.last_open_dir)
        except Exception:
            pass

        # 填充视频信息
        try:
            fps = self.video_player.fps()
            dur = self.video_player.durationSeconds()
            fc = self.video_player.frameCount()
            w = self.video_player.width()
            h = self.video_player.height()
            self.video_info_res_label.setText(f"分辨率: {w} x {h}")
            self.video_info_fps_label.setText(f"帧率: {fps:.2f} fps")
            self.video_info_frames_label.setText(f"总帧数: {fc}")
            if dur > 0:
                m, s = divmod(int(dur), 60)
                hh, mm = divmod(m, 60)
                self.video_info_dur_label.setText(
                    f"时长: {hh:02d}:{mm:02d}:{s:02d}")
            else:
                self.video_info_dur_label.setText("时长: --")
        except Exception:
            pass

        # 更新滑动条范围（精确到0.01秒 = dur*100）
        try:
            if dur and dur > 0:
                max_val = int(dur * 100)
                self.video_start_slider.blockSignals(True)
                self.video_end_slider.blockSignals(True)
                self.video_start_slider.setRange(0, max_val)
                self.video_start_slider.setValue(0)
                self.video_end_slider.setRange(0, max_val)
                self.video_end_slider.setValue(max_val)
                self.video_start_slider.blockSignals(False)
                self.video_end_slider.blockSignals(False)
                self._update_slider_labels()
        except Exception:
            pass
        try:
            if fps and fps > 0:
                if self.video_target_fps_spin.value() <= 0:
                    self.video_target_fps_spin.setValue(1.0)
        except Exception:
            pass

    def choose_video_output_dir(self):
        # Bound to default_save_dir (annotation save dir)
        self.change_save_dir_dialog()

    @staticmethod
    def _format_slider_seconds(sec: float) -> str:
        m, s = divmod(int(sec), 60)
        cs = int((sec - int(sec)) * 100)
        return f"{m:02d}:{s:02d}.{cs:02d}"

    def _update_slider_labels(self):
        ss = self.video_start_slider.value() / 100.0
        se = self.video_end_slider.value() / 100.0
        self.video_start_label.setText(self._format_slider_seconds(ss))
        self.video_end_label.setText(self._format_slider_seconds(se))

    def _on_start_slider_changed(self, val):
        if self.video_end_slider.value() < val:
            self.video_end_slider.blockSignals(True)
            self.video_end_slider.setValue(val)
            self.video_end_slider.blockSignals(False)
        self._update_slider_labels()

    def _on_end_slider_changed(self, val):
        if self.video_start_slider.value() > val:
            self.video_start_slider.blockSignals(True)
            self.video_start_slider.setValue(val)
            self.video_start_slider.blockSignals(False)
        self._update_slider_labels()

    def _sync_video_output_dir_with_default(self):
        try:
            if getattr(self, "default_save_dir", None):
                self.video_output_dir_edit.setText(self.default_save_dir)
        except Exception:
            pass

    def _sync_video_dirs_to_data_dir(self, data_dir: str):
        """
        Bind video output directory to current data directory, and keep default_save_dir consistent.
        """
        if not data_dir:
            return
        try:
            self.last_open_dir = data_dir
        except Exception:
            pass
        try:
            self.default_save_dir = data_dir
        except Exception:
            pass
        try:
            if hasattr(self, "video_output_dir_edit"):
                self.video_output_dir_edit.setText(data_dir)
        except Exception:
            pass

    def start_video_process(self):
        if self._video_thread is not None:
            self.error_message("任务进行中", "已有视频任务正在运行，请等待完成。")
            return

        video_path = self.video_player.videoPath()
        if not video_path or not os.path.exists(video_path):
            self.error_message("未选择视频", "请先选择一个视频文件。")
            return

        # Bind output dir to current data dir (open_file/open_dir/video dir),
        # and keep default_save_dir same.
        data_dir = getattr(self, "last_open_dir", None) or ""
        if not data_dir:
            try:
                data_dir = os.path.dirname(video_path)
            except Exception:
                data_dir = ""
        if not data_dir:
            self.error_message("未设置数据目录", "请先打开图片目录/图片，或重新选择视频。")
            return
        self._sync_video_dirs_to_data_dir(data_dir)
        output_dir = data_dir

        mode = "fps" if self.video_mode_fps.isChecked() else "interval"
        target_fps = float(self.video_target_fps_spin.value())
        interval_s = float(self.video_interval_spin.value())
        start_s = self.video_start_slider.value() / 100.0
        end_s = self.video_end_slider.value() / 100.0

        self.video_progress.setValue(0)
        self.video_status.setText("准备开始...")
        self.video_run_btn.setEnabled(False)
        self.video_detect_btn.setEnabled(False)
        self.video_stop_btn.setEnabled(True)

        self._video_thread = QThread(self)
        self._video_worker = VideoProcessWorker(
            video_path=video_path,
            output_dir=output_dir,
            mode=mode,
            target_fps=target_fps,
            interval_s=interval_s,
            start_s=start_s,
            end_s=end_s,
        )
        self._video_worker.moveToThread(self._video_thread)
        self._video_thread.started.connect(self._video_worker.run)
        self._video_worker.progress.connect(self._on_video_worker_progress)
        self._video_worker.finished.connect(self._on_video_worker_finished)
        self._video_worker.failed.connect(self._on_video_worker_failed)
        self._video_thread.start()

    def _cleanup_video_worker(self):
        try:
            if self._video_thread is not None:
                self._video_thread.quit()
                self._video_thread.wait(3000)
        except Exception:
            pass
        self._video_thread = None
        self._video_worker = None
        try:
            self.video_run_btn.setEnabled(True)
            self.video_stop_btn.setEnabled(False)
        except Exception:
            pass

    def _on_video_worker_progress(self, percent: int, message: str):
        try:
            self.video_progress.setValue(int(percent))
        except Exception:
            pass
        try:
            self.video_status.setText(str(message))
        except Exception:
            pass

    def _on_video_worker_finished(self, result: dict):
        frames_dir = result.get("frames_dir", "")
        msg = f"抽帧完成。\n抽帧目录: {frames_dir}"
        self.video_status.setText(msg)
        try:
            QMessageBox.information(self, "视频抽帧", msg)
        except Exception:
            pass

        self._last_extract_result = result
        self._cleanup_video_worker()
        try:
            self.video_detect_btn.setEnabled(True)
        except Exception:
            pass

    def _on_video_worker_failed(self, error: str):
        self._cleanup_video_worker()
        try:
            if getattr(self, "_last_extract_result", None):
                self.video_detect_btn.setEnabled(True)
        except Exception:
            pass
        self.video_status.setText(f"失败: {error}")
        if str(error) == "用户停止":
            return
        try:
            self.error_message("视频任务失败", str(error))
        except Exception:
            pass

    def start_video_detect(self):
        if self._video_thread is not None:
            self.error_message("任务进行中", "已有视频任务正在运行，请等待完成。")
            return
        result = getattr(self, "_last_extract_result", None)
        if not result:
            self.error_message("未抽帧", "请先完成抽帧再检测。")
            return

        # Ensure model is loaded
        if getattr(self, "yolo_model", None) is None:
            reply = QMessageBox.question(
                self,
                "未加载模型",
                "视频检测需要先加载模型，是否现在加载？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                return
            self.openModel()
            if getattr(self, "yolo_model", None) is None:
                self.error_message("未加载模型", "已取消或加载失败，无法检测。")
                return

        # Compute default export video path
        video_path = result.get("video", "")
        default_export_path = ""
        if video_path:
            base = os.path.splitext(video_path)[0]
            default_export_path = base + "_detect.mp4"

        prev_defaults = getattr(self, "_video_detect_defaults", {}) or {}
        merged_defaults = dict(prev_defaults)
        if default_export_path:
            merged_defaults["export_path"] = default_export_path

        dlg = VideoDetectDialog(
            self,
            defaults=merged_defaults,
            classes=getattr(
                self.yolo_model,
                "classes",
                []) if getattr(
                self,
                "yolo_model",
                None) is not None else [],
        )
        if dlg.exec_() != QDialog.Accepted:
            return
        opts = dlg.values()
        self._last_video_detect_opts = opts
        self._video_detect_defaults = {
            "use_zh": opts.get("use_zh", False),
            "show_conf": opts.get("show_conf", True),
            "label_map_text": opts.get("label_map_text", ""),
            "label_map": opts.get("label_map", {}) or {},
            "export_video": opts.get("export_video", False),
            "export_fps": opts.get("export_fps", 0.0),
            "export_path": opts.get("export_path", ""),
        }
        if not opts.get("do_detect", True):
            return

        self._start_video_detect_from_extract(result, opts)

    def stop_video_process(self):
        if self._video_worker is not None:
            self._video_worker._stopped = True
        self._cleanup_video_worker()
        self.video_status.setText("已停止")

    def _start_video_detect_from_extract(
            self, extract_result: dict, detect_opts: dict):
        if self._video_thread is not None:
            return
        self.video_progress.setValue(0)
        self.video_status.setText("开始检测...")
        self.video_run_btn.setEnabled(False)
        self.video_detect_btn.setEnabled(False)
        self.video_stop_btn.setEnabled(True)

        self._video_thread = QThread(self)
        self._video_worker = VideoDetectWorker(
            video_path=extract_result.get("video", ""),
            output_dir=extract_result.get("output_dir", self.default_save_dir),
            base=extract_result.get("base", ""),
            ts=extract_result.get(
                "ts", datetime.datetime.now().strftime("%Y%m%d_%H%M%S")),
            frames=extract_result.get("frames", []),
            width=extract_result.get("width", 0),
            height=extract_result.get("height", 0),
            src_fps=extract_result.get("src_fps", 0.0),
            yolo_model=getattr(self, "yolo_model", None),
            use_zh=bool(detect_opts.get("use_zh", False)),
            show_conf=bool(detect_opts.get("show_conf", True)),
            label_map=detect_opts.get("label_map", {}) or {},
            export_video=bool(detect_opts.get("export_video", False)),
            export_video_path=str(detect_opts.get("export_path", "") or ""),
            export_fps=float(detect_opts.get("export_fps", 0.0) or 0.0),
        )
        self._video_worker.moveToThread(self._video_thread)
        self._video_thread.started.connect(self._video_worker.run)
        self._video_worker.progress.connect(self._on_video_worker_progress)
        self._video_worker.finished.connect(self._on_video_detect_finished)
        self._video_worker.failed.connect(self._on_video_worker_failed)
        self._video_thread.start()

    def _on_video_detect_finished(self, result: dict):
        self._cleanup_video_worker()
        try:
            self.video_detect_btn.setEnabled(True)
        except Exception:
            pass
        export_video = result.get("export_video", "")
        msg = "检测完成。"
        if export_video:
            msg += f"\n导出视频: {export_video}"
        self.video_status.setText(msg)
        try:
            QMessageBox.information(self, "视频检测", msg)
        except Exception:
            pass

        # 导出视频自动加载到播放器
        if export_video and os.path.exists(export_video):
            try:
                self.video_player.openVideo(export_video)
                self.video_path_label.setText(export_video)
                # 更新视频信息和滑动条
                fps = self.video_player.fps()
                dur = self.video_player.durationSeconds()
                w = self.video_player.width()
                h = self.video_player.height()
                fc = self.video_player.frameCount()
                try:
                    self.video_info_res_label.setText(f"分辨率: {w} x {h}")
                    self.video_info_fps_label.setText(f"帧率: {fps:.2f} fps")
                    self.video_info_frames_label.setText(f"总帧数: {fc}")
                    if dur > 0:
                        m, s = divmod(int(dur), 60)
                        hh, mm = divmod(m, 60)
                        self.video_info_dur_label.setText(
                            f"时长: {hh:02d}:{mm:02d}:{s:02d}")
                except Exception:
                    pass
                if dur > 0:
                    max_val = int(dur * 100)
                    self.video_start_slider.setRange(0, max_val)
                    self.video_start_slider.setValue(0)
                    self.video_end_slider.setRange(0, max_val)
                    self.video_end_slider.setValue(max_val)
                    self._update_slider_labels()
            except Exception:
                pass

        # Closed loop: write into self.detection_results and load into image
        # list for analysis dock.
        try:
            self._ingest_video_detections_into_app(
                result, detect_opts=getattr(
                    self, "_last_video_detect_opts", None))
        except Exception:
            pass

    def _ingest_video_detections_into_app(
            self, detect_result: dict, detect_opts: dict = None):
        dets = detect_result.get("detections", []) or []
        if not dets:
            return
        detect_opts = detect_opts or {}
        try:
            self._active_label_map = detect_opts.get("label_map", {}) or {}
            self._active_use_zh = bool(detect_opts.get("use_zh", False))
        except Exception:
            self._active_label_map = {}
            self._active_use_zh = False

        # Ensure frames are loaded as current dataset so user can browse like
        # "video detection".
        first_frame = dets[0].get("frame")
        if first_frame:
            try:
                first_frame = os.path.abspath(first_frame)
            except Exception:
                pass
        if first_frame and os.path.exists(first_frame):
            frames_dir = os.path.dirname(first_frame)
            try:
                self.open_dir_dialog(dir_path=frames_dir, silent=True)
            except Exception:
                pass

        # 获取图像尺寸用于 YOLO 归一化
        img_w = img_h = 0
        for r in dets:
            fp = r.get("frame")
            if fp and os.path.exists(fp):
                try:
                    test_img = cv2.imdecode(
                        np.fromfile(
                            fp,
                            dtype=np.uint8),
                        cv2.IMREAD_COLOR)
                    if test_img is not None:
                        img_h, img_w = test_img.shape[:2]
                    break
                except Exception:
                    pass

        for r in dets:
            fp = r.get("frame")
            if not fp or not os.path.exists(fp):
                continue
            try:
                fp = os.path.abspath(fp)
            except Exception:
                pass
            boxes, scores, class_ids = r.get("result", [[], [], []])
            predicted_boxes = []
            yolo_entries = []
            for i in range(len(boxes)):
                try:
                    cid = int(class_ids[i])
                except Exception:
                    cid = -1
                try:
                    conf = float(scores[i])
                except Exception:
                    conf = 0.0
                try:
                    name = self.yolo_model.classes[cid] if (
                            cid >= 0 and cid < len(self.yolo_model.classes)) else str(cid)
                except Exception:
                    name = str(cid)
                try:
                    x1, y1, x2, y2 = boxes[i]
                except Exception:
                    continue
                predicted_boxes.append(([x1, y1, x2, y2], str(name)))
                # YOLO 归一化格式: cx, cy, w, h, class_id
                if img_w > 0 and img_h > 0:
                    cx = (x1 + x2) / 2.0 / img_w
                    cy = (y1 + y2) / 2.0 / img_h
                    w = (x2 - x1) / img_w
                    h = (y2 - y1) / img_h
                else:
                    cx, cy, w, h = 0.0, 0.0, 0.0, 0.0
                yolo_entries.append([cx, cy, w, h, cid])

            # 填充 pre_img_txt 以支持 "save all pre labels" 按钮
            if yolo_entries:
                yolo_entries.append(fp)
                self.pre_img_txt.append(yolo_entries)

            # No ground-truth for video frames: treat all predictions as FP so
            # preview dock still works.
            analysis_result = {
                "true_positives": [],
                "false_positives": [{"box": b, "iou": 0.0, "label": lbl} for b, lbl in predicted_boxes],
                "false_negatives": [],
            }

            self.detection_results[fp] = {
                "image": fp,
                "result": [boxes, scores, class_ids],
                "stats": analysis_result,
            }

        # Make UI show first frame's analysis preview.
        try:
            if first_frame and os.path.exists(first_frame):
                try:
                    # Ensure analysis dock is visible for video results.
                    self.results_dock.show()
                except Exception:
                    pass
                self.load_file(first_frame)
                self.update_preview_display()
        except Exception:
            pass

        # Sync video player to show "detection view" by playing extracted
        # frames as a sequence.
        try:
            seq = []
            for r in dets:
                fp = r.get("frame")
                if not fp or not os.path.exists(fp):
                    continue
                try:
                    seq.append(os.path.abspath(fp))
                except Exception:
                    seq.append(fp)
            if seq:
                use_zh = bool(detect_opts.get("use_zh", False))
                show_conf = bool(detect_opts.get("show_conf", True))
                label_map = detect_opts.get("label_map", {}) or {}
                classes = getattr(self.yolo_model, "classes", [])
                images = []
                for r in dets:
                    fp = r.get("frame")
                    if not fp or not os.path.exists(fp):
                        continue
                    try:
                        fp = os.path.abspath(fp)
                    except Exception:
                        pass
                    try:
                        # 优先使用 infer() 返回的已渲染图像（含 boxes/masks）
                        fr = r.get("rendered")
                        if fr is None:
                            fr = cv2.imdecode(
                                np.fromfile(
                                    fp,
                                    dtype=np.uint8),
                                cv2.IMREAD_COLOR)
                            boxes, scores, class_ids = r.get(
                                "result", [[], [], []])
                            fr = VideoProcessWorker._draw_boxes(
                                fr,
                                boxes=boxes,
                                scores=scores,
                                class_ids=class_ids,
                                classes=classes,
                                label_map=label_map,
                                use_zh=use_zh,
                                show_conf=show_conf,
                            )
                        images.append(_cv_bgr_to_qimage(fr))
                    except Exception:
                        continue
                if images:
                    self.video_player.openQImageSequence(
                        images, fps=self.video_player.fps() or 10.0)
                    self.video_player.play()
        except Exception:
            pass

        # 标记视频检测上下文，使得 load_file 可自动加载检测框到画布
        self._video_detect_context = True

    # moxingjiazai
    def _load_model_from_path(self, model_path):
        if not model_path or not os.path.exists(model_path):
            return False
        self.last_model_dir = os.path.dirname(os.path.abspath(model_path))

        # 在 ONNX Runtime 初始化 CUDA 之前，先让 torch 加载 CUDA 库，
        # 避免后续加载 SAM 模型时 cuDNN DLL 冲突
        try:
            import torch
            if torch.cuda.is_available():
                _ = torch.zeros(1, device='cuda')
        except Exception:
            pass

        progress = QProgressDialog(
            "正在加载 YOLO 模型...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)
        progress.show()
        QApplication.processEvents()
        try:
            confidence_thres = self.confidence_spinbox.value()
            iou_thres = self.iou_spinbox.value()
            selected_model_type = self.model_type_combobox.currentText()
            if selected_model_type == "V8":
                self.yolo_model = YOLOv8(model_path, confidence_thres, iou_thres)
            elif selected_model_type == "V8_Seg":
                self.yolo_model = YOLOv8_Seg(model_path, confidence_thres, iou_thres)
            else:
                self.yolo_model = YOLOv7(model_path, confidence_thres, iou_thres)
            model_output_count = len(self.yolo_model.model_outputs)
            if model_output_count > 0:
                self.statusBar().showMessage("模型加载完成")
                self.statusBar().show()
                if hasattr(self, "sync_model_classes_button"):
                    self.sync_model_classes_button.setEnabled(True)
            else:
                self.error_message("模型加载失败", "无法获取模型输出层")
                return False
        except Exception as e:
            self.error_message("模型加载失败", str(e))
            return False
        finally:
            progress.close()
        return True

    def _on_model_select_changed(self, filename):
        if not filename or filename.startswith("("):
            return
        model_dir = os.path.join(os.path.dirname(__file__), "model")
        model_path = os.path.join(model_dir, filename)
        if os.path.exists(model_path):
            self._load_model_from_path(model_path)

    # --- 模型列表热加载 ---
    def _refresh_sam_models(self):
        """重新扫描 weights/ 目录，刷新 SAM 模型下拉列表"""
        combo = self.sam_model_combo
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("")  # 空白初始项
        if os.path.isdir(self._sam_weights_dir):
            for f in sorted(os.listdir(self._sam_weights_dir)):
                if f.endswith(".pt"):
                    combo.addItem(f)
        # 恢复之前选中的项
        idx = combo.findText(current)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def _refresh_onnx_models(self):
        """重新扫描 model/ 目录，刷新 ONNX 模型下拉列表"""
        combo = self.model_select_combobox
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("")  # 空白初始项
        _model_dir = os.path.join(os.path.dirname(__file__), "model")
        if os.path.isdir(_model_dir):
            for f in sorted(os.listdir(_model_dir)):
                if f.endswith(".onnx"):
                    combo.addItem(f)
        # 恢复之前选中的项
        idx = combo.findText(current)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    # --- SAM 交互式分割 ---
    def _on_sam_model_selected(self, filename):
        """用户从下拉框选择了 SAM 模型 → 加载"""
        if not filename or filename.startswith("(无"):
            return
        checkpoint = os.path.join(self._sam_weights_dir, filename)
        if not os.path.exists(checkpoint):
            return

        # 进度条
        progress = QProgressDialog(
            f"正在加载 {filename} ...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)
        progress.show()
        QApplication.processEvents()

        try:
            self.sam_client.load_model(checkpoint)
        except Exception as e:
            progress.close()
            self.error_message("SAM模型加载失败", f"错误: {str(e)}")
            return
        finally:
            progress.close()

        self.sam_checkbox.setEnabled(True)

        # 更新文字提示可用性
        mt = self.sam_client.model_type or "?"
        type_names = {"sam1": "SAM 1", "sam2": "SAM 2", "sam3": "SAM 3"}
        type_label = type_names.get(mt, mt.upper())
        text_support = "支持文字提示" if self.sam_client.supports_text else "不支持文字提示"
        QMessageBox.information(
            self, "SAM模型",
            f"SAM 模型已加载：{filename}\n识别版本：{type_label}\n{text_support}")

        # 非 SAM3 时禁用文字提示选项
        self._update_sam_text_prompt_availability()
        # 如果当前有图片，立刻编码
        self._sam_encode_current_image()

    def _on_sam_toggle_shortcut(self):
        """快捷键 S 切换 SAM 模式"""
        if not self.sam_client.is_loaded:
            self.error_message("未加载SAM模型",
                               "请先从下拉框选择 weights/ 下的 SAM 模型文件")
            return
        self.sam_checkbox.toggle()

    def _on_sam_toggled(self, checked):
        """SAM 模式开关"""
        if checked:
            if not self.sam_client.is_loaded:
                self.error_message("未加载SAM模型",
                                   "请先从下拉框选择 weights/ 下的 SAM 模型文件")
                self.sam_checkbox.blockSignals(True)
                self.sam_checkbox.setChecked(False)
                self.sam_checkbox.blockSignals(False)
                return
            self.canvas.sam_enabled = True
            self.canvas.sam_client = self.sam_client
            mode_text = self.sam_prompt_combo.currentText()
            if mode_text == "框提示":
                self.canvas.sam_mode = "box"
            elif mode_text == "文字提示":
                self.canvas.sam_mode = "text"
            else:
                self.canvas.sam_mode = "point"
            self._sam_encode_current_image()
        else:
            self.canvas.sam_enabled = False
            self.canvas.sam_drag_start = None
            self.canvas.sam_drag_rect = None
            self.canvas.update()

    def _sam_encode_if_needed(self):
        """延迟编码：切图后标记需要编码时执行"""
        if self._sam_needs_encode and self.canvas.sam_enabled:
            self._sam_encode_current_image()

    def _sam_encode_current_image(self):
        """对当前图片执行 SAM 特征编码"""
        self._sam_needs_encode = False
        if (not self.sam_client.is_loaded
                or not self.file_path
                or not os.path.exists(self.file_path)):
            return
        try:
            img_bgr = cv2.imdecode(
                np.fromfile(self.file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            img_bgr = cv2.imread(self.file_path)
        if img_bgr is not None:
            self.sam_client.set_image(img_bgr)

    def _set_sam_text_visible(self, visible):
        """显隐 SAM 文字提示控件（输入框+按钮）"""
        self.sam_text_label.setVisible(visible)
        self.sam_text_input.setVisible(visible)
        self.sam_text_btn.setVisible(visible)

    def _update_sam_text_prompt_availability(self):
        """根据模型类型启用/禁用文字提示选项"""
        idx = self.sam_prompt_combo.findText("文字提示")
        if idx >= 0:
            model = self.sam_prompt_combo.model()
            item = model.item(idx)
            if not self.sam_client.supports_text:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
                # 如果当前选中了文字提示，自动切到点提示
                if self.sam_prompt_combo.currentText() == "文字提示":
                    self.sam_prompt_combo.setCurrentIndex(0)
            else:
                item.setFlags(item.flags() | Qt.ItemIsEnabled)

    def _on_sam_prompt_mode_changed(self, text):
        """切换点提示/框提示/文字提示模式"""
        if text == "文字提示" and not self.sam_client.supports_text:
            # 非 SAM3 不支持文字提示，强制切回点提示
            self.sam_prompt_combo.setCurrentIndex(0)
            self.error_message("不支持文字提示",
                               "当前模型不支持文字提示（仅 SAM3 支持），请使用点提示或框提示")
            return
        if text == "框提示":
            self.canvas.sam_mode = "box"
            self._set_sam_text_visible(False)
        elif text == "文字提示":
            self.canvas.sam_mode = "text"
            self._set_sam_text_visible(True)
        else:
            self.canvas.sam_mode = "point"
            self._set_sam_text_visible(False)
        self.canvas.sam_drag_start = None
        self.canvas.sam_drag_rect = None
        self.canvas.update()

    def _choose_sam_cursor_color(self, _checked=False):
        color = QColorDialog.getColor(self._sam_cursor_color, self, "选择SAM光标颜色")
        if color.isValid():
            self._sam_cursor_color = color
            self.statusBar().showMessage("SAM: " + color.name(), 3000)
            self.canvas.update_sam_cursor(color)
            if self.canvas.sam_enabled:
                self.canvas.setCursor(self.canvas._sam_cross_cursor)
            self.sam_cursor_color_btn.setStyleSheet(
                "QToolButton { background-color: " + color.name() + "; border: 1px solid #999; min-width: 24px; min-height: 24px; border-radius: 4px; }"
            )

    def _on_sam_text_triggered(self):
        """文字提示推理"""
        prompt_text = self.sam_text_input.text().strip()
        if not prompt_text:
            return
        if not self.sam_client.is_loaded or not self.sam_client.current_image is not None:
            return
        # 解析提示词：逗号/空格/顿号分割
        import re
        text_list = [t.strip() for t in re.split(r'[,，、\s]+', prompt_text) if t.strip()]
        if not text_list:
            return
        self.statusBar().showMessage("\u6587\u5b57\u63d0\u793a\u63a8\u7406\u4e2d: " + str(text_list), 0)
        QApplication.processEvents()
        QApplication.processEvents()
        try:
            items = self.sam_client.predict_text(text_list)
        except Exception as e:
            self.statusBar().showMessage("\u6587\u5b57\u63d0\u793a\u9519\u8bef: " + str(e), 5000)
            return
        if items:
            self._on_sam_result_ready({"items": items})
            self.statusBar().showMessage(
                "\u6587\u5b57\u63d0\u793a\u5b8c\u6210: " + str(len(items)) + " \u4e2a\u76ee\u6807", 5000)
        else:
            self.statusBar().showMessage("\u6587\u5b57\u63d0\u793a: \u672a\u68c0\u6d4b\u5230\u76ee\u6807", 3000)

    def _on_sam_result_ready(self, result):
        """Canvas SAM 推理完成 → 直接挂载 Shape 到画布"""
        output_type = self.sam_output_combo.currentText()

        # 点提示/框提示模式下需要先显示再弹标签选择
        is_prompt_label = result.get("prompt_label") and not self.use_default_label_checkbox.isChecked()

        # 获取标签：优先用传入的 src_label，否则用默认标签
        if result.get("src_label"):
            default_label = result["src_label"]
        elif self.use_default_label_checkbox.isChecked() and self.default_label_text_line.text():
            default_label = self.default_label_text_line.text()
        else:
            default_label = "sam" if not is_prompt_label else "__sam_placeholder__"

        # 文字提示：多目标
        items = result.get("items")
        if items:
            mask_list = items
        else:
            mask_list = [(result.get("mask"), result.get("bbox"), default_label)]

        # 保存到 pre_img_seg（供 "save all pre labels" 导出）
        img_path = self.file_path
        seg_entry = []
        created_shapes = []
        # 获取图像尺寸用于归一化多边形坐标
        _sam_img = self.sam_client.current_image
        if _sam_img is not None:
            _sam_h, _sam_w = _sam_img.shape[:2]
        else:
            _sam_h = self.image_data.height()
            _sam_w = self.image_data.width()

        for mask, bbox, label in mask_list:
            if mask is None:
                continue
            if output_type == "目标框(BBox)" and bbox:
                x1, y1, x2, y2 = bbox
                shape = Shape(label=label, shape_type="rectangle")
                shape.add_point(QPointF(x1, y1))
                shape.add_point(QPointF(x2, y1))
                shape.add_point(QPointF(x2, y2))
                shape.add_point(QPointF(x1, y2))
                shape.close()
                shape._mask_for_label = mask  # 暂存 mask，对话框确定后再计算 cid
            else:
                pts = SamClient.mask_to_polygon(mask)
                if len(pts) < 3:
                    continue
                shape = Shape(label=label, shape_type="polygon")
                for x, y in pts:
                    shape.add_point(QPointF(x, y))
                shape.close()
                shape._pts_for_cid = pts  # 暂存多边形坐标，对话框确定后再计算 cid
                if not is_prompt_label:
                    # 非点提示模式：立即用当前标签收集 seg_entry
                    try:
                        if label in self.label_hist:
                            cid = self.label_hist.index(label)
                        else:
                            cid = int(label) if label.isdigit() else 0
                    except (ValueError, TypeError):
                        cid = 0
                    flat = [cid]
                    for x, y in pts:
                        flat.append(x / _sam_w)
                        flat.append(y / _sam_h)
                    seg_entry.append(flat)

            shape.is_sam = True  # 标记为 SAM 生成
            self.canvas.shapes.append(shape)
            self.add_label(shape)
            created_shapes.append(shape)

        # 点提示：先显示形状，再弹标签选择对话框
        if is_prompt_label and created_shapes:
            self.canvas.update()
            QApplication.processEvents()
            if len(self.label_hist) > 0:
                self.label_dialog = LabelDialog(parent=self, list_item=self.label_hist)
            result_label = self.label_dialog.pop_up(text=self.prev_label_text)

            if result_label is None:
                # 用户取消 — 移除刚添加的形状
                for sh in created_shapes:
                    if sh in self.canvas.shapes:
                        self.canvas.shapes.remove(sh)
                    item = self.shapes_to_items.pop(sh, None)
                    if item:
                        self.label_list.takeItem(self.label_list.row(item))
                        self.items_to_shapes.pop(item, None)
                self.canvas.update()
                return
            # 赋标签
            for sh in created_shapes:
                sh.label = result_label
                item = self.shapes_to_items.get(sh)
                if item:
                    item.setText(result_label)
                    item.setBackground(generate_color_by_text(result_label))
            self.prev_label_text = result_label

            # 点提示：对话框确定后构建 pre_img_seg 条目
            for sh in created_shapes:
                pts = getattr(sh, '_pts_for_cid', None)
                mask = getattr(sh, '_mask_for_label', None)
                if pts:
                    try:
                        if result_label in self.label_hist:
                            cid = self.label_hist.index(result_label)
                        else:
                            cid = int(result_label) if result_label.isdigit() else 0
                    except (ValueError, TypeError):
                        cid = 0
                    flat = [cid]
                    for x, y in pts:
                        flat.append(x / _sam_w)
                        flat.append(y / _sam_h)
                    seg_entry.append(flat)
            if seg_entry and img_path:
                seg_entry.append(img_path)
                replaced = False
                for i, item in enumerate(self.pre_img_seg):
                    if str(item[-1]) == img_path:
                        self.pre_img_seg[i] = seg_entry
                        replaced = True
                        break
                if not replaced:
                    self.pre_img_seg.append(seg_entry)

            self.canvas.update()
            self.set_dirty()
            self._sort_label_list_by_class()
            return

        # 非点提示模式（文字提示/框提示）：用原始标签写入 pre_img_seg
        if seg_entry and img_path:
            seg_entry.append(img_path)
            replaced = False
            for i, item in enumerate(self.pre_img_seg):
                if str(item[-1]) == img_path:
                    self.pre_img_seg[i] = seg_entry
                    replaced = True
                    break
            if not replaced:
                self.pre_img_seg.append(seg_entry)

        self.canvas.update()
        self.set_dirty()
        self._sort_label_list_by_class()

    def _batch_sam_text_detect(self):
        """批量 SAM3 文字提示检测 — 遍历目录所有图片执行文字推理"""
        prompt_text = self.sam_text_input.text().strip()
        if not prompt_text:
            return
        if not self.sam_client.is_loaded:
            self.error_message("未加载SAM模型", "请先加载SAM模型")
            return
        if not self.m_img_list:
            self.error_message("无图片", "请先打开包含图片的文件夹")
            return

        import re
        text_list = [t.strip() for t in re.split(r'[,，、\s]+', prompt_text) if t.strip()]
        if not text_list:
            return

        # 与 YOLO 批量检测一致：不自动加载框到画布
        self._video_detect_context = False

        output_is_bbox = self.sam_output_combo.currentText() == "目标框(BBox)"

        # 清空对应模式的数据
        if output_is_bbox:
            self.pre_img_txt = []
        else:
            self.pre_img_seg = []

        # 构建 label → class_id 映射（优先查 classes.txt）
        label_to_id = {}
        classes_txt_path = None
        if self.default_save_dir:
            ct = os.path.join(self.default_save_dir, "classes.txt")
            if os.path.isfile(ct):
                classes_txt_path = ct
        if not classes_txt_path and self.txt_path and os.path.isdir(self.txt_path):
            ct = os.path.join(self.txt_path, "classes.txt")
            if os.path.isfile(ct):
                classes_txt_path = ct
        if classes_txt_path:
            with open(classes_txt_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    label_to_id[line.rstrip()] = idx

        def _get_class_id(label_name):
            if label_name in label_to_id:
                return label_to_id[label_name]
            # fallback：尝试把 label 本身当 ID
            try:
                return int(label_name)
            except (ValueError, TypeError):
                return 0

        start_idx = self.cur_img_idx if self.batch_detect_from_current else 0
        target_images = self.m_img_list[start_idx:]
        total = len(target_images)

        progress = QProgressDialog(
            "正在批量 SAM 文字检测...", "取消", 0, total, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        processed = 0
        try:
            for i, image_path in enumerate(target_images):
                if progress.wasCanceled():
                    break
                progress.setValue(i)
                progress.setLabelText(
                    f"正在处理: {os.path.basename(image_path)} {start_idx + i + 1}/{len(self.m_img_list)}")
                QApplication.processEvents()

                if not os.path.exists(image_path):
                    continue

                try:
                    img_bgr = cv2.imdecode(
                        np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                except Exception:
                    img_bgr = cv2.imread(image_path)
                if img_bgr is None:
                    continue

                # 每张图重新编码特征
                self.sam_client.set_image(img_bgr)

                items = self.sam_client.predict_text(text_list)
                if not items:
                    continue

                h, w = img_bgr.shape[:2]

                is_current = (image_path == self.file_path)
                _cur_shapes = []

                if output_is_bbox:
                    # 目标框(BBox) → YOLO 格式 [class_id, cx, cy, w, h, image_path]
                    batch_entry = []
                    for mask, bbox, label in items:
                        if bbox is None:
                            continue
                        x1, y1, x2, y2 = bbox
                        cx = (x1 + x2) / 2.0 / w
                        cy = (y1 + y2) / 2.0 / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        cid = _get_class_id(label)
                        batch_entry.append([cid, cx, cy, bw, bh])
                        if is_current:
                            shape = Shape(label=label, shape_type="rectangle")
                            shape.add_point(QPointF(x1, y1))
                            shape.add_point(QPointF(x2, y1))
                            shape.add_point(QPointF(x2, y2))
                            shape.add_point(QPointF(x1, y2))
                            shape.close()
                            shape.is_sam = True
                            _cur_shapes.append(shape)
                    if batch_entry:
                        batch_entry.append(image_path)
                        self.pre_img_txt.append(batch_entry)
                else:
                    # 分割Mask(Polygon) → [class_id, x1, y1, x2, y2, ..., image_path]
                    batch_entry = []
                    for mask, bbox, label in items:
                        if mask is None:
                            continue
                        pts = SamClient.mask_to_polygon(mask)
                        if len(pts) < 3:
                            continue
                        cid = _get_class_id(label)
                        flat = [cid]
                        for x, y in pts:
                            flat.append(x / w)
                            flat.append(y / h)
                        batch_entry.append(flat)
                        if is_current:
                            shape = Shape(label=label, shape_type="polygon")
                            for x, y in pts:
                                shape.add_point(QPointF(x, y))
                            shape.close()
                            shape.is_sam = True
                            _cur_shapes.append(shape)
                    if batch_entry:
                        batch_entry.append(image_path)
                        self.pre_img_seg.append(batch_entry)

                if is_current and _cur_shapes:
                    self._sam_batch_current_shapes = _cur_shapes

                processed += 1

        finally:
            progress.close()

        # 将当前图片的检测结果加载到画布（支持 Ctrl+S 常规保存）
        _batch_shapes = getattr(self, '_sam_batch_current_shapes', None)
        if _batch_shapes:
            self.canvas.shapes.clear()
            self.items_to_shapes.clear()
            self.shapes_to_items.clear()
            self.label_list.clear()
            for shape in _batch_shapes:
                self.canvas.shapes.append(shape)
                self.add_label(shape)
            self._sam_batch_current_shapes = None
            self.canvas.update()
            self.set_dirty()

        QMessageBox.information(
            self, "批量 SAM 文字检测完成",
            f"共处理 {processed} 张图片\n"
            f"提示词: {prompt_text}\n"
            f"输出模式: {'目标框(BBox)' if output_is_bbox else '分割Mask(Polygon)'}\n\n"
            "当前图片检测结果已加载到画布，可按 Ctrl+S 保存\n"
            "使用「Save all pre labels」批量导出所有图片的标签")

    # --- SAM 结束 ---

    def _show_dataset_stats(self):
        import glob
        import datetime

        data_dir = getattr(self, "last_open_dir", None) or ""
        if not data_dir or not os.path.isdir(data_dir):
            self.error_message("\u7edf\u8ba1\u5931\u8d25",
                               "\u8bf7\u5148\u6253\u5f00\u5305\u542b\u56fe\u7247\u548c\u6807\u6ce8\u6587\u4ef6\u7684\u6570\u636e\u96c6\u76ee\u5f55")
            return

        class_names = {}
        for _dir in [data_dir, getattr(self, "default_save_dir", None)]:
            if not _dir:
                continue
            ct = os.path.join(_dir, "classes.txt")
            if not os.path.isfile(ct):
                continue
            with open(ct, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            cid = int(parts[0])
                            class_names.setdefault(cid, parts[1])
                        except ValueError:
                            pass
                    elif len(parts) == 1:
                        idx = len(class_names)
                        class_names[idx] = line

        txt_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
        txt_files = [f for f in txt_files if not os.path.basename(f) == "classes.txt"]

        total_images = len(txt_files)
        empty_files = 0
        failed_files = 0
        per_class = {}

        for fpath in txt_files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = [l.strip() for l in f if l.strip()]
                if not content:
                    empty_files += 1
                    continue
                for line in content:
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        cid = int(parts[0])
                    except ValueError:
                        continue
                    if cid not in per_class:
                        per_class[cid] = [0, set()]
                    per_class[cid][0] += 1
                    per_class[cid][1].add(fpath)
            except Exception:
                failed_files += 1

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # --- \u8f85\u52a9\u51fd\u6570\uff1a\u8ba1\u7b97\u5b57\u7b26\u4e32\u663e\u793a\u5bbd\u5ea6\uff08\u975eASCII\u5b57\u7b26\u7b972\u5bbd\u5ea6\uff09---
        def _display_width(s):
            w = 0
            for ch in s:
                w += 2 if ord(ch) > 127 else 1
            return w

        def _pad(s, target_w):
            cur = _display_width(s)
            return s + " " * max(0, target_w - cur)

        # --- \u6982\u89c8\u4fe1\u606f ---
        lines_out = [
            "                    ===== " + now_str + " =====",
            "\u6570\u636e\u96c6: " + data_dir,
            "\u603b\u6587\u4ef6\u6570: " + str(total_images) + "    \u7a7a\u6807\u6ce8\u6587\u4ef6: " + str(empty_files) + "    \u8bfb\u53d6\u5931\u8d25: " + str(failed_files),
            "",
        ]

        # --- \u52a8\u6001\u8ba1\u7b97\u5217\u5bbd ---
        id_w = max(8, _display_width("\u7c7b\u522bID"))
        name_w = max(16, _display_width("\u7c7b\u522b\u540d\u79f0"))
        cnt_w = max(10, _display_width("\u6807\u6ce8\u6570\u91cf"))
        img_w = max(12, _display_width("\u6d89\u53ca\u56fe\u7247\u6570"))

        sorted_cids = sorted(per_class.keys())
        for cid in sorted_cids:
            cnt, img_set = per_class[cid]
            name = class_names.get(cid, "\u672a\u77e5")
            id_w = max(id_w, _display_width(str(cid)))
            name_w = max(name_w, _display_width(name))
            cnt_w = max(cnt_w, _display_width(str(cnt)))
            img_w = max(img_w, _display_width(str(len(img_set))))

        # --- \u6784\u5efa\u8868\u683c ---
        sep = "+" + "-" * (id_w + 2) + "+" + "-" * (name_w + 2) + "+" + "-" * (cnt_w + 2) + "+" + "-" * (img_w + 2) + "+"

        # \u8868\u5934
        header = ("| " + _pad("\u7c7b\u522bID", id_w) + " | "
                  + _pad("\u7c7b\u522b\u540d\u79f0", name_w) + " | "
                  + _pad("\u6807\u6ce8\u6570\u91cf", cnt_w) + " | "
                  + _pad("\u6d89\u53ca\u56fe\u7247\u6570", img_w) + " |")

        lines_out.append(sep)
        lines_out.append(header)
        lines_out.append(sep)

        for cid in sorted_cids:
            cnt, img_set = per_class[cid]
            name = class_names.get(cid, "\u672a\u77e5")
            row = ("| " + _pad(str(cid), id_w) + " | "
                   + _pad(name, name_w) + " | "
                   + _pad(str(cnt), cnt_w) + " | "
                   + _pad(str(len(img_set)), img_w) + " |")
            lines_out.append(row)

        lines_out.append(sep)
        text = "\n".join(lines_out)

        # --- \u5bf9\u8bdd\u6846 ---
        dialog = QDialog(self)
        dialog.setWindowTitle("\u6570\u636e\u96c6\u7edf\u8ba1")
        dialog.resize(750, 520)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel("\u6570\u636e\u96c6\u7edf\u8ba1")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        text_edit = QTextEdit()
        text_edit.setPlainText(text)
        text_edit.setReadOnly(True)
        font = QFont("Consolas", 12)
        text_edit.setFont(font)
        text_edit.setLineWrapMode(QTextEdit.NoWrap)
        layout.addWidget(text_edit, 1)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("\u5173\u95ed")
        close_btn.setMinimumWidth(80)
        close_btn.setMinimumHeight(32)
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
        dialog.exec_()

    def openModel(self, _value=False):
        """?????????????"""
        start_dir = self.last_model_dir if self.last_model_dir and os.path.exists(
            self.last_model_dir) else ""
        model_path, _ = QFileDialog.getOpenFileName(self, "?? YOLO ????", start_dir,
                                                    "PyTorch ?? (*.onnx);;???? (*.*)")
        if model_path:
            self._load_model_from_path(model_path)

    def save_currentTxt_action(self, _value=False):
        """
        保存当前图片的标签到txt文件
        """
        # 检查是否有打开的文件
        if not self.file_path:
            self.error_message("无文件", "请先打开一个图片文件")
            return

        # 确定要保存的数据源和文件名后缀
        sources = []
        for suffix, data_list in self._get_save_sources():
            if len(data_list) > 0:
                sources.append((suffix, data_list))

        if not sources:
            self.error_message("无标签", "当前图片没有对应模式的标签")
            return

        try:
            basename = os.path.basename(os.path.splitext(self.file_path)[0])

            if self.default_save_dir and len(self.default_save_dir) > 1:
                dir_path = self.default_save_dir
            else:
                dir_path = os.path.dirname(self.file_path)

            classes_path = os.path.join(dir_path, 'classes.txt')

            for suffix, data_list in sources:
                save_path = os.path.join(dir_path, basename + suffix + TXT_EXT)

                if os.path.exists(save_path):
                    replys = QMessageBox.question(
                        self, '文件已存在',
                        f'文件 {os.path.basename(save_path)} 已存在，是否替换？',
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                    )
                    if replys == QMessageBox.No:
                        continue

                # 查找当前图片对应的数据项
                target_item = None
                for item in data_list:
                    if str(item[-1]) == self.file_path:
                        target_item = item
                        break

                if target_item is None:
                    continue

                with open(save_path, 'w', encoding='utf-8') as f:
                    for i in range(len(target_item) - 1):
                        yolo_line = ' '.join(map(str, target_item[i]))
                        f.write(yolo_line + '\n')

                self.statusBar().showMessage(f"标签已保存到: {save_path}")
                self.statusBar().show()

            # 更新classes.txt文件
            if hasattr(
                    self,
                    'yolo_model') and self.yolo_model and self._dataset_can_generate_classes_txt and not os.path.exists(
                classes_path):
                with open(classes_path, 'w', encoding='utf-8') as f:
                    for class_name in self.yolo_model.classes:
                        f.write(str(class_name) + '\n')

            if self.file_path:
                self.canvas.shapes = []
                self.items_to_shapes.clear()
                self.shapes_to_items.clear()
                self.label_list.clear()
                self.show_bounding_box_from_annotation_file(self.file_path)
                self.canvas.update()
        except Exception as e:
            self.error_message("保存失败", f"{e}")

    def loaddata(self, _value=False):

        infer_imgs = ustr(QFileDialog.getExistingDirectory(
            self,
            '%s - Open Directory' % __appname__,
            "",  # 关键：去掉默认路径
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        ))

        if infer_imgs:
            self.m_img_lists = self.scan_all_images(infer_imgs)
            self.detection_results = {}
        # 遍历所有图片并存储到 detection_results
        for img_path in self.m_img_list:
            try:
                # 获取不带扩展名的文件名
                img_name_without_ext = os.path.splitext(
                    os.path.basename(img_path))[0]

                # 直接搜索：img_name_without_ext.*
                search_pattern = os.path.join(
                    infer_imgs, f"{img_name_without_ext}.*")
                matching_files = glob.glob(search_pattern)  # 返回所有匹配的文件列表

                target_img_path = matching_files[0] if matching_files else None

                # 如果找到了同名文件
                if target_img_path and os.path.exists(target_img_path):
                    # 读取图片
                    with open(target_img_path, 'rb') as f:
                        img_data = cv2.imdecode(np.frombuffer(
                            f.read(), np.uint8), cv2.IMREAD_COLOR)

                    # 存储到 detection_results
                    self.detection_results[img_path] = {
                        'image': img_data,
                        'stats': {}
                    }
                else:
                    print(f"未找到匹配的图片文件: {img_name_without_ext}")
            except Exception as e:
                print(f"读取图片 {img_path} 失败: {str(e)}")

        self.update_preview_display()

    def errordata(self, _value=False, dir_path=None):
        if len(self.pre_error_img_txt) > 0:
            reply = QMessageBox.question(
                self,
                '确认保存',
                '确定要提取所有错误预测数据？',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

            default_open_dir_path = dir_path if dir_path else '.'
            if self.last_open_dir and os.path.exists(self.last_open_dir):
                default_open_dir_path = self.last_open_dir
            else:
                default_open_dir_path = os.path.dirname(
                    self.file_path) if self.file_path else '.'

            errordatapath = default_open_dir_path + "_PreErrorData"

            replys = QMessageBox.question(
                self,
                f'默认保存路径{errordatapath}',
                '是否自定义保存路径？',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if replys == QMessageBox.Yes:
                error_save = ustr(QFileDialog.getExistingDirectory(self,
                                                                   '%s - Open Directory' % __appname__,
                                                                   default_open_dir_path,
                                                                   QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))
            else:
                error_save = errordatapath
                if os.path.exists(error_save):
                    shutil.rmtree(error_save)
                os.makedirs(error_save)

            # self.last_open_dir = error_save

            for i in self.pre_error_img_txt:
                img_name = os.path.basename(i)
                shutil.copy2(i, os.path.join(error_save, img_name))
                shutil.copy2(
                    i.replace(
                        '.jpg', '.txt'), os.path.join(
                        error_save, img_name).replace(
                        '.jpg', '.txt'))

            a = os.path.join(default_open_dir_path, 'classes.txt')
            if os.path.exists(a):
                shutil.copy2(a, os.path.join(error_save, 'classes.txt'))

        else:
            self.error_message("无错误数据", "无错误数据")

    def _get_save_sources(self):
        save_mode = self.save_mode_combobox.currentText() if hasattr(
            self, 'save_mode_combobox') else "目标框标注"
        has_txt = len(self.pre_img_txt) > 0
        has_seg = len(self.pre_img_seg) > 0
        sources = []
        if save_mode in ("目标框标注", "两者"):
            if has_txt:
                sources.append(("", self.pre_img_txt))
            elif has_seg and save_mode == "目标框标注":
                # Fallback: no bbox data but seg data exists (e.g. SAM3
                # polygon output while combo still on bbox mode)
                sources.append(("", self.pre_img_seg))
        if save_mode in ("分割标注", "两者"):
            if has_seg:
                suffix = "_seg" if save_mode == "两者" else ""
                sources.append((suffix, self.pre_img_seg))
            elif has_txt and save_mode == "分割标注":
                # Fallback: no seg data but bbox data exists
                sources.append(("", self.pre_img_txt))
        return sources

    # asd
    def savetxtaction(self, _value=False):
        reply = QMessageBox.question(
            self,
            '确认保存',
            '确定要保存所有图片的标签吗？',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.No:
            return

        sources = self._get_save_sources()
        if not sources:
            self.error_message("无标签", "没有可保存的标签数据")
            return

        if reply == QMessageBox.Yes:

            replys = QMessageBox.question(
                self,
                '确认保存',
                '确定保存标签到原路径？',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if replys == QMessageBox.Yes:
                try:
                    for suffix, data_list in sources:
                        total = len(data_list)
                        progress = QProgressDialog(
                            "正在保存...", "取消", 0, total, self)
                        progress.setWindowModality(Qt.WindowModal)
                        progress.show()
                        saveClassesTxt = ''
                        for idx, item in enumerate(data_list):
                            if progress.wasCanceled():
                                break
                            img_path = str(item[-1])
                            if self.default_save_dir is not None and len(
                                    self.default_save_dir) > 1:
                                default_save_path = self.default_save_dir
                                saveClassesTxt = os.path.join(
                                    default_save_path, 'classes.txt')
                                base_name = os.path.basename(img_path)
                                name_without_ext = os.path.splitext(base_name)[
                                    0]
                                label_name = os.path.join(
                                    default_save_path, name_without_ext + suffix + '.txt')
                            else:
                                txt_path = os.path.splitext(
                                    img_path)[0] + suffix + '.txt'
                                label_name = txt_path
                                saveClassesTxt = os.path.join(
                                    self.last_open_dir, 'classes.txt') if self.last_open_dir else os.path.join(
                                    os.path.dirname(img_path), 'classes.txt')
                            progress.setValue(idx)
                            progress.setLabelText(
                                f"正在保存: {os.path.basename(label_name)}")
                            QApplication.processEvents()
                            with open(label_name, 'w', encoding='utf-8') as f:
                                for i in range(len(item) - 1):
                                    yolo_line = ' '.join(map(str, item[i]))
                                    f.write(yolo_line + '\n')

                        if self._dataset_can_generate_classes_txt and saveClassesTxt and not os.path.exists(
                                saveClassesTxt):
                            with open(saveClassesTxt, 'w', encoding='utf-8') as f:
                                for i in self.yolo_model.classes:
                                    f.write(str(i) + '\n')

                        # Write Chinese label mapping to separate file if
                        # active (keep classes.txt English-only)
                        if self._dataset_can_generate_classes_txt and saveClassesTxt and getattr(
                                self, "_active_use_zh", False):
                            try:
                                label_map = getattr(
                                    self, "_active_label_map", {}) or {}
                                if label_map:
                                    zh_path = os.path.splitext(saveClassesTxt)[
                                                  0] + '_zh.txt'
                                    with open(zh_path, 'w', encoding='utf-8') as f:
                                        for i in self.yolo_model.classes:
                                            en = str(i)
                                            zh = label_map.get(en, "")
                                            if zh:
                                                f.write(f"{en}={zh}\n")
                            except Exception:
                                pass

                        progress.setValue(total)
                        progress.close()
                except Exception as e:
                    self.error_message("保存失败")
            if replys == QMessageBox.No:
                dir_path = None
                default_open_dir_path = dir_path if dir_path else '.'
                if self.last_open_dir and os.path.exists(self.last_open_dir):
                    default_open_dir_path = self.last_open_dir
                else:
                    default_open_dir_path = os.path.dirname(
                        self.file_path) if self.file_path else '.'

                default_save_path = ustr(QFileDialog.getExistingDirectory(self,
                                                                          '%s - Open Directory' % __appname__,
                                                                          default_open_dir_path,
                                                                          QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))

                for suffix, data_list in sources:
                    total = len(data_list)
                    progress = QProgressDialog("正在保存...", "取消", 0, total, self)
                    progress.setWindowModality(Qt.WindowModal)
                    progress.show()
                    saveClassesTxt = os.path.join(
                        default_save_path, 'classes.txt')
                    for idx, item in enumerate(data_list):
                        if progress.wasCanceled():
                            break
                        img_path = str(item[-1])
                        base_name = os.path.basename(img_path)
                        name_without_ext = os.path.splitext(base_name)[0]
                        label_name = os.path.join(
                            default_save_path, name_without_ext + suffix + '.txt')

                        with open(label_name, 'w', encoding='utf-8') as f:
                            for i in range(len(item) - 1):
                                yolo_line = ' '.join(map(str, item[i]))
                                f.write(yolo_line + '\n')

                    if self._dataset_can_generate_classes_txt and saveClassesTxt and not os.path.exists(
                            saveClassesTxt):
                        with open(saveClassesTxt, 'w', encoding='utf-8') as f:
                            for i in self.yolo_model.classes:
                                f.write(str(i) + '\n')

                    # Write Chinese label mapping to separate file if active
                    if self._dataset_can_generate_classes_txt and saveClassesTxt and getattr(
                            self, "_active_use_zh", False):
                        try:
                            label_map = getattr(
                                self, "_active_label_map", {}) or {}
                            if label_map:
                                zh_path = os.path.splitext(saveClassesTxt)[
                                              0] + '_zh.txt'
                                with open(zh_path, 'w', encoding='utf-8') as f:
                                    for i in self.yolo_model.classes:
                                        en = str(i)
                                        zh = label_map.get(en, "")
                                        if zh:
                                            f.write(f"{en}={zh}\n")
                        except Exception:
                            pass

                    progress.setValue(total)
                    progress.close()

    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes [x1, y1, x2, y2]"""
        x1, y1, x2, y2 = box1
        xx1, yy1, xx2, yy2 = box2

        # Calculate intersection area
        xi1 = max(x1, xx1)
        yi1 = max(y1, yy1)
        xi2 = min(x2, xx2)
        yi2 = min(y2, yy2)

        if xi2 < xi1 or yi2 < yi1:
            return 0

        inter_area = (xi2 - xi1) * (yi2 - yi1)

        # Calculate box areas
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (xx2 - xx1) * (yy2 - yy1)

        # Calculate union area
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0

        return inter_area / union_area

    def compute_iou(self, box1, box2):
        """
        计算两个边界框的交并比（IoU）
        box = [x1, y1, x2, y2]
        """
        label, points, _, _, _ = box2
        if len(points) >= 2:
            # Get bbox from points
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            box2 = [min(xs), min(ys), max(xs), max(ys)],

        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])

        inter_area = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area != 0 else 0

    def find_missed_and_false_detections(
            self, pred_boxes, gt_boxes, iou_threshold=0.3):
        """
        同时找出漏检和误检
        pred_boxes: 模型预测的框（[x1, y1, x2, y2]）
        gt_boxes: 真实标签框（[x1, y1, x2, y2]）
        返回：
            missed_boxes: 未被检测到的真实框
            false_positives: 没有真实框对应的预测框（误检）
        """
        missed_boxes = []
        false_positives = []

        gt_matched = [False] * len(gt_boxes)
        pred_matched = [False] * len(pred_boxes)

        for i, pred in enumerate(pred_boxes):
            for j, gt in enumerate(gt_boxes):
                if not gt_matched[j]:
                    iou = self.compute_iou(pred[0], gt)
                    if iou >= iou_threshold:
                        gt_matched[j] = True
                        pred_matched[i] = True
                        break

        # 漏检：未匹配到的真实框
        for i, matched in enumerate(gt_matched):
            if not matched:
                missed_boxes.append(gt_boxes[i])

        # 误检：未匹配到的预测框
        for i, matched in enumerate(pred_matched):
            if not matched:
                false_positives.append(pred_boxes[i])

        return missed_boxes, false_positives

    def analyze_false_detections(
            self,
            predicted_boxes,
            ground_truth_shapes,
            iou_threshold=0.5):
        """Analyze false positives and false negatives"""
        # Convert ground truth shapes to boxes
        gt_boxes = []
        for shape in ground_truth_shapes:
            label = shape[0]
            points = shape[1]
            if len(points) >= 2:
                # Get bbox from points
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                gt_boxes.append({
                    'box': [min(xs), min(ys), max(xs), max(ys)],
                    'label': label
                })

        # Track matched GT boxes and predicted boxes
        matched_gt = [False] * len(gt_boxes)
        matched_pred = [False] * len(predicted_boxes)
        false_positives = []
        true_positives = []

        # First pass: Match predicted boxes with ground truth boxes based on IoU
        # For each GT box, find the best matching predicted box
        for gt_idx, gt in enumerate(gt_boxes):
            if matched_gt[gt_idx]:
                continue

            best_iou = 0
            best_pred_idx = -1

            # Find best matching predicted box (regardless of label)
            for pred_idx, (pred_box, pred_label) in enumerate(predicted_boxes):
                if matched_pred[pred_idx]:
                    continue

                iou = self.calculate_iou(pred_box, gt['box'])
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_pred_idx = pred_idx

            # If found a match
            if best_pred_idx >= 0:
                matched_gt[gt_idx] = True
                matched_pred[best_pred_idx] = True
                pred_box, pred_label = predicted_boxes[best_pred_idx]

                # Check if labels match

                gt_class_name = gt['label']

                if pred_label == gt_class_name:
                    # True positive: correct detection with correct label
                    true_positives.append({
                        'box': pred_box,
                        'label': pred_label,
                        'iou': best_iou
                    })
                else:
                    # False positive: detection is correct but label is wrong
                    false_positives.append({
                        'box': pred_box,
                        'label': pred_label,
                        'gt_label': gt['label'],
                        'iou': best_iou
                    })

        # Second pass: Remaining unmatched predicted boxes are false positives
        for pred_idx, (pred_box, pred_label) in enumerate(predicted_boxes):
            if not matched_pred[pred_idx]:
                false_positives.append({
                    'box': pred_box,
                    'label': pred_label
                })

        # False negatives are unmatched GT boxes
        false_negatives = []
        for idx, gt in enumerate(gt_boxes):
            if not matched_gt[idx]:
                false_negatives.append(gt)

        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }

    def update_preview_display(self):
        """Update the left preview with detected image and error visualization"""
        # Show detection result on top
        if not self.file_path or self.file_path not in self.detection_results:
            self.detected_preview_label.setText("暂无检测结果")
            self.detected_preview_label.setPixmap(QPixmap())
            self.detected_info_label.setText("原图")
            self.detected_info_label.setPixmap(QPixmap())
            return

        result = self.detection_results[self.file_path]
        stats = result.get('stats', {})

        # Top: Show full detection result (使用原始尺寸，让ZoomableImageView处理缩放)
        qimage = result['image']
        if isinstance(qimage, str):
            try:
                input_img = cv2.imdecode(
                    np.fromfile(
                        qimage,
                        dtype=np.uint8),
                    cv2.IMREAD_COLOR)
            except Exception:
                input_img = _imread_unicode(qimage)
            if input_img is None:
                self.detected_preview_label.setPixmap(QPixmap())
                return
            img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            resultboxes, resultscores, resultclass_ids = result['result']
            if getattr(self, "yolo_model", None) is not None:
                # Draw segmentation masks
                mask_maps = result.get('mask_maps', None)
                if mask_maps is not None and len(mask_maps) > 0 and hasattr(
                        self.yolo_model, 'draw_masks'):
                    try:
                        img_rgb = self.yolo_model.draw_masks(
                            img_rgb, resultboxes, resultclass_ids, 0.4, mask_maps)
                    except Exception as e:
                        print("绘制分割掩码失败:", e)
                        pass
                try:
                    use_zh = bool(getattr(self, "_active_use_zh", False))
                    label_map = getattr(self, "_active_label_map", {}) or {}
                except Exception:
                    use_zh = False
                    label_map = {}
                # If we need Chinese label rendering, use PIL-based drawing to
                # avoid garbled text.
                if use_zh and label_map:
                    try:
                        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                        img_bgr = VideoProcessWorker._draw_boxes(
                            img_bgr,
                            boxes=resultboxes,
                            scores=resultscores,
                            class_ids=resultclass_ids,
                            classes=getattr(self.yolo_model, "classes", []),
                            label_map=label_map,
                            use_zh=True,
                            show_conf=True,
                        )
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    except Exception:
                        pass
                else:
                    for i in range(len(resultboxes)):
                        box, cls_id, score = resultboxes[i], int(
                            resultclass_ids[i]), float(resultscores[i])
                        try:
                            color = self.yolo_model.color_palette[cls_id]
                            label = self.yolo_model.classes[cls_id]
                        except Exception:
                            color = (0, 255, 0)
                            label = str(cls_id)
                        img_rgb = self.yolo_model.plot_one_box(
                            x=box, im=img_rgb, color=color, label=label, score=score)
        else:
            img_rgb = cv2.cvtColor(qimage, cv2.COLOR_BGR2RGB)

        height, width, channels = img_rgb.shape
        bytes_per_line = channels * width
        qimage = QImage(
            img_rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.detected_preview_label.setText("")
        self.detected_preview_label.setPixmap(pixmap)

        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        pil_img = Image.open(self.file_path).copy()
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')

        # Bottom: Show original image with only false positives and false
        # negatives
        if len(stats) > 0:

            # Create a copy of original image
            draw = ImageDraw.Draw(pil_img)
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()

            # Draw only false positives (red) and false negatives (yellow)
            fp_count = len(stats.get('false_positives', []))
            fn_count = len(stats.get('false_negatives', []))

            # Draw false positives in red
            for fp in stats.get('false_positives', []):
                box = fp['box']
                draw.rectangle(box, outline='red', width=10)
                draw.text((box[0], box[1]), '✗', fill='red', font=font)

            # Draw false negatives in yellow
            for fn in stats.get('false_negatives', []):
                box = fn['box']
                draw.rectangle(box, outline='yellow', width=10)
                draw.text((box[0], box[1]), '⊙', fill='yellow', font=font)

            # Convert to QImage (使用原始尺寸)
            width, height = pil_img.size
            img_bytes = pil_img.tobytes()
            annotated_qimage = QImage(
                img_bytes, width, height, QImage.Format_RGB888)
            annotated_pixmap = QPixmap.fromImage(annotated_qimage)
            self.detected_info_label.setText("")
            self.detected_info_label.setPixmap(annotated_pixmap)
        else:
            # 没有统计信息，只显示原图
            width, height = pil_img.size
            img_bytes = pil_img.tobytes()
            annotated_qimage = QImage(
                img_bytes, width, height, QImage.Format_RGB888)
            annotated_pixmap = QPixmap.fromImage(annotated_qimage)
            self.detected_info_label.setText("")
            self.detected_info_label.setPixmap(annotated_pixmap)

    # A修改

    def open_batch_modify_window(self):
        if not os.path.exists(self.default_save_dir):
            QMessageBox.warning(self, "提示", "请先设置标签目录")
            return
        dialog = BatchModifyWindow(self)
        dialog.exec_()

    def resize_images(self):
        current_dir = self.last_open_dir or self.dir_name or ""
        dialog = ResizeDialog(self, current_dir)
        if not dialog.exec_():
            return

        params = dialog.get_params()
        target_dir = dialog.get_dir()
        do_backup = dialog.should_backup()

        if not os.path.isdir(target_dir):
            QMessageBox.warning(self, "错误", "目录不存在")
            return

        extensions = ['.%s' % fmt.data().decode("ascii").lower()
                      for fmt in QImageReader.supportedImageFormats()]
        images = []
        for root, dirs, files in os.walk(target_dir):
            for f in files:
                if f.lower().endswith(tuple(extensions)):
                    images.append(os.path.join(root, f))
        natural_sort(images, key=lambda x: x.lower())

        if not images:
            QMessageBox.information(self, "提示", "目录中没有图片文件")
            return

        # 备份目录
        backup_dir = None
        if do_backup:
            backup_dir = os.path.join(target_dir, "_backup_")
            os.makedirs(backup_dir, exist_ok=True)

        progress = QProgressDialog("正在缩放图片...", "取消", 0, len(images), self)
        progress.setWindowTitle("图像缩放")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        for idx, img_path in enumerate(images):
            if progress.wasCanceled():
                break
            progress.setValue(idx)
            progress.setLabelText(f"正在处理: {os.path.basename(img_path)}")

            try:
                img = cv2.imdecode(
                    np.fromfile(
                        img_path,
                        dtype=np.uint8),
                    cv2.IMREAD_COLOR)
                if img is None:
                    continue
                h, w = img.shape[:2]

                # 计算新尺寸
                if params["mode"] == "scale":
                    new_w = int(w * params["scale"])
                    new_h = int(h * params["scale"])
                else:
                    tw = params["width"]
                    th = params["height"]
                    keep = params["keep_ratio"]
                    if tw > 0 and th > 0:
                        new_w, new_h = tw, th
                        if keep:
                            ratio = min(tw / w, th / h)
                            new_w = int(w * ratio)
                            new_h = int(h * ratio)
                    elif tw > 0:
                        new_w = tw
                        new_h = int(
                            h * (tw / w)) if keep else (th if th > 0 else h)
                    elif th > 0:
                        new_h = th
                        new_w = int(
                            w * (th / h)) if keep else (tw if tw > 0 else w)
                    else:
                        new_w, new_h = w, h

                # 备份原图
                if backup_dir:
                    base = os.path.basename(img_path)
                    backup_path = os.path.join(backup_dir, base)
                    if not os.path.exists(backup_path):
                        shutil.copy2(img_path, backup_path)

                # Resize
                resized = cv2.resize(
                    img,
                    (new_w,
                     new_h),
                    interpolation=cv2.INTER_AREA if params["mode"] == "scale" and params[
                        "scale"] < 1.0 else cv2.INTER_LINEAR)
                cv2.imencode(
                    os.path.splitext(img_path)[1],
                    resized)[1].tofile(img_path)

                # 更新 YOLO 标签（仅非等比缩放时需要调整归一化坐标）
                if params["mode"] == "wh" and (
                        float(new_w) / w != float(new_h) / h):
                    txt_path = os.path.splitext(img_path)[0] + ".txt"
                    if os.path.exists(txt_path):
                        try:
                            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                                lines = f.readlines()
                            new_lines = []
                            w_ratio = float(new_w) / w
                            h_ratio = float(new_h) / h
                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) == 5:
                                    cls_id, cx, cy, bw, bh = parts
                                    new_cx = float(cx) * w_ratio
                                    new_cy = float(cy) * h_ratio
                                    new_bw = float(bw) * w_ratio
                                    new_bh = float(bh) * h_ratio
                                    new_lines.append(
                                        f"{cls_id} {new_cx:.6f} {new_cy:.6f} {new_bw:.6f} {new_bh:.6f}\n")
                                else:
                                    new_lines.append(line)
                            with open(txt_path, "w", encoding="utf-8") as f:
                                f.writelines(new_lines)
                        except Exception:
                            pass

            except Exception:
                continue

        progress.setValue(len(images))
        progress.close()
        QApplication.processEvents()

        # 如果处理的目录是当前打开的目录，刷新列表并恢复当前图片
        if target_dir == (self.last_open_dir or self.dir_name):
            prev_path = os.path.abspath(
                self.file_path) if self.file_path else None
            self.import_dir_images(target_dir)
            # import_dir_images 会加载第一张图片，尝试恢复用户之前查看的图片
            if prev_path and os.path.exists(prev_path):
                abs_list = [os.path.abspath(p) for p in self.m_img_list]
                try:
                    restore_idx = abs_list.index(os.path.normcase(prev_path))
                except ValueError:
                    restore_idx = 0
                self.cur_img_idx = restore_idx
                self.load_file(self.m_img_list[restore_idx])
            # 强制刷新画布确保滚轮/拖拽正常
            self.adjust_scale(initial=True)
            self.paint_canvas()

        QMessageBox.information(
            self, "完成", f"图像缩放完成，处理了 {min(idx + 1, len(images))} 张图片")

    def open_label_merge_dialog(self):
        dialog = LabelMergeDialog(self)
        if not dialog.exec_():
            return

        src_dir = dialog.get_src_dir()
        tgt_dir = dialog.get_tgt_dir()
        mapping = dialog.get_mapping()  # {src_class_id: tgt_class_id}
        force_append = dialog.is_force_append()
        iou_threshold = dialog.get_iou_threshold()

        # 收集源目录所有 txt 文件
        src_txt_files = {}
        for f in os.listdir(src_dir):
            if f.lower().endswith(".txt") and f != "classes.txt":
                src_txt_files[os.path.splitext(
                    f)[0]] = os.path.join(src_dir, f)

        if not src_txt_files:
            QMessageBox.information(self, "提示", "源目录中没有标签文件")
            return

        # 统计需要处理的文件
        tasks = []
        for base, src_path in src_txt_files.items():
            tgt_path = os.path.join(tgt_dir, base + ".txt")
            if os.path.exists(tgt_path):
                tasks.append((base, src_path, tgt_path))

        if not tasks:
            QMessageBox.information(self, "提示", "没有找到同名的目标标签文件")
            return

        progress = QProgressDialog("正在合并标签...", "取消", 0, len(tasks), self)
        progress.setWindowTitle("标签追加合并")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        processed = 0
        total_appended = 0

        for idx, (base, src_path, tgt_path) in enumerate(tasks):
            if progress.wasCanceled():
                break
            progress.setValue(idx)
            progress.setLabelText(f"正在处理: {base}")

            try:
                # 读取源标签，只取勾选类别的行
                with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
                    src_lines = f.readlines()

                filtered_src = []
                for line in src_lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        if cls_id in mapping:
                            filtered_src.append(parts)

                if not filtered_src:
                    continue

                # 读取目标标签
                with open(tgt_path, "r", encoding="utf-8", errors="ignore") as f:
                    tgt_lines = f.readlines()

                tgt_boxes = {}
                for line in tgt_lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        try:
                            cid = int(parts[0])
                            tgt_boxes.setdefault(cid, []).append(
                                [float(x) for x in parts[1:5]])
                        except ValueError:
                            pass

                # 获取图片尺寸用于坐标转换
                img_w, img_h = 640, 640  # 默认
                for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                    img_path = os.path.join(tgt_dir, base + ext)
                    if os.path.exists(img_path):
                        try:
                            img = cv2.imdecode(
                                np.fromfile(
                                    img_path,
                                    dtype=np.uint8),
                                cv2.IMREAD_COLOR)
                            if img is not None:
                                img_h, img_w = img.shape[:2]
                        except Exception:
                            pass
                        break

                # 处理每个源框
                appended = []
                for parts in filtered_src:
                    src_cid = int(parts[0])
                    tgt_cid = mapping[src_cid]
                    cx, cy, w_n, h_n = float(
                        parts[1]), float(
                        parts[2]), float(
                        parts[3]), float(
                        parts[4])

                    if force_append:
                        appended.append(
                            f"{tgt_cid} {cx:.6f} {cy:.6f} {w_n:.6f} {h_n:.6f}\n")
                    else:
                        # 将源框转为绝对坐标 x1y1x2y2
                        src_x1 = (cx - w_n / 2.0) * img_w
                        src_y1 = (cy - h_n / 2.0) * img_h
                        src_x2 = (cx + w_n / 2.0) * img_w
                        src_y2 = (cy + h_n / 2.0) * img_h
                        src_box = np.array([src_x1, src_y1, src_x2, src_y2])

                        # 获取目标中同 tgt_cid 的框
                        existing = tgt_boxes.get(tgt_cid, [])
                        if existing:
                            tgt_abs = []
                            for eb in existing:
                                ecx, ecy, ew_n, eh_n = eb
                                ex1 = (ecx - ew_n / 2.0) * img_w
                                ey1 = (ecy - eh_n / 2.0) * img_h
                                ex2 = (ecx + ew_n / 2.0) * img_w
                                ey2 = (ecy + eh_n / 2.0) * img_h
                                tgt_abs.append([ex1, ey1, ex2, ey2])
                            tgt_arr = np.array(tgt_abs)
                            ious = compute_iou(src_box, tgt_arr)
                            max_iou = np.max(ious) if len(ious) > 0 else 0.0
                            if max_iou > iou_threshold:
                                continue  # 重叠，跳过

                        appended.append(
                            f"{tgt_cid} {cx:.6f} {cy:.6f} {w_n:.6f} {h_n:.6f}\n")

                if appended:
                    with open(tgt_path, "a", encoding="utf-8") as f:
                        f.writelines(appended)
                    total_appended += len(appended)
                    processed += 1

            except Exception:
                continue

        progress.setValue(len(tasks))

        # 同步目标目录 classes.txt（填充新增类别）
        src_classes = dialog.get_src_classes()
        if src_classes and mapping:
            tgt_classes_path = os.path.join(tgt_dir, "classes.txt")
            tgt_classes = []
            if os.path.exists(tgt_classes_path):
                try:
                    with open(tgt_classes_path, "r", encoding="utf-8", errors="ignore") as f:
                        tgt_classes = [line.strip()
                                       for line in f if line.strip()]
                except Exception:
                    pass

            max_tgt_id = max(mapping.values()) if mapping else -1
            while len(tgt_classes) <= max_tgt_id:
                tgt_classes.append("")

            for src_cid, tgt_cid in mapping.items():
                if src_cid < len(src_classes):
                    if not tgt_classes[tgt_cid]:
                        tgt_classes[tgt_cid] = src_classes[src_cid]

            try:
                with open(tgt_classes_path, "w", encoding="utf-8") as f:
                    for name in tgt_classes:
                        f.write(name + "\n")
            except Exception:
                pass

        # 如果目标目录是当前打开的，刷新
        if tgt_dir == (self.last_open_dir or self.dir_name):
            self.import_dir_images(tgt_dir)
            if self.m_img_list:
                self.load_file(
                    self.m_img_list[0] if self.file_path is None else self.file_path)

        QMessageBox.information(self, "完成",
                                f"标签合并完成，处理了 {processed} 个文件，追加了 {total_appended} 个标签")

    def _get_class_name(self, class_id):
        """获取类别名称，优先从模型获取，其次从当前 classes.txt 获取"""
        if hasattr(
                self,
                'yolo_model') and self.yolo_model and hasattr(
            self.yolo_model,
            'classes'):
            try:
                return str(self.yolo_model.classes[class_id])
            except (IndexError, Exception):
                pass
        # Fallback to current dataset classes.txt
        if self.txt_path and os.path.exists(self.txt_path):
            try:
                with open(self.txt_path, "r", encoding="utf-8", errors="ignore") as f:
                    names = [line.strip() for line in f if line.strip()]
                if class_id < len(names) and names[class_id]:
                    return names[class_id]
            except Exception:
                pass
        return str(class_id)

    def append_detections_to_label(self, _value=False):
        """将检测结果追加到标签文件，支持修改目标类别和批量应用"""
        save_mode = self.save_mode_combobox.currentText() if hasattr(
            self, 'save_mode_combobox') else "目标框标注"
        if save_mode not in ("目标框标注", "两者"):
            QMessageBox.information(self, "无检测结果", "当前保存模式不支持目标框检测结果")
            return

        if not self.pre_img_txt:
            QMessageBox.information(self, "无检测结果", "没有检测结果，请先运行批量检测")
            return

        # 构建 image_path -> [(cid, [cx,cy,w,h]), ...] 的映射
        all_detections = {}
        for item in self.pre_img_txt:
            if len(item) < 2:
                continue
            img_path = str(item[-1])
            dets = []
            for entry in item[:-1]:
                if isinstance(entry, (list, tuple)) and len(entry) >= 5:
                    try:
                        cid = int(entry[0])
                        dets.append((cid, [float(x) for x in entry[:5]]))
                    except (ValueError, TypeError):
                        continue
            if dets:
                all_detections[img_path] = dets

        if not all_detections:
            QMessageBox.information(self, "无检测结果", "没有有效的检测框")
            return

        # 聚合所有图片的类别数量用于对话框显示
        class_counts = {}
        for dets in all_detections.values():
            for cid, _ in dets:
                class_counts[cid] = class_counts.get(cid, 0) + 1

        detected_info = []
        for cid in sorted(class_counts):
            name = self._get_class_name(cid)
            detected_info.append((cid, name, class_counts[cid]))

        dialog = DetectionAppendDialog(self, detected_classes=detected_info)
        if not dialog.exec_():
            return

        class_mapping = dialog.get_class_mapping()  # {原始cid: 目标cid}
        class_names = dialog.get_class_names()  # {目标cid: 类别名}
        iou_threshold = dialog.get_iou_threshold()
        is_batch = dialog.is_batch_apply()

        # 确定处理哪些图片
        if is_batch:
            target_images = list(all_detections.keys())
        else:
            if not self.file_path:
                self.error_message("未加载图片", "请先加载要处理的图片")
                return
            target_images = []
            abs_current = os.path.normcase(os.path.abspath(self.file_path))
            for img_path in all_detections:
                if os.path.normcase(os.path.abspath(img_path)) == abs_current:
                    target_images.append(img_path)
                    break
            if not target_images:
                QMessageBox.information(self, "无检测结果", "当前图片没有检测结果，请先运行检测")
                return

        total_appended = 0
        processed = 0

        if is_batch and len(target_images) > 1:
            progress = QProgressDialog(
                "正在批量追加标签...", "取消", 0, len(target_images), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

        for idx, img_path in enumerate(target_images):
            if is_batch and len(target_images) > 1:
                if progress.wasCanceled():
                    break
                progress.setValue(idx)
                progress.setLabelText(
                    f"处理: {os.path.basename(img_path)} {idx + 1}/{len(target_images)}")
                QApplication.processEvents()

            dets = all_detections.get(img_path, [])
            basename = os.path.splitext(os.path.basename(img_path))[0]

            # 确定目标 txt 路径
            if self.default_save_dir and len(self.default_save_dir) > 1:
                tgt_path = os.path.join(
                    self.default_save_dir, basename + TXT_EXT)
            else:
                tgt_path = os.path.splitext(img_path)[0] + TXT_EXT

            # 读取已有标签
            existing_boxes = {}
            if os.path.exists(tgt_path):
                try:
                    with open(tgt_path, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                try:
                                    ecid = int(parts[0])
                                    existing_boxes.setdefault(ecid, []).append(
                                        [float(x) for x in parts[1:5]])
                                except ValueError:
                                    pass
                except Exception:
                    pass

            # 获取图片尺寸
            img_w, img_h = 640, 640
            if img_path == getattr(
                    self,
                    'file_path',
                    None) and not self.image.isNull():
                img_h = self.image.height()
                img_w = self.image.width()
            elif os.path.exists(img_path):
                try:
                    img = cv2.imdecode(
                        np.fromfile(
                            img_path,
                            dtype=np.uint8),
                        cv2.IMREAD_COLOR)
                    if img is not None:
                        img_h, img_w = img.shape[:2]
                except Exception:
                    pass

            # 过滤并追加
            appended_lines = []
            for cid, parts in dets:
                if cid not in class_mapping:
                    continue
                target_cid = class_mapping[cid]

                cx, cy, w_n, h_n = parts[1], parts[2], parts[3], parts[4]

                # IoU 检查（与目标类别的已有框比较）
                existing = existing_boxes.get(target_cid, [])
                if existing:
                    src_x1 = (cx - w_n / 2.0) * img_w
                    src_y1 = (cy - h_n / 2.0) * img_h
                    src_x2 = (cx + w_n / 2.0) * img_w
                    src_y2 = (cy + h_n / 2.0) * img_h
                    src_box = np.array([src_x1, src_y1, src_x2, src_y2])

                    tgt_abs = []
                    for eb in existing:
                        ecx, ecy, ew_n, eh_n = eb
                        ex1 = (ecx - ew_n / 2.0) * img_w
                        ey1 = (ecy - eh_n / 2.0) * img_h
                        ex2 = (ecx + ew_n / 2.0) * img_w
                        ey2 = (ecy + eh_n / 2.0) * img_h
                        tgt_abs.append([ex1, ey1, ex2, ey2])
                    tgt_arr = np.array(tgt_abs)
                    ious = compute_iou(src_box, tgt_arr)
                    if len(ious) > 0 and np.max(ious) > iou_threshold:
                        continue

                appended_lines.append(
                    f"{target_cid} {cx:.6f} {cy:.6f} {w_n:.6f} {h_n:.6f}\n")
                existing_boxes.setdefault(
                    target_cid, []).append([cx, cy, w_n, h_n])

            if appended_lines:
                try:
                    with open(tgt_path, "a", encoding="utf-8") as f:
                        f.writelines(appended_lines)
                except Exception as e:
                    if not is_batch:
                        self.error_message("写入失败", str(e))
                    continue

                # 同步 classes.txt
                target_ids = set(class_mapping.values())
                self._sync_classes_txt_for_ids(
                    tgt_path, target_ids, class_names)

                total_appended += len(appended_lines)
                processed += 1

        if is_batch and len(target_images) > 1:
            try:
                progress.close()
            except Exception:
                pass

        # 重新加载当前图片以显示更新后的标签
        if self.file_path:
            self.load_file(self.file_path)

        if is_batch:
            self.statusBar().showMessage(
                f"批量追加完成: {total_appended} 个标签到 {processed} 张图片")
        else:
            if total_appended > 0:
                self.statusBar().showMessage(f"已追加 {total_appended} 个标签")
            else:
                self.statusBar().showMessage("没有新的标签需要追加")

    def _sync_classes_txt_for_ids(self, txt_path, class_ids, class_names=None):
        """确保 classes.txt 覆盖给定的 class_ids；
        名称来源优先级：用户自定义名 > 模型类别名 > 数字字符串"""
        dir_path = os.path.dirname(txt_path)
        classes_path = os.path.join(dir_path, "classes.txt")
        class_names = class_names or {}

        existing = []
        if os.path.exists(classes_path):
            try:
                with open(classes_path, "r", encoding="utf-8", errors="ignore") as f:
                    existing = [line.strip() for line in f if line.strip()]
            except Exception:
                pass

        model_classes = []
        if hasattr(
                self,
                'yolo_model') and self.yolo_model and hasattr(
            self.yolo_model,
            'classes'):
            model_classes = [str(c) for c in self.yolo_model.classes]

        max_id = max(class_ids) if class_ids else -1
        while len(existing) <= max_id:
            idx = len(existing)
            if idx in class_names and class_names[idx]:
                existing.append(class_names[idx])
            elif idx < len(model_classes) and model_classes[idx]:
                existing.append(model_classes[idx])
            else:
                existing.append(str(idx))

        # 同时用用户自定义名覆盖已有的空/纯数字名
        for cid, name in class_names.items():
            if cid < len(existing):
                cur = existing[cid]
                if not cur or cur == str(cid):
                    existing[cid] = name

        try:
            with open(classes_path, "w", encoding="utf-8") as f:
                for name in existing:
                    f.write(name + "\n")
        except Exception:
            pass

    def safe_batch_modify(self, mapping, modify_classes):
        """
        mapping: {old_id: new_id}
        modify_classes: false or bool
        """
        import glob

        if not mapping:
            return

        # =========================
        # 1. 确认弹窗
        # =========================
        msg = "确认要批量修改标签吗？\n\n"
        for k, v in mapping.items():
            msg += f"{k} -> {v}\n"

        reply = QMessageBox.question(
            self,
            "确认修改",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # =========================
        # 2. 读取 classes.txt
        # =========================
        label_dir = self.default_save_dir if self.default_save_dir else self.last_open_dir
        classes_path = os.path.join(label_dir, "classes.txt")
        if not os.path.exists(classes_path):
            QMessageBox.warning(self, "错误", f"classes.txt 不存在: {classes_path}")
            return

        with open(classes_path, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]

        # =========================
        # 3. 将mapping拆分为两个list
        # =========================
        old_classes = list(mapping.keys())
        new_classes = list(mapping.values())

        # =========================
        # 4. 判断是否是置换映射（互换/循环）还是融合映射
        # =========================
        keys_set = set(mapping.keys())
        values_set = set(mapping.values())
        is_permutation = (
                                 keys_set == values_set) and (
                                 len(mapping) == len(values_set))

        # =========================
        # 5. 处理所有 txt 标签
        # =========================
        txt_files = glob.glob(os.path.join(label_dir, "*.txt"))
        txt_files = [f for f in txt_files if not f.endswith("classes.txt")]

        if not txt_files:
            QMessageBox.warning(self, "警告", f"在 {label_dir} 目录下没有找到标签文件")
            return

        progress = QProgressDialog("正在修改标签...", "取消", 0, len(txt_files), self)
        progress.setWindowTitle("批量修改")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        modified_count = 0
        for i, txt_file in enumerate(txt_files):
            if progress.wasCanceled():
                break

            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception as e:
                continue

            modified_lines = []
            has_mapping_id = False
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue

                try:
                    class_id = int(parts[0])
                except ValueError:
                    continue

                if class_id in old_classes:
                    class_id += 1000
                    parts[0] = str(class_id)
                    has_mapping_id = True

                modified_lines.append(" ".join(parts))

            final_lines = []
            for line in modified_lines:
                parts = line.split()
                try:
                    class_id = int(parts[0])
                except ValueError:
                    continue

                if class_id >= 1000:
                    original_id = class_id - 1000
                    if original_id in mapping:
                        new_id = mapping[original_id]
                        parts[0] = str(new_id)

                final_lines.append(" ".join(parts))

            try:
                with open(txt_file, "w", encoding="utf-8") as f:
                    for line in final_lines:
                        f.write(line + "\n")
                if has_mapping_id:
                    modified_count += 1
            except Exception as e:
                pass

            progress.setValue(i + 1)
            QApplication.processEvents()

        progress.close()

        # =========================
        # 6. 根据mapping修改classes.txt
        # =========================
        if modify_classes:
            # 找到所有使用到的最大类别ID
            max_used_id = -1
            for txt_file in txt_files:
                try:
                    with open(txt_file, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                try:
                                    cid = int(parts[0])
                                    max_used_id = max(max_used_id, cid)
                                except ValueError:
                                    continue
                except Exception:
                    continue

            # 如果是置换映射，按环移位
            if is_permutation:
                new_classes_order = classes.copy()
                visited = set()

                for start_id in mapping.keys():
                    if start_id in visited:
                        continue

                    cycle = []
                    current = start_id
                    while current in mapping and current not in visited:
                        visited.add(current)
                        cycle.append(current)
                        current = mapping[current]

                    if len(cycle) >= 2 and mapping.get(cycle[-1]) == cycle[0]:
                        for i in range(len(cycle)):
                            old_pos = cycle[i]
                            new_pos = mapping[old_pos]
                            if 0 <= new_pos < len(
                                    classes) and 0 <= old_pos < len(classes):
                                new_classes_order[new_pos] = classes[old_pos]

                # 如果新位置超出了原classes长度，补齐
                if len(new_classes_order) <= max_used_id:
                    # 扩展并用数字填充
                    while len(new_classes_order) <= max_used_id:
                        new_classes_order.append(str(len(new_classes_order)))
                classes = new_classes_order

            else:
                # 融合映射或其他情况：直接扩展classes到最大使用ID
                while len(classes) <= max_used_id:
                    classes.append(str(len(classes)))

            # 写回classes.txt
            try:
                with open(classes_path, "w", encoding="utf-8") as f:
                    for cls_name in classes:
                        f.write(cls_name + "\n")
            except Exception as e:
                pass

        # =========================
        # 7. 刷新当前页面显示
        # =========================
        if self.file_path:
            current_path = os.path.abspath(self.file_path)
            if current_path in self._image_label_cache:
                del self._image_label_cache[current_path]

            self.canvas.setUpdatesEnabled(False)
            try:
                self.canvas.shapes = []
                self.canvas.selected_shape = None
                self.items_to_shapes.clear()
                self.shapes_to_items.clear()
                self.label_list.clear()
                self.combo_box.cb.clear()

                self.show_bounding_box_from_annotation_file(self.file_path)

                self._update_class_filter_items()

                if self._only_show_selected_class_labels:
                    self._apply_label_visibility_filter()

            finally:
                self.canvas.setUpdatesEnabled(True)
                self.canvas.update()

        QMessageBox.information(self, "完成",
                                f"批量修改完成\n共处理 {len(txt_files)} 个文件，修改 {modified_count} 个文件")

    def batch_delete_labels(self, delete_ids, modify_classes):
        # 弹出确认框
        confirm = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除 类别 {delete_ids} 的所有标注吗？\n删除后无法恢复，已自动备份！",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        # 如果点否，直接退出
        if confirm != QMessageBox.Yes:
            return

        # ========== 原有逻辑 ==========
        label_dir = self.default_save_dir

        # 读取原始类别
        classes_path = os.path.join(label_dir, "classes.txt")
        with open(classes_path, "r", encoding="utf-8") as f:
            old_classes = [line.strip() for line in f if line.strip()]

        # 要保留的类别
        new_classes = [c for i, c in enumerate(
            old_classes) if i not in delete_ids]

        # 旧ID → 新ID 映射
        id_map = {}
        new_id = 0
        for old_id in range(len(old_classes)):
            if old_id not in delete_ids:
                id_map[old_id] = new_id
                new_id += 1

        # 处理所有标签文件
        for f in os.listdir(label_dir):
            if not f.endswith(".txt") or f == "classes.txt":
                continue
            path = os.path.join(label_dir, f)

            new_lines = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    p = line.strip().split()
                    # if len(p) != 5:
                    #     new_lines.append(line)
                    #     continue

                    cid = int(p[0])
                    if cid in delete_ids:
                        continue

                    if modify_classes:
                        # 需要同步修改 classes.txt：做 ID 重新映射
                        if cid in id_map:
                            p[0] = str(id_map[cid])
                    # else: modify_classes=False，保持原 ID 不变

                    new_lines.append(" ".join(p) + "\n")

            with open(path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

        if modify_classes:
            # 同步更新 classes.txt（删除被删类别，并重新排列）
            with open(classes_path, "w", encoding="utf-8") as f:
                f.write("\n".join(new_classes) + "\n")

        # 刷新当前页面显示
        if self.file_path:
            current_path = os.path.abspath(self.file_path)
            if current_path in self._image_label_cache:
                del self._image_label_cache[current_path]

            self.canvas.setUpdatesEnabled(False)
            try:
                self.canvas.shapes = []
                self.canvas.selected_shape = None
                self.items_to_shapes.clear()
                self.shapes_to_items.clear()
                self.label_list.clear()
                self.combo_box.cb.clear()

                self.show_bounding_box_from_annotation_file(self.file_path)

                self._update_class_filter_items()

                if self._only_show_selected_class_labels:
                    self._apply_label_visibility_filter()

            finally:
                self.canvas.setUpdatesEnabled(True)
                self.canvas.update()

        QMessageBox.information(self, "完成", "✅ 所选类别已删除！")

    def _load_shapes_from_detection_result(self, file_path):
        """Convert detection boxes to canvas shapes if detection_results exists."""
        fp = os.path.abspath(file_path)
        dr = self.detection_results.get(fp)
        if not dr:
            return False
        boxes, scores, class_ids = dr.get("result", [[], [], []])
        if not boxes:
            return False
        shapes = []
        classes = getattr(self.yolo_model, "classes", [])
        for i in range(len(boxes)):
            try:
                x1, y1, x2, y2 = boxes[i]
            except Exception:
                continue
            try:
                cid = int(class_ids[i])
            except Exception:
                cid = -1
            try:
                name = classes[cid] if 0 <= cid < len(classes) else str(cid)
            except Exception:
                name = str(cid)
            # Apply Chinese label mapping if active
            try:
                if getattr(self, "_active_use_zh", False):
                    label_map = getattr(self, "_active_label_map", {}) or {}
                    mapped = label_map.get(name, "")
                    if mapped:
                        name = mapped
            except Exception:
                pass
            shapes.append((name, [(float(x1), float(y1)), (float(x2), float(y2))],
                           None, None, False, "rectangle"))
        if shapes:
            self.load_labels(shapes)
            return True
        return False

    def detectimg(self, _value=False):
        # SAM 文字提示模式 → 触发文字推理而非 YOLO 检测
        if (getattr(self.canvas, 'sam_enabled', False)
                and self.canvas.sam_mode == "text"
                and self.sam_text_input.text().strip()):
            self._on_sam_text_triggered()
            return
        # SAM 框提示模式 → 遍历图上所有目标框喂 SAM
        if (getattr(self.canvas, 'sam_enabled', False)
                and self.canvas.sam_mode == "box"):
            # 确保当前图片已编码
            if self.sam_client.current_image is None:
                self._sam_encode_current_image()
            if self.sam_client.current_image is None:
                self.statusBar().showMessage("框提示：无法读取当前图片", 3000)
                return
            # 只处理原始形状（跳过已 SAM 生成的）
            shapes = [s for s in self.canvas.shapes
                      if s.points and not getattr(s, 'is_sam', False)]
            if not shapes:
                self.statusBar().showMessage("框提示：当前图片没有待处理的目标框", 3000)
                return
            self.statusBar().showMessage(f"SAM 框提示推理中... 共 {len(shapes)} 个框")
            count = 0
            for shape in shapes:
                xs = [p.x() for p in shape.points]
                ys = [p.y() for p in shape.points]
                x1, y1, x2, y2 = map(int, [min(xs), min(ys), max(xs), max(ys)])
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                mask = self.sam_client.predict_box(x1, y1, x2, y2)
                if mask is not None:
                    bbox = self.sam_client._mask_to_bbox(mask)
                    self._on_sam_result_ready(
                        {"type": "box", "mask": mask, "bbox": bbox,
                         "src_label": shape.label})
                    count += 1
            self.statusBar().showMessage(f"SAM 框提示完成：{count}/{len(shapes)} 个框生成分割", 3000)
            return
        # Check if model is loaded
        if self.yolo_model is None:
            self.error_message("未加载模型", "请先加载YOLO模型")
            return

        # Check if image is loaded
        if self.image.isNull():
            self.error_message("未加载图片", "请先加载要检测的图片")
            return

        # Check if file path exists
        if not self.file_path:
            self.error_message("文件路径无效", "无法获取文件路径")
            return

        try:
            # Run YOLO detection on the image
            self.statusBar().showMessage("正在检测...")
            self.statusBar().show()

            self._ensure_classes_txt_for_detection(self.file_path)
            if isinstance(self.yolo_model, YOLOv8_Seg):
                input_image, resultboxes, resultscores, resultclass_ids, resultboxesxywhns, _ = self.yolo_model.infer(
                    self.file_path)
            else:
                input_image, resultboxes, resultscores, resultclass_ids, resultboxesxywhns = self.yolo_model.infer(
                    self.file_path)

            predicted_boxes = []
            for i in range(len(resultboxes)):
                class_id = int(resultclass_ids[i])
                confidence = float(resultscores[i])

                if self._dataset_uses_numeric_labels:
                    class_name = str(class_id)
                else:
                    try:
                        class_name = self.yolo_model.classes[class_id]
                    except Exception:
                        class_name = str(class_id)

                x1, y1, x2, y2 = resultboxes[i]
                predicted_boxes.append(([x1, y1, x2, y2], class_name))

            resultboxesxywhns.append(self.file_path)  # 将 image_path 添加到 resultboxesxywhns 中

            # 根据保存模式存入对应数据结构
            save_mode = self.save_mode_combobox.currentText() if hasattr(self, 'save_mode_combobox') else "目标框标注"
            if save_mode in ("目标框标注", "两者"):
                self.pre_img_txt.append(resultboxesxywhns)
            if save_mode in ("分割标注", "两者"):
                if isinstance(self.yolo_model, YOLOv8_Seg):
                    mm = getattr(self.yolo_model, 'mask_maps', None)
                    if mm is not None and len(mm) > 0:
                        img_h, img_w = self.yolo_model.img_height, self.yolo_model.img_width
                        polys = YOLOv8_Seg._masks_to_polygons(mm, resultclass_ids, img_w, img_h)
                        if polys:
                            seg_entry = []
                            for cid, pts in polys:
                                seg_entry.append([cid] + pts)
                            seg_entry.append(self.file_path)
                            self.pre_img_seg.append(seg_entry)

            # Load ground truth from annotation file
            ground_truth_shapes = []
            annotation_path = None

            if self.default_save_dir:
                basename = os.path.basename(os.path.splitext(self.file_path)[0])
                xml_path = os.path.join(self.default_save_dir, basename + XML_EXT)
                txt_path = os.path.join(self.default_save_dir, basename + TXT_EXT)

                if os.path.isfile(xml_path):
                    annotation_path = xml_path
                    reader = PascalVocReader(xml_path)
                    ground_truth_shapes = reader.get_shapes()
                elif os.path.isfile(txt_path):
                    annotation_path = txt_path
                    class_list_path = self.txt_path if getattr(self, "_dataset_has_classes_txt",
                                                               False) and self.txt_path and os.path.exists(
                        self.txt_path) else None
                    reader = YoloReader(txt_path, self.image, class_list_path=class_list_path)
                    ground_truth_shapes = reader.get_shapes()
            else:
                xml_path = os.path.splitext(self.file_path)[0] + XML_EXT
                txt_path = os.path.splitext(self.file_path)[0] + TXT_EXT

                if os.path.isfile(xml_path):
                    annotation_path = xml_path
                    reader = PascalVocReader(xml_path)
                    ground_truth_shapes = reader.get_shapes()
                elif os.path.isfile(txt_path):
                    annotation_path = txt_path
                    class_list_path = self.txt_path if getattr(self, "_dataset_has_classes_txt",
                                                               False) and self.txt_path and os.path.exists(
                        self.txt_path) else None
                    reader = YoloReader(txt_path, self.image, class_list_path=class_list_path)
                    ground_truth_shapes = reader.get_shapes()

            if len(ground_truth_shapes):
                # Analyze false detections
                analysis_result = self.analyze_false_detections(predicted_boxes, ground_truth_shapes)
                tp_count = len(analysis_result['true_positives'])
                fp_count = len(analysis_result['false_positives'])
                fn_count = len(analysis_result['false_negatives'])
                self.statusBar().showMessage(f"检测完成 | 正确:{tp_count} 误检:{fp_count} 漏检:{fn_count}")
                self.statusBar().show()
            elif len(predicted_boxes) > 0:
                analysis_result = {
                    'true_positives': [],
                    'false_positives': [
                        {'box': box, 'iou': 0.0, 'label': lbl}
                        for box, lbl in predicted_boxes
                    ],
                    'false_negatives': []
                }
            else:
                analysis_result = {
                    'true_positives': [],
                    'false_positives': [],
                    'false_negatives': [
                    ]
                }

            # Store detection result with statistics
            self.detection_results[self.file_path] = {
                'image': input_image,
                'stats': analysis_result
            }

            # Update left preview
            self.update_preview_display()


        except Exception as e:
            tb = traceback.format_exc()
            print(tb)
            self.error_message("检测失败", f"错误: {str(e)}")
            self.statusBar().showMessage(f"检测失败: {str(e)}")
            self.statusBar().show()

    def batch_detectimg(self, _value=False):
        """Batch detect all images in the current directory"""
        # SAM 文字提示模式 → 触发批量文字推理
        if (getattr(self.canvas, 'sam_enabled', False)
                and self.canvas.sam_mode == "text"
                and self.sam_text_input.text().strip()):
            self._batch_sam_text_detect()
            return
        # Check if model is loaded
        self.detection_results = {}
        self.pre_error_img_txt = []
        self.pre_img_txt = []
        self.pre_img_seg = []
        if self.yolo_model is None:
            self.error_message("未加载模型", "请先加载YOLO模型")
            return

        # 图片批量检测并非视频检测上下文，不要自动加载检测框到画布
        self._video_detect_context = False
        self._active_label_map = {}
        self._active_use_zh = False

        # Check if there are images to detect
        if not self.m_img_list or len(self.m_img_list) == 0:
            self.error_message("无图片", "请先打开包含图片的文件夹")
            return

        try:
            # Show progress dialog
            start_idx = self.cur_img_idx if self.batch_detect_from_current else 0
            remaining_images = self.m_img_list[start_idx:]
            progress = QProgressDialog(
                "正在批量检测...", "取消", 0, len(remaining_images), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            success_count = 0
            for idx, image_path in enumerate(
                    remaining_images, start=start_idx):
                if progress.wasCanceled():
                    break

                progress.setValue(idx - start_idx)
                progress.setLabelText(
                    f"正在检测: {os.path.basename(image_path)} {idx - start_idx + 1}/{len(remaining_images)} ")
                QApplication.processEvents()

                analysis_result = []

                self._ensure_classes_txt_for_detection(image_path)

                # Check if file exists
                if not os.path.exists(image_path):
                    continue

                try:
                    # Run YOLO detection
                    # with torch.no_grad():
                    if isinstance(self.yolo_model, YOLOv8_Seg):
                        input_image, resultboxes, resultscores, resultclass_ids, resultboxesxywhns, mask_maps = self.yolo_model.infer(
                            image_path, draw=False)
                    else:
                        input_image, resultboxes, resultscores, resultclass_ids, resultboxesxywhns = self.yolo_model.infer(
                            image_path)
                        mask_maps = None

                    # result = results[0]
                    # Get the annotated image
                    annotated_image = input_image

                    # 将 image_path 添加到 resultboxesxywhns 中
                    resultboxesxywhns.append(image_path)

                    # 根据保存模式存入对应数据结构
                    save_mode = self.save_mode_combobox.currentText() if hasattr(
                        self, 'save_mode_combobox') else "目标框标注"
                    if save_mode in ("目标框标注", "两者"):
                        self.pre_img_txt.append(resultboxesxywhns)
                    if save_mode in ("分割标注", "两者"):
                        if isinstance(
                                self.yolo_model,
                                YOLOv8_Seg) and mask_maps is not None and len(mask_maps) > 0:
                            img_h, img_w = input_image.shape[:2]
                            polys = YOLOv8_Seg._masks_to_polygons(
                                mask_maps, resultclass_ids, img_w, img_h)
                            if polys:
                                seg_entry = []
                                for cid, pts in polys:
                                    seg_entry.append([cid] + pts)
                                seg_entry.append(image_path)
                                self.pre_img_seg.append(seg_entry)

                    # Get predicted boxes
                    predicted_boxes = []
                    for i in range(len(resultboxes)):
                        class_id = int(resultclass_ids[i])
                        confidence = float(resultscores[i])
                        if self._dataset_uses_numeric_labels:
                            class_name = str(class_id)
                        else:
                            try:
                                class_name = self.yolo_model.classes[class_id]
                            except Exception:
                                class_name = str(class_id)

                        # if hasattr(result, 'names'):
                        #     class_name = result.names[class_id]
                        # else:
                        #     class_name = str(class_id)

                        x1, y1, x2, y2 = resultboxes[i]

                        predicted_boxes.append(([x1, y1, x2, y2], class_name))

                    # Load ground truth from annotation file
                    ground_truth_shapes = []
                    annotation_path = None

                    if self.default_save_dir:
                        basename = os.path.basename(
                            os.path.splitext(image_path)[0])
                        xml_path = os.path.join(
                            self.default_save_dir, basename + XML_EXT)
                        txt_path = os.path.join(
                            self.default_save_dir, basename + TXT_EXT)

                        if os.path.isfile(xml_path):
                            annotation_path = xml_path
                            reader = PascalVocReader(xml_path)
                            ground_truth_shapes = reader.get_shapes()
                        elif os.path.isfile(txt_path):
                            annotation_path = txt_path
                            # 使用当前图片尺寸读取 YOLO 标签，避免尺寸错位
                            image_for_reader = read(image_path, None)
                            class_list_path = self.txt_path if getattr(
                                self, "_dataset_has_classes_txt", False) and self.txt_path and os.path.exists(
                                self.txt_path) else None
                            reader = YoloReader(
                                txt_path, image_for_reader, class_list_path=class_list_path)
                            ground_truth_shapes = reader.get_shapes()
                    else:
                        xml_path = os.path.splitext(image_path)[0] + XML_EXT
                        txt_path = os.path.splitext(image_path)[0] + TXT_EXT

                        if os.path.isfile(xml_path):
                            annotation_path = xml_path
                            reader = PascalVocReader(xml_path)
                            ground_truth_shapes = reader.get_shapes()
                        elif os.path.isfile(txt_path):
                            annotation_path = txt_path
                            image_for_reader = read(image_path, None)
                            class_list_path = self.txt_path if getattr(
                                self, "_dataset_has_classes_txt", False) and self.txt_path and os.path.exists(
                                self.txt_path) else None
                            reader = YoloReader(
                                txt_path, image_for_reader, class_list_path=class_list_path)
                            ground_truth_shapes = reader.get_shapes()

                    if len(ground_truth_shapes):
                        # Analyze false detections
                        analysis_result = self.analyze_false_detections(
                            predicted_boxes, ground_truth_shapes)

                    else:
                        if len(predicted_boxes) > 0:
                            analysis_result = {
                                'true_positives': [],
                                'false_positives': [
                                    {'box': box, 'iou': 0.0, 'label': lbl}
                                    for box, lbl in predicted_boxes
                                ],
                                'false_negatives': []
                            }

                    # missed_boxes, false_positives =
                    # self.find_missed_and_false_detections(predicted_boxes,
                    # ground_truth_shapes)

                    # Store result in dictionary
                    # detection_count = len(result.boxes) if hasattr(result,
                    # 'boxes') else 0
                    self.detection_results[image_path] = {
                        'image': image_path,
                        'result': [resultboxes, resultscores, resultclass_ids],
                        'stats': analysis_result,
                        'mask_maps': mask_maps,
                    }

                    # if '0ddc90b41770a38578ac4e0d1bb53c88.jpg' in  image_path:
                    #     print(1)
                    # 只要 FP 或 TP 任意一个非空，就把图片路径记为“有结果”
                    if len(analysis_result) == 0:
                        self.pre_error_img_txt.append(image_path)
                    else:
                        if analysis_result.get(
                                'false_positives') or analysis_result.get('false_negatives'):
                            self.pre_error_img_txt.append(image_path)

                    # self.pre_error_img_txt.append(analysis_result)

                    success_count += 1

                    del input_image, resultboxes, resultscores, resultclass_ids, resultboxesxywhns, mask_maps
                    # torch.cuda.empty_cache()

                except Exception as e:
                    print(f"检测 {image_path} 失败: {e}")
                    traceback.print_exc()
                    continue

            progress.setValue(len(remaining_images))
            progress.close()

            # Update preview for current image
            self.update_preview_display()

            self.statusBar().showMessage(
                f"批量检测完成: {success_count}/{len(remaining_images)} 张图片")
            self.statusBar().show()

        except Exception as e:
            traceback.print_exc()
            self.error_message("批量检测失败", f"错误: {str(e)}")
            self.statusBar().showMessage(f"批量检测失败: {str(e)}")
            self.statusBar().show()

    def open_file(self, _value=False):
        if not self.may_continue():
            return
        path = os.path.dirname(ustr(self.file_path)) if self.file_path else '.'
        formats = ['*.%s' % fmt.data().decode("ascii").lower()
                   for fmt in QImageReader.supportedImageFormats()]
        filters = "Image & Label files (%s)" % ' '.join(
            formats + ['*%s' % LabelFile.suffix])
        filename = QFileDialog.getOpenFileName(
            self, '%s - Choose Image or Label file' %
                  __appname__, path, filters)
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
            try:
                data_dir = os.path.dirname(ustr(filename))
                self._sync_video_dirs_to_data_dir(data_dir)
            except Exception:
                pass
            self.cur_img_idx = 0
            self.img_count = 1
            self.load_file(filename)

    def save_file(self, _value=False):
        if self.default_save_dir is not None and len(
                ustr(self.default_save_dir)):
            if self.file_path:
                image_file_name = os.path.basename(self.file_path)
                saved_file_name = os.path.splitext(image_file_name)[0]
                saved_path = os.path.join(
                    ustr(self.default_save_dir), saved_file_name)
                self._save_file(saved_path)
        else:
            image_file_dir = os.path.dirname(self.file_path)
            image_file_name = os.path.basename(self.file_path)
            saved_file_name = os.path.splitext(image_file_name)[0]
            saved_path = os.path.join(image_file_dir, saved_file_name)
            self._save_file(saved_path if self.label_file
                            else self.save_file_dialog(remove_ext=False))

    def save_file_as(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        self._save_file(self.save_file_dialog())

    def save_file_dialog(self, remove_ext=True):
        caption = '%s - Choose File' % __appname__
        filters = 'File (*%s)' % LabelFile.suffix
        open_dialog_path = self.current_path()
        dlg = QFileDialog(self, caption, open_dialog_path, filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        filename_without_extension = os.path.splitext(self.file_path)[0]
        dlg.selectFile(filename_without_extension)
        dlg.setOption(QFileDialog.DontUseNativeDialog, False)
        if dlg.exec_():
            full_file_path = ustr(dlg.selectedFiles()[0])
            if remove_ext:
                # Return file path without the extension.
                return os.path.splitext(full_file_path)[0]
            else:
                return full_file_path
        return ''

    def _save_file(self, annotation_file_path):
        if annotation_file_path and self.save_labels(annotation_file_path):
            self.set_clean()
            self.statusBar().showMessage('Saved to  %s' % annotation_file_path)
            self.statusBar().show()

    def close_file(self, _value=False):
        if not self.may_continue():
            return
        self.reset_state()
        self.set_clean()
        self.toggle_actions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def delete_image(self):
        delete_path = self.file_path
        if delete_path is not None:
            current_idx = self.cur_img_idx

            # 删除文件
            if os.path.exists(delete_path):
                os.remove(delete_path)

            # 重新扫描目录（更新 m_img_list）
            self.import_dir_images(self.last_open_dir)

            # 定位到下一张：原索引位置现在是旧列表中的下一张
            if self.img_count > 0:
                new_idx = min(current_idx, self.img_count - 1)
                self.cur_img_idx = new_idx
                self.load_file(self.m_img_list[self.cur_img_idx])

    def reset_all(self):
        self.settings.reset()
        self.close()
        process = QProcess()
        process.startDetached(os.path.abspath(__file__))

    def may_continue(self):
        if not self.dirty:
            return True
        else:
            discard_changes = self.discard_changes_dialog()
            if discard_changes == QMessageBox.No:
                return True
            elif discard_changes == QMessageBox.Yes:
                self.save_file()
                return True
            else:
                return False

    def discard_changes_dialog(self):
        yes, no, cancel = QMessageBox.Yes, QMessageBox.No, QMessageBox.Cancel
        msg = u'You have unsaved changes, would you like to save them and proceed?\nClick "No" to undo all changes.'
        return QMessageBox.warning(self, u'Attention', msg, yes | no | cancel)

    def error_message(self, title, message):
        return QMessageBox.critical(self, title,
                                    '<p><b>%s</b></p>%s' % (title, message))

    def current_path(self):
        return os.path.dirname(self.file_path) if self.file_path else '.'

    def choose_color1(self):
        color = self.color_dialog.getColor(self.line_color, u'Choose line color',
                                           default=DEFAULT_LINE_COLOR)
        if color:
            self.line_color = color
            Shape.line_color = color
            self.canvas.set_drawing_color(color)
            self.canvas.update()
            self.set_dirty()

    def delete_selected_shape(self):
        self.remove_label(self.canvas.delete_selected())
        self.set_dirty()
        if self._only_show_selected_class_labels:
            self._apply_label_visibility_filter()
        self._on_current_image_labels_changed()
        if self.no_shapes():
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)

    def choose_shape_line_color(self):
        color = self.color_dialog.getColor(self.line_color, u'Choose Line Color',
                                           default=DEFAULT_LINE_COLOR)
        if color:
            self.canvas.selected_shape.line_color = color
            self.canvas.update()
            self.set_dirty()

    def choose_shape_fill_color(self):
        color = self.color_dialog.getColor(self.fill_color, u'Choose Fill Color',
                                           default=DEFAULT_FILL_COLOR)
        if color:
            self.canvas.selected_shape.fill_color = color
            self.canvas.update()
            self.set_dirty()

    def copy_shape(self):
        self.canvas.end_move(copy=True)
        self.add_label(self.canvas.selected_shape)
        self.set_dirty()

    def move_shape(self):
        self.canvas.end_move(copy=False)
        self.set_dirty()

    def load_predefined_classes(self, predef_classes_file):
        if os.path.exists(predef_classes_file) is True:
            with codecs.open(predef_classes_file, 'r', 'utf8') as f:
                for line in f:
                    line = line.strip()
                    if self.label_hist is None:
                        self.label_hist = [line]
                    else:
                        self.label_hist.append(line)

    def load_pascal_xml_by_filename(self, xml_path):
        if self.file_path is None:
            return
        if os.path.isfile(xml_path) is False:
            return

        self.set_format(FORMAT_PASCALVOC)

        t_voc_parse_reader = PascalVocReader(xml_path)
        shapes = t_voc_parse_reader.get_shapes()
        self.load_labels(shapes)
        self.canvas.verified = t_voc_parse_reader.verified

    def load_yolo_txt_by_filename(self, txt_path):
        if self.file_path is None:
            return
        if os.path.isfile(txt_path) is False:
            return

        self.set_format(FORMAT_YOLO)
        class_list_path = self.txt_path if getattr(
            self,
            "_dataset_has_classes_txt",
            False) and self.txt_path and os.path.exists(
            self.txt_path) else None
        t_yolo_parse_reader = YoloReader(
            txt_path, self.image, class_list_path=class_list_path)
        shapes = t_yolo_parse_reader.get_shapes()
        # print(shapes)
        self.load_labels(shapes)
        self.canvas.verified = t_yolo_parse_reader.verified

        # Apply Chinese label mapping from labels_zh.txt if present
        try:
            txt_dir = os.path.dirname(
                txt_path) if os.path.dirname(txt_path) else "."
            zh_path = os.path.join(txt_dir, 'labels_zh.txt')
            if os.path.exists(zh_path):
                mapping = {}
                with open(zh_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or '=' not in line:
                            continue
                        en, zh = line.split('=', 1)
                        en = en.strip()
                        zh = zh.strip()
                        if en and zh:
                            mapping[en] = zh
                if mapping:
                    for shape in self.canvas.shapes:
                        if hasattr(shape, 'label') and shape.label in mapping:
                            shape.label = mapping[shape.label]
                    self.canvas.update()
        except Exception:
            pass

    def load_create_ml_json_by_filename(self, json_path, file_path):
        if self.file_path is None:
            return
        if os.path.isfile(json_path) is False:
            return

        self.set_format(FORMAT_CREATEML)

        create_ml_parse_reader = CreateMLReader(json_path, file_path)
        shapes = create_ml_parse_reader.get_shapes()
        self.load_labels(shapes)
        self.canvas.verified = create_ml_parse_reader.verified

    def copy_previous_bounding_boxes(self):
        current_index = self.m_img_list.index(self.file_path)
        if current_index - 1 >= 0:
            prev_file_path = self.m_img_list[current_index - 1]
            self.show_bounding_box_from_annotation_file(prev_file_path)
            self.save_file()

    def toggle_paint_labels_option(self):
        for shape in self.canvas.shapes:
            shape.paint_label = self.display_label_option.isChecked()

    def batch_detect_from_current_changed(self):
        self.batch_detect_from_current = self.actions.batchDetectFromCurrent.isChecked()

    def toggle_draw_square(self):
        self.canvas.set_drawing_shape_to_square(self.draw_squares_option.isChecked())

    def toggle_draw_two_clicks(self):
        self.canvas.set_drawing_by_two_clicks(self.draw_two_clicks_option.isChecked())


def inverted(color): 455


def read(filename, default=None):
    try:
        reader = QImageReader(filename)
        reader.setAutoTransform(True)
        return reader.read()
    except:
        return default


def get_main_app(argv=[]):
    """
    Standard boilerplate Qt application code.
    Do everything but app.exec_() -- so that we can test the application in one thread
    """
    app = QApplication(argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(new_icon("app"))
    try:
        _setup_app_logging(__appname__)
    except Exception:
        pass
    # Tzutalin 201705+: Accept extra agruments to change predefined class file
    argparser = argparse.ArgumentParser()
    argparser.add_argument("image_dir", nargs="?")
    argparser.add_argument("class_file",
                           default=os.path.join(os.path.dirname(__file__), "data", "predefined_classes.txt"),
                           nargs="?")
    argparser.add_argument("save_dir", nargs="?")
    args = argparser.parse_args(argv[1:])

    args.image_dir = args.image_dir and os.path.normpath(args.image_dir)
    args.class_file = args.class_file and os.path.normpath(args.class_file)
    args.save_dir = args.save_dir and os.path.normpath(args.save_dir)

    # Usage : labelImg.py image classFile saveDir
    win = MainWindow(args.image_dir,
                     args.class_file,
                     args.save_dir)
    win.show()
    return app, win


def main():
    """construct main app and run it"""
    app, _win = get_main_app(sys.argv)
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
