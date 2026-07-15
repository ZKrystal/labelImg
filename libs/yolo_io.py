#!/usr/bin/env python
# -*- coding: utf8 -*-
import sys
import os
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
from libs.constants import DEFAULT_ENCODING

TXT_EXT = '.txt'
ENCODE_METHOD = DEFAULT_ENCODING

class YOLOWriter:

    def __init__(self, folder_name, filename, img_size, database_src='Unknown', local_img_path=None):
        self.folder_name = folder_name
        self.filename = filename
        self.database_src = database_src
        self.img_size = img_size
        self.box_list = []
        self.polygon_list = []
        self.local_img_path = local_img_path
        self.verified = False

    def add_bnd_box(self, x_min, y_min, x_max, y_max, name, difficult):
        bnd_box = {'xmin': x_min, 'ymin': y_min, 'xmax': x_max, 'ymax': y_max}
        bnd_box['name'] = name
        bnd_box['difficult'] = difficult
        self.box_list.append(bnd_box)

    def add_polygon(self, name, points, difficult):
        """Add a polygon shape. points is a list of (x, y) tuples."""
        poly = {'name': name, 'points': points, 'difficult': difficult}
        self.polygon_list.append(poly)

    def bnd_box_to_yolo_line(self, box, class_list=[]):
        x_min = box['xmin']
        x_max = box['xmax']
        y_min = box['ymin']
        y_max = box['ymax']

        x_center = float((x_min + x_max)) / 2 / self.img_size[1]
        y_center = float((y_min + y_max)) / 2 / self.img_size[0]

        w = float((x_max - x_min)) / self.img_size[1]
        h = float((y_max - y_min)) / self.img_size[0]

        box_name = box['name']
        if box_name not in class_list:
            class_list.append(box_name)

        class_index = class_list.index(box_name)

        return class_index, x_center, y_center, w, h

    def polygon_to_yolo_line(self, poly, class_list=[]):
        """Convert polygon to YOLO segmentation format string."""
        name = poly['name']
        if name not in class_list:
            class_list.append(name)
        class_index = class_list.index(name)

        parts = [str(class_index)]
        for x, y in poly['points']:
            xn = float(x) / self.img_size[1]
            yn = float(y) / self.img_size[0]
            parts.append("%.6f" % xn)
            parts.append("%.6f" % yn)
        return " ".join(parts)

    def save(self, class_list=[], target_file=None):

        out_file = None  # Update yolo .txt
        out_class_file = None   # Update class list .txt

        if target_file is None:
            out_file = open(
            self.filename + TXT_EXT, 'w', encoding=ENCODE_METHOD)
            classes_file = os.path.join(os.path.dirname(os.path.abspath(self.filename)), "classes.txt")
            out_class_file = open(classes_file, 'w')

        else:
            out_file = codecs.open(target_file, 'w', encoding=ENCODE_METHOD)
            classes_file = os.path.join(os.path.dirname(os.path.abspath(target_file)), "classes.txt")
            out_class_file = open(classes_file, 'w')


        for box in self.box_list:
            class_index, x_center, y_center, w, h = self.bnd_box_to_yolo_line(box, class_list)
            out_file.write("%d %.6f %.6f %.6f %.6f\n" % (class_index, x_center, y_center, w, h))

        for poly in self.polygon_list:
            line = self.polygon_to_yolo_line(poly, class_list)
            out_file.write(line + "\n")

        for c in class_list:
            out_class_file.write(c+'\n')

        out_class_file.close()
        out_file.close()



class YoloReader:

    def __init__(self, file_path, image, class_list_path=None):
        # shapes type:
        # [label, [(x1,y1), (x2,y2), ...], color, color, difficult, shape_type]
        self.shapes = []
        self.file_path = file_path

        if class_list_path is None:
            dir_path = os.path.dirname(os.path.realpath(self.file_path))
            self.class_list_path = os.path.join(dir_path, "classes.txt")
        else:
            self.class_list_path = class_list_path

        if os.path.exists(self.class_list_path):
            classes_file = open(self.class_list_path, 'r')
            self.classes = classes_file.read().strip('\n').split('\n')
        else:
            self.classes = [str(i) for i in range(1000)]

        img_size = [image.height(), image.width(),
                    1 if image.isGrayscale() else 3]

        self.img_size = img_size

        self.verified = False
        self.parse_yolo_format()

    def get_shapes(self):
        return self.shapes

    def add_shape(self, label, x_min, y_min, x_max, y_max, difficult):
        points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        self.shapes.append((label, points, None, None, difficult, 'rectangle'))

    def add_polygon_shape(self, label, points, difficult):
        """Add a polygon shape with arbitrary number of points."""
        self.shapes.append((label, points, None, None, difficult, 'polygon'))

    def yolo_line_to_shape(self, class_index, x_center, y_center, w, h):
        if int(class_index) >= len(self.classes):
           label = class_index
        else:
           label = self.classes[int(class_index)]
        x_min = max(float(x_center) - float(w) / 2, 0)
        x_max = min(float(x_center) + float(w) / 2, 1)
        y_min = max(float(y_center) - float(h) / 2, 0)
        y_max = min(float(y_center) + float(h) / 2, 1)

        x_min = round(self.img_size[1] * x_min)
        x_max = round(self.img_size[1] * x_max)
        y_min = round(self.img_size[0] * y_min)
        y_max = round(self.img_size[0] * y_max)

        return label, x_min, y_min, x_max, y_max

    def yolo_line_to_polygon_shape(self, parts):
        """Parse YOLO segmentation format line to polygon shape."""
        class_index = parts[0]
        if int(class_index) >= len(self.classes):
            label = class_index
        else:
            label = self.classes[int(class_index)]

        points = []
        coords = parts[1:]
        for i in range(0, len(coords), 2):
            x = round(self.img_size[1] * float(coords[i]))
            y = round(self.img_size[0] * float(coords[i + 1]))
            points.append((x, y))

        return label, points

    def parse_yolo_format(self):
        bnd_box_file = open(self.file_path, 'r')
        for bndBox in bnd_box_file:
            line = bndBox.strip()
            if not line:
                continue
            parts = line.split(' ')
            if len(parts) < 5:
                continue
            if len(parts) == 5:
                # Bounding box format: class x_center y_center w h
                class_index, x_center, y_center, w, h = parts
                label, x_min, y_min, x_max, y_max = self.yolo_line_to_shape(class_index, x_center, y_center, w, h)
                self.add_shape(label, x_min, y_min, x_max, y_max, False)
            elif len(parts) >= 7 and len(parts) % 2 == 1:
                # Polygon format: class x1 y1 x2 y2 ... xn yn
                label, points = self.yolo_line_to_polygon_shape(parts)
                self.add_polygon_shape(label, points, False)
