#!/usr/bin/env python
# -*- coding: utf8 -*-
import codecs
import json
import os

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
        self.local_img_path = local_img_path
        self.verified = False

    def add_bnd_box(self, x_min, y_min, x_max, y_max, name, difficult):
        bnd_box = {'xmin': x_min, 'ymin': y_min, 'xmax': x_max, 'ymax': y_max}
        bnd_box['name'] = name
        bnd_box['difficult'] = difficult
        self.box_list.append(bnd_box)

    def bnd_box_to_yolo_line(self, box, class_list=[]):
        x_min = box['xmin']
        x_max = box['xmax']
        y_min = box['ymin']
        y_max = box['ymax']

        x_center = float((x_min + x_max)) / 2 / self.img_size[1]
        y_center = float((y_min + y_max)) / 2 / self.img_size[0]

        w = float((x_max - x_min)) / self.img_size[1]
        h = float((y_max - y_min)) / self.img_size[0]

        # PR387
        box_name = box['name']
        if box_name not in class_list:
            class_list.append(box_name)

        class_index = class_list.index(box_name)

        return class_index, x_center, y_center, w, h

    def save(self, class_list=[], target_file=None, default_prefdef_class_file=None):

        out_file = None  # Update yolo .txt
        out_class_file = None   # Update class list .txt

        if target_file is None:
            out_file = open(self.filename + TXT_EXT, 'w', encoding=ENCODE_METHOD)
            if os.path.isfile(default_prefdef_class_file):
                classes_file = default_prefdef_class_file
            else:
                classes_file = os.path.join(os.path.dirname(os.path.abspath(self.filename)), "classes.txt")
            out_class_file = open(classes_file, 'w')

        else:
            out_file = codecs.open(target_file, 'w', encoding=ENCODE_METHOD)
            if os.path.isfile(default_prefdef_class_file):
                classes_file = default_prefdef_class_file
            else:
                classes_file = os.path.join(os.path.dirname(os.path.abspath(target_file)), "classes.txt")
            out_class_file = open(classes_file, 'w')

        for box in self.box_list:
            class_index, x_center, y_center, w, h = self.bnd_box_to_yolo_line(box, class_list)
            # print (classIndex, x_center, y_center, w, h)
            out_file.write("%d %.6f %.6f %.6f %.6f\n" % (class_index, x_center, y_center, w, h))

        # print (classList)
        # print (out_class_file)
        for c in class_list:
            out_class_file.write(c+'\n')

        out_class_file.close()
        out_file.close()
        self.save_verified_status(target_file)

    def save_verified_status(self, target_annotation_file_path):
        dir_path = os.path.dirname(os.path.realpath(target_annotation_file_path))
        file_name = os.path.split(target_annotation_file_path)[1]
        verified_status_file = os.path.join(dir_path, "verified_status.json")
        json_dict = {}
        if os.path.isfile(verified_status_file):
            with open(verified_status_file, 'r') as json_file:
                json_dict = json.load(json_file)

        json_dict[file_name] = self.verified
        with open(verified_status_file, 'w+') as json_file:
            json.dump(json_dict, json_file)


class YoloReader:

    def __init__(self, file_path, image, class_list_path=None):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
        self.shapes = []
        self.file_path = file_path

        if class_list_path is None:
            dir_path = os.path.dirname(os.path.realpath(self.file_path))
            self.class_list_path = os.path.join(dir_path, "classes.txt")
        else:
            self.class_list_path = class_list_path

        # print (file_path, self.class_list_path)

        classes_file = open(self.class_list_path, 'r')
        self.classes = classes_file.read().strip('\n').split('\n')

        # print (self.classes)

        img_size = [image.height(), image.width(),
                    1 if image.isGrayscale() else 3]

        self.img_size = img_size

        self.verified = self.load_verified_status()
        # try:
        self.parse_yolo_format()
        # except:
        #     pass

    def load_verified_status(self):
        dir_path = os.path.dirname(os.path.realpath(self.file_path))
        file_name = os.path.split(self.file_path)[1]
        verified_status_file = os.path.join(dir_path, "verified_status.json")
        try:
            with open(verified_status_file, 'r') as json_file:
                return json.load(json_file)[file_name]
        except:
            return False

    def get_shapes(self):
        return self.shapes

    def add_shape(self, label, x_min, y_min, x_max, y_max, difficult):

        points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        self.shapes.append((label, points, None, None, difficult))

    def yolo_line_to_shape(self, class_index, x_center, y_center, w, h):
        label = self.classes[int(float(class_index))]

        x_min = max(float(x_center) - float(w) / 2, 0)
        x_max = min(float(x_center) + float(w) / 2, 1)
        y_min = max(float(y_center) - float(h) / 2, 0)
        y_max = min(float(y_center) + float(h) / 2, 1)

        x_min = round(self.img_size[1] * x_min)
        x_max = round(self.img_size[1] * x_max)
        y_min = round(self.img_size[0] * y_min)
        y_max = round(self.img_size[0] * y_max)

        return label, x_min, y_min, x_max, y_max

    def parse_yolo_format(self):
        bnd_box_file = open(self.file_path, 'r')
        for bndBox in bnd_box_file:
            class_index, x_center, y_center, w, h = bndBox.strip().split(' ')
            label, x_min, y_min, x_max, y_max = self.yolo_line_to_shape(class_index, x_center, y_center, w, h)

            # Caveat: difficult flag is discarded when saved as yolo format.
            self.add_shape(label, x_min, y_min, x_max, y_max, False)


class YoloReaderFromPred(YoloReader):
    def __init__(self, pred, image, class_list_path):
        self.img_size = [image.height(), image.width()]
        self.shapes = []
        self.class_list_path = class_list_path
        classes_file = open(self.class_list_path, 'r')
        self.classes = classes_file.read().strip('\n').split('\n')
        self.parse_yolo_format(pred)
        self.verified = False

    def parse_yolo_format(self, pred):
        for bndBox in pred:
            x1, y1, x2, y2, _, class_index = bndBox
            # class_index, x_center, y_center, w, h = bndBox.strip().split(' ')
            x_center = (x1 + x2) / (2 * self.img_size[1])
            y_center = (y1 + y2) / (2 * self.img_size[0])
            w = (x2 - x1) / self.img_size[1]
            h = (y2 - y1) / self.img_size[0]
            label, x_min, y_min, x_max, y_max = self.yolo_line_to_shape(class_index, x_center, y_center, w, h)

            # Caveat: difficult flag is discarded when saved as yolo format.
            self.add_shape(label, x_min, y_min, x_max, y_max, False)