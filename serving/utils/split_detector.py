#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
"""

import logging
import numpy as np
import cv2
from functools import wraps

logger = logging.getLogger(__name__)


def nms_test(bounding_boxes, confidence_score, threshold):
    picked_boxes = []
    picked_score = []
    picked_index = []

    if len(bounding_boxes) == 0:
        return picked_boxes, picked_score, picked_index

    # 边界框
    boxes = np.array(bounding_boxes)
    # 边界框坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    score = np.array(confidence_score)

    areas = (x2 - x1) * (y2 - y1)

    order = np.argsort(score)
    while order.size > 0:
    
        index = order[-1]
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        picked_index.append(index)

        _x1 = np.maximum(x1[index], x1[order[:-1]])
        _y1 = np.maximum(y1[index], y1[order[:-1]])
        _x2 = np.minimum(x2[index], x2[order[:-1]])
        _y2 = np.minimum(y2[index], y2[order[:-1]])
       
        w = np.maximum(0.0, _x2 - _x1 + 1)
        h = np.maximum(0.0, _y2 - _y1 + 1)
        intersection = w * h
       
        ratio = intersection / (areas[index] + areas[order[:-1]] -
                                intersection)
        left = np.where(ratio < threshold)
        order = order[left]
    return picked_boxes, picked_score, picked_index


class SplitDetector():
    def __init__(self, split_width_num=2, split_height_num=2):
        self.split_width_num = split_width_num
        self.split_height_num = split_height_num
        self.move_pads = []
        self.sub_images = []
        self.image_size = {}

    @property
    def split_width(self):
        return self.split_width_num

    @split_width.setter
    def split_width(self, split_width):
        assert isinstance(split_width, int)
        assert split_width > 0
        self.split_width_num = split_width

    @property
    def split_height(self):
        return self.split_height_num

    @split_height.setter
    def split_height(self, split_height):
        assert isinstance(split_height, int)
        assert split_height > 0
        self.split_height_num = split_height

    def split_image(self, image):
        # init
        sub_images = []
        move_pads = []
        image_size = {'width': image.shape[1], 'height': image.shape[0]}
        if len(image.shape) == 2:
            image_size['depth'] = image.shape[2]

        split_width_num = self.split_width_num
        split_height_num = self.split_height_num

        original_height, original_width = image.shape[:2]
        split_height = int(original_height / split_height_num)
        split_width = int(original_width / split_width_num)

        for r in range(split_height_num):
            for c in range(split_width_num):
                top = max(0, int(split_height * r - split_height * 0.2))
                bottom = min(original_height, int(top + split_height * 1.4))
                left = max(0, int(split_width * c - split_width * 0.2))
                right = min(original_width, int(left + split_width * 1.4))
                sub_image = image[top:bottom, left:right]
                sub_images.append(sub_image)
                move_pads.append({'left': left, 'top': top})
        return sub_images, move_pads, image_size

    def split_image_list(self, image_list):
        sub_image_list = []
        move_pads_list = []
        image_size_list = []
        for image in image_list:
            sub_image, move_pads, image_size = self.split_image(image)
            sub_image_list += sub_image
            move_pads_list += move_pads
            image_size_list.append(image_size)
        return sub_image_list, move_pads_list, image_size_list

    def add_movepad(self, output, move_pad, data_format='YXYX'):
        data = output['prediction']
        title_index_map = {key: index for index, key in enumerate(output['title'])}

        # current only accept dict format
        assert isinstance(
            data, list), "data format must be dict like: [label, score, xmin, ymin, xmax, ymax]"

        if data_format != 'YXYX':
            raise ('Current only sopport YXYX format bbox')


        if isinstance(data, list):
            for i, box in enumerate(data):
                data[i][title_index_map['ymin']] += move_pad['top']
                data[i][title_index_map['xmin']] += move_pad['left']
                data[i][title_index_map['ymax']] += move_pad['top']
                data[i][title_index_map['xmax']] += move_pad['left']
        else:
            data[k][:, title_index_map['ymin']] = data[k][:, title_index_map['ymin']] + move_pad['top']
            data[k][:, title_index_map['xmin']] = data[k][:, title_index_map['xmin']] + move_pad['left']
            data[k][:, title_index_map['ymax']] = data[k][:, title_index_map['ymax']] + move_pad['top']
            data[k][:, title_index_map['xmax']] = data[k][:, title_index_map['xmax']] + move_pad['left']
        return output

    def filter_edge(self, output, pass_side=[], edge_width=0.05):
        """filter edge predictions and convert coordiate to absolute
        """
        data = output['prediction']
        height = output['imageSize']['height']
        width = output['imageSize']['width']

        title_index_map = {key: index for index, key in enumerate(output['title'])}
    
        height_edge = int(height * edge_width)
        width_edge = int(width * edge_width)

        filtered_data = []
        for bbox in data:

            _ymin = bbox[title_index_map['ymin']] = bbox[title_index_map['ymin']] * height
            _xmin = bbox[title_index_map['xmin']] = bbox[title_index_map['xmin']] * width
            _ymax = bbox[title_index_map['ymax']] = bbox[title_index_map['ymax']] * height
            _xmax = bbox[title_index_map['xmax']] = bbox[title_index_map['xmax']] * width

            if 'left' not in pass_side and _xmin < width_edge:
                continue
            elif 'right' not in pass_side and _xmax > width - width_edge:
                continue
            elif 'top' not in pass_side and _ymin < height_edge:
                continue
            elif 'bottom' not in pass_side and _ymax > height - height_edge:
                continue
            else:
                filtered_data.append(bbox)
        output['prediction'] = filtered_data
        return output

    def merge_outputs(self,
                      outputs,
                      move_pads,
                      image_size,
                      conf_threshold=0.35,
                      nms_threshold=0.45):
        """make sure x1, y1, x2, y2 were at top4 for every output line
        """
        merged_outputs = {'prediction': [], 'imageSize': {}, 'title': ''}

        assert 'prediction' in merged_outputs, "[ERROR] there mast be 'prediction' in outputs"
        assert 'imageSize' in merged_outputs, "[ERROR] there mast be 'imageSize' in outputs"
        assert 'title' in merged_outputs, "[ERROR] there mast be 'title' in outputs"
        prediction_title = outputs[0]['title']
        title_index_map = {key: index for index, key in enumerate(prediction_title)}
        score_index = title_index_map['confidence_score']
        xmin_index = title_index_map['xmin']
        ymin_index = title_index_map['ymin']
        xmax_index = title_index_map['xmax']
        ymax_index = title_index_map['ymax']
        
        merged_output = {}
        for index, (output, move_pad) in enumerate(zip(outputs, move_pads)):
            # get output data
            if len(output['prediction']) > 0:
                assert isinstance(
                    output['prediction'], list
                ), '[ERROR] there mast be [[label, score, xmin, ymin, xmax, ymax]] format dict in prediction'
            assert prediction_title == output['title']

            # filter edge objs
            pass_side = []
            if index < self.split_width_num:
                pass_side.append('top')
            if (index + self.split_width_num
                ) >= self.split_width_num * self.split_height_num:
                pass_side.append('bottom')
            if index % self.split_height_num == 0:
                pass_side.append('left')
            if (index + 1) % self.split_height_num == 0:
                pass_side.append('right')

            output = self.filter_edge(output,
                                    pass_side=pass_side)

            # move pixcel
            output = self.add_movepad(output, move_pad)

            for k in output.keys():
                if k not in merged_output.keys():
                    merged_output[k] = []
                if isinstance(output[k], list):
                    merged_output[k] += output[k]
                else:
                    merged_output[k].append(output[k])

        temp_bboxes = [[b[ymin_index], b[xmin_index], b[ymax_index], b[xmax_index]] for b in merged_output['prediction']]
        temp_scores = [b[score_index] for b in merged_output['prediction']]

        _temp_bboxes, _temp_scores, indices = nms_test(
            temp_bboxes, temp_scores, nms_threshold)

        merged_output['prediction'] = [merged_output['prediction'][i] for i in indices]
        merged_output['imageSize'] = image_size
        merged_output['title'] = prediction_title
        merged_output['categories'] = outputs[0]['categories']
        merged_output['category_color_map'] = outputs[0]['category_color_map']
        merged_output['usingTime'] = sum([op['usingTime'] for op in outputs])

        # convert coordinate to relative
        for score_bbox in merged_output['prediction']:
            score_bbox[ymin_index] /= image_size['height']
            score_bbox[xmin_index] /= image_size['width']
            score_bbox[ymax_index] /= image_size['height']
            score_bbox[xmax_index] /= image_size['width']

        return merged_output

    def inference(self, image, inference_func, **kwargs):
        sub_images, move_pads = self.split_image(image, self.split_width_num,
                                                 self.split_height_num)
        sub_outputs = []
        for index, sub_image in enumerate(sub_images):
            sub_output = inference_func(sub_image, kwargs)
            sub_outputs.append(sub_output)
        outputs = self.merge_outputs(sub_outputs, move_pads)
        return outputs


def SPLITINFERENCE():
    def decorate(func):
        # init split
        spliter = SplitDetector()
        default_split_width = 1
        default_split_height = 1

        @wraps(func)
        def wrapper(*args, **kwargs):
            if isinstance(args[0], list) and isinstance(args[0][0], np.ndarray):
                image_list = args.pop(0)
            elif 'image_list' in kwargs:
                image_list = kwargs.pop('image_list')
            else:
                raise ValueError(
                    '[ERROR] mast inclued image_list in kwargs or as the first param'
                )
            
            if 'splitWidth' in kwargs:
                spliter.split_width = kwargs.pop('splitWidth')
            else:
                spliter.split_width = default_split_width
            if 'splitHeight' in kwargs:
                spliter.split_height = kwargs.pop('splitHeight')
            else:
                spliter.split_height = default_split_height
            # logger.info(f'split width: {spliter.split_width} split height: {spliter.split_height}')
            sub_images, move_pads, original_image_size = spliter.split_image_list(image_list)

            outputs = func(*args, image_list=sub_images, **kwargs)
            assert len(outputs) == len(sub_images), '[ERROR] the shape of outputs must the same as input images'

            sub_num = spliter.split_width * spliter.split_height
            outputs_list = []
            for index in range(len(image_list)):
                start = index * sub_num
                end = start + sub_num
                merged_outputs = spliter.merge_outputs(outputs[start:end], move_pads[start:end], original_image_size[index])
                outputs_list.append(merged_outputs)
            return outputs_list

        return wrapper

    return decorate