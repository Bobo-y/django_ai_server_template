#!/usr/bin/env python3
# coding:utf-8

from algorithms.pipeline_base import PipelineBase
from algorithms.AnimalDetection import main_class as AnimalDetection
from algorithms.AnimalClassification import main_class as AnimalClassification
from algorithms.args_tools import title_to_name_map
from algorithms.args_tools import categories_to_id_map
import logging
import copy
logger = logging.getLogger(__name__)


class AnimalPipeline(PipelineBase):
    """反光衣检测流水线
    """

    def __init__(self, model_serving):
        self.model_serving = model_serving
        self.det_server = AnimalDetection(model_serving)
        self.cls_server_list = [AnimalClassification(model_serving)]
        self.crop_name_list = [["animal"]]

    @staticmethod
    def document():
        return {
            'parameters': [
                AnimalDetection.default_args,
                AnimalClassification.default_args
            ]
        }

    def crop_image(self, image, det_output, crop_name_list):
        """
        Args:
        Returns:
            sub_images: n 个分类器的图片数据
            sub_det_data_list: n 个分类器拆分到的检测结果
        """
        title_index_map = title_to_name_map(det_output['title'])
        id_category_map = categories_to_id_map(det_output['categories'])

        # init new data
        sub_det_data_list = []
        sub_images = []
        for name_list in crop_name_list:
            sub_det_data = copy.deepcopy(det_output)
            sub_det_data['prediction'] = []
            sub_det_data_list.append(sub_det_data)
            sub_images.append([])

        ymin_index = title_index_map['ymin']
        xmin_index = title_index_map['xmin']
        ymax_index = title_index_map['ymax']
        xmax_index = title_index_map['xmax']
        label_id_index = title_index_map['label_id']

        height = det_output['imageSize']['height']
        width = det_output['imageSize']['width']

        for bbox in det_output['prediction']:
            ymin = int(bbox[ymin_index] * height)
            xmin = int(bbox[xmin_index] * width)
            ymax = int(bbox[ymax_index] * height)
            xmax = int(bbox[xmax_index] * width)
            w = xmax - xmin
            h = ymax - ymin
            sub_image = image[ymin:ymax, xmin:xmax]
            # add data to sub
            category_name = id_category_map[int(bbox[label_id_index])]
            for index in range(len(crop_name_list)):
                if category_name in crop_name_list[index]:
                    sub_det_data_list[index]['prediction'].append(bbox)
                    sub_images[index].append(sub_image)

        return sub_images, sub_det_data_list