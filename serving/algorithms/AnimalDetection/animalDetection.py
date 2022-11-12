#!/usr/bin/env python3
# coding:utf-8


import logging
from algorithms.object_detection_base import ObjectDetection
from algorithms.AnimalAndCarDetection import main_class as AnimalAndCarDetection
from algorithms.args_tools import title_to_name_map
from algorithms.args_tools import categories_to_name_map
logger = logging.getLogger(__name__)


class AnimalDetection(ObjectDetection):
    """
    """
    default_args = AnimalAndCarDetection.default_args
    
    def __init__(self, model_serving):
        self.model_serving = model_serving
        self.animal_and_car = AnimalAndCarDetection(model_serving)

    def call_image_batch(self, image_list, params={}):
        outputs = self.animal_and_car.call_image_batch(image_list, params)
        outputs = self.postprocess(outputs)
        return outputs

    def postprocess(self, ph_outputs):
        outputs = []
        for ph_output in ph_outputs:
            output = self.postprocess_one(ph_output)
            outputs.append(output)
        return outputs

    def postprocess_one(self, ph_output):
        title_index_map = title_to_name_map(ph_output['title'])
        category_index_map = categories_to_name_map(ph_output['categories'])
        animal_id = category_index_map['animal']

        label_id_index = title_index_map['label_id']
        new_prediction = []
        for index in range(len(ph_output['prediction'])):
            class_id = ph_output['prediction'][index][label_id_index]
            if class_id == animal_id:
                new_prediction.append(ph_output['prediction'][index])
                
        ph_output['prediction'] = new_prediction
        ph_output['categories'] = [{"label_id": animal_id, "name": "animal"}]
        ph_output['category_color_map'] = {"animal": ph_output['category_color_map']['animal']}
        return ph_output