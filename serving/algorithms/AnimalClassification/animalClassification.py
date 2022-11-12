#!/usr/bin/env python3
# coding:utf-8


import numpy as np
from algorithms.classification_base import Classification
from utils.image_func import resize_image
from utils.utils import softmax


class AnimalClassification(Classification):
    """
    """
    default_args = {
        'confThreshold': 0.85,
        "categories": [
            {"label_id": 0, "name": "dog"},
            {"label_id": 1, "name": "cat"},
            {"label_id": 2, "name": "bird"},
            {"label_id": 3, "name": "sheep"},
            {"label_id": 4, "name": "cow"},
            {"label_id": 5, "name": "horse"},

        ],
        "category_color_map": {
            "dog": [255, 0, 0],
            "cat": [0, 0, 255],
            "bird": [0, 255, 0],
            "sheep": [255, 255, 0],
            "cow": [0, 255, 255],
            "horse": [0, 255, 220]
        },
        "model_serving": {
            "version": None,
            "model_spec_name": "animal_classify",
            "model_spec_signature_name": "serving_default",
            "inputs": [{'node_name': 'images', 'node_type': 'FP32', 'node_shape': [3, 112, 112]}],
            "outputs": [{'node_name': 'output', 'node_type': 'FP32', 'node_shape': [1, 6]}],
        }
    }

    def format_image(self, image):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_np, _ = resize_image(image, 112, 'inter_area')
        image_np = image_np.astype(np.float32)
        image_np -= mean
        image_np /= std
        image_np = image_np.transpose(2, 0, 1)
        return image_np

    def preprocess(self, image_list, params, result_dict_list):
        # assert len(image_list) > 0, 'must have input image'
        if len(image_list) == 0:
            return []

        inf_inputs = []
        for image in image_list:
            inf_input = params['model_serving']['inputs']
            input_data = self.format_image(image)
            inf_input[0]['node_data'] = input_data
            inf_inputs.append(inf_input)
            
        if len(inf_inputs) == 0:
            return []

        return inf_inputs

    def postprocess_one(self, inference_output, params, result_dict):
        title = ["label_id", "confidence_score", "xmin", "ymin", "xmax", "ymax"]
        if not inference_output:
            result_dict = {}
            result_dict['prediction'] = []
            result_dict['title'] = title
            result_dict = [result_dict]
            return result_dict

        probs = inference_output['output']
        probs = softmax(probs).squeeze()
        index = np.argmax(probs)
        score = probs[index]
        prediction = [[index, score, 0.0, 0.0, 1.0, 1.0]]
        result_dict['prediction'] = prediction
        result_dict['title'] = title
        return result_dict

    def postprocess(self, inference_outputs, params, result_dict_list):
        if not inference_outputs:
            title = ["label_id", "confidence_score", "xmin", "ymin", "xmax", "ymax"]
            result_dict['prediction'] = []
            result_dict['title'] = title
            result_dict = [result_dict]
            return result_dict
        for inf_output, result_dict in zip(inference_outputs, result_dict_list):
            result_dict = self.postprocess_one(inf_output, params, result_dict)
        return result_dict_list
