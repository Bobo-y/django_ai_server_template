#!/usr/bin/env python3
# coding:utf-8
"""image classificatoin base service
"""
import abc
from algorithms.algorithm_base import AlgorithmBase
from utils.image_func import  imread
import time
import logging

logger = logging.getLogger(__name__)


class Classification(AlgorithmBase):
    """model inferecnce
    """
    default_args = {
        'confThreshold': 0.0,
        "categories": [
            {"label_id": 1, "name": "class_one"},
            {"label_id": 2, "name": "class_two"},
        ],
        "category_color_map": {
            "class_one": [0, 0, 255],
            "class_two": [0, 255, 0]
        },
        "model_serving": {
            "version": None,
            "model_spec_name": "classification_spec_name",
            "model_spec_signature_name": "serving_default",
            "inputs": [{'node_name': 'input', 'node_type': 'FP32', 'node_shape': [1, -1, -1, 3]}],
            "outputs": [{'node_name': 'output', 'node_type': 'FP32', 'node_shape': [-1, 7]}]
        }
    }

    def __init__(self, model_serving):
        super().__init__(model_serving)

    def call(self, image_path, params={}):
        """inference

        Args:
            image_path: the path of image;
            params: a dict of args;

        Returns:
            inference result
        """
        image = imread(image_path, format='RGB')
        params = params['parameters'][0]
        result = self.call_image_batch([image], params)[0]
        return result

    def _preprocess(self, image_batch, params, result_dict_list):
        """ call preprocess
        """
        outputs = self.preprocess(image_batch, params, result_dict_list)
        return outputs

    @abc.abstractmethod
    def preprocess(self, image_list, params, result_dict_list):
        """
        """
        return NotImplemented

    def _postprocess(self, inference_outputs, params, result_dict_list):
        """ call postprocess
        """
        result_dict_list = self.postprocess(inference_outputs, params, result_dict_list)
        assert 'prediction' in result_dict_list[0]
        assert 'title' in result_dict_list[0]
        return result_dict_list

    @abc.abstractmethod
    def postprocess(self, inference_outputs, result_dict_list):
        """
        """
        return NotImplemented

    def call_image_batch(self, image_list, params={}):
        """
        """
        # assert len(image_list) > 0, 'must have image'
        params = self.merge_args(params, self.default_args)

        result_dict_list = [{} for _ in range(len(image_list))]

        request_inputs = self._preprocess(image_list, params, result_dict_list)

        model_serving_configs = params['model_serving']
        outputs = []
        using_time = []
        for request_input in request_inputs:
            model_serving_configs['inputs'] = request_input
            inference_outputs, inference_using_time = self.predict(model_serving_configs)
            outputs.append(inference_outputs)
            using_time.append(inference_using_time)

        if len(using_time) == 1:
            using_time = [using_time[0] / len(image_list)] * len(image_list)

        result_dict_list = self._postprocess(outputs, params, result_dict_list)
        print(result_dict_list)
        if len(image_list) > 0:
            for index in range(len(image_list)):
                result_dict_list[index]['usingTime'] = using_time[index]
                rows, cols, depth = image_list[index].shape
                image_size = {'height': rows, 'width': cols, 'depth': depth}
                result_dict_list[index]['imageSize'] = image_size
                result_dict_list[index]['categories'] = params['categories']
                result_dict_list[index]["category_color_map"] = params["category_color_map"]
        else:
            result_dict_list[0]['usingTime'] = 0.0
            image_size = {'height': 0, 'width': 0, 'depth': 0}
            result_dict_list[0]['imageSize'] = image_size
            result_dict_list[0]['categories'] = params['categories']
            result_dict_list[0]["category_color_map"] = params["category_color_map"]

        return result_dict_list

    def predict(self, model_serving_configs):
        """ detect person in the pic, headDetect deal one img; classify deal one_img_one box

        Args:
            model_serving_configs: request info
        Returns:
            inference_outputs: a list of inference outputs.
            inference_using_time: using time for inference
        """
        inputs = model_serving_configs['inputs']
        outputs = model_serving_configs['outputs']
        request_configs = {
            "version": model_serving_configs['version'],
            "model_spec_signature_name": model_serving_configs["model_spec_signature_name"],
            "inputs": self.convert2node(inputs),
            "outputs": self.convert2node(outputs)
        }
        inference_start = time.time()
        inference_outputs = self.model_serving.inference(model_serving_configs['model_spec_name'],
                                                         request_configs)
        inference_using_time = time.time() - inference_start

        return inference_outputs, inference_using_time
