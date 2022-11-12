#!/usr/bin/env python3
# coding:utf-8
"""
"""
import abc
from algorithms.algorithm_base import AlgorithmBase
from utils.image_func import imread
from utils.split_detector import SPLITINFERENCE
import time
import logging

logger = logging.getLogger(__name__)


class ObjectDetection(AlgorithmBase):
    """
    """
    default_args = {
        "confThreshold": 0.4,
        "categories": [
            {"label_id": 1, "name": "xxx"},
        ],
        "category_color_map": {
            "xxx": (0, 0, 255)
        },
        "model_serving": {
            "version": None,
            "model_spec_name": "",
            "model_spec_signature_name": "serving_default",
            "inputs": [{'node_name': 'input', 'node_type': 'FP32', 'node_shape': [1, -1, -1, 3]}],
            "outputs": [{'node_name': 'output', 'node_type': 'FP32', 'node_shape': [-1, 7]}]
        }
    }

    def __init__(self, model_serving):
        super().__init__(model_serving)

    def get_split_num(self, args):
        split_width = 1
        split_height = 1
        if 'splitWidth' in args:
            split_width = args['splitWidth']
        if 'splitHeight' in args:
            split_height = args['splitHeight']
        return split_height, split_width

    def call(self, image_path, params=None):
        """inference

        Args:
            image_path: the path of image;
            params: a dict of args;

        Returns:
            inference result
        """
        image = imread(image_path, format='RGB')
        args = params['parameters'][0]
        result = self.call_image_batch(image_list=[image], params=args)[0]
        return result

    def _preprocess(self, image_list, params, result_dict_list):
        """ call preprocess
        """
        outputs = self.preprocess(image_list, params, result_dict_list)
        return outputs

    @abc.abstractmethod
    def preprocess(self, image_list: list, params: dict, result_dict_list: list) -> list:
        """some func before inference
        Args:
            image_list: a list of ndarray image.
            inference_inputs: a list of inference input info
        Returns:
            a inference inputs including image data.
        """
        return NotImplemented

    def _postprocess(self, inference_outputs, params: dict, result_dict_list):
        """ call postprocess
        """
        result_dict_list = self.postprocess(inference_outputs, params, result_dict_list)
        assert 'prediction' in result_dict_list[0], f'prediction not in keys list: {result_dict_list[0].keys()}'
        assert 'title' in result_dict_list[0], f'title not in keys list: {result_dict_list[0].keys()}'
        return result_dict_list

    @abc.abstractmethod
    def postprocess(self, inference_outputs: list, params: dict, result_dict_list: list):
        """some func after inference.

        Args:
            inference_outputs: a list of inference outputs.
            result_dict_list: form preprocess.
        Returns:
            a list of result dict
        """
        return NotImplemented

    def call_image_batch(self, image_list, params={}):
        """Convert image list to inference data and get output from inference
        Args:
            image_list: a list of ndarray image
            params: a dict including some params
        Returns:
            a list of result dict
        """
        params = self.merge_args(params, self.default_args)
        if 'splitWidth' in params or 'splitHeight' in params:
            split_height, split_width = self.get_split_num(params)
            outputs = self._call_split_image_batch(image_list=image_list, params=params, splitWidth=split_width,
                                                   splitHeight=split_height)
        else:
            outputs = self._call_image_batch(image_list, params)
        return outputs

    @SPLITINFERENCE()
    def _call_split_image_batch(self, image_list, params={}):
        outputs = self._call_image_batch(image_list=image_list, params=params)
        return outputs

    def _call_image_batch(self, image_list, params={}):
        """Convert image list to inference data and get output from inference
        Args:
            image_list: a list of ndarray image
            params: a dict including some params
        Returns:
            a list of result dict
        """
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

        result_dict_list = self._postprocess(outputs, params, result_dict_list)

        for inference_using_time, result_dict in zip(using_time, result_dict_list):
            result_dict['usingTime'] = inference_using_time
            result_dict['categories'] = params['categories']
            result_dict["category_color_map"] = params["category_color_map"]

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
        inference_outputs = self.model_serving.inference(model_serving_configs['model_spec_name'], request_configs)
        inference_using_time = time.time() - inference_start

        return inference_outputs, inference_using_time
