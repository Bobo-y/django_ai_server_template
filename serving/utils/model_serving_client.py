#!/usr/bin/env python3
# coding:utf-8
r"""Model Serving Client Base
"""
import numpy as np
import abc

TYPE_MAPE = {
    "FP32": np.float32,
    "UINT8": np.uint8
}

class ModelInfo(object):
    def __init__(self, model_name):
        self.model_name = model_name
    
    def add_attribute(self, attribute_name):
        pass


class NodeInfo():
    def __init__(self, node_name='input', node_type='FLOAT32', node_shape=[], node_data=None):
        self.node_name = node_name
        self.node_type = node_type
        self.node_shape = node_shape
        self.node_data = node_data

    def from_dict(self, node_dict: dict):
        # must have node_name
        self.node_name = node_dict['node_name']
        if 'node_type' in node_dict:
            self.node_type = node_dict['node_type']
        if 'node_shape' in node_dict:
            self.node_shape = node_dict['node_shape']
        if 'node_data' in node_dict:
            data = self.convert_data_type(node_dict['node_data'], node_dict['node_type'])
            self.node_data = node_dict['node_data']

    def convert_data_type(self, data, type_name):
        new_data = data.astype(TYPE_MAPE[type_name])
        return new_data

    def reshape_data(self, data, shape):
        new_data = data.reshape(shape)
        return new_data


class ModelServingServer(object):
    def __init__(self, model_serving):
        self.load_serving(model_serving)

    def load_serving(self, model_serving):
        if model_serving is None:
            self.model_serving = None
            return
        if model_serving['TF_SERVING'] is not None and model_serving['TRITON_SERVING'] is not None:
            self.model_serving = model_serving['TF_SERVING']
        elif model_serving['TF_SERVING'] is None:
            self.model_serving = model_serving['TRITON_SERVING']
        elif model_serving['TRITON_SERVING'] is None:
            self.model_serving = model_serving['TF_SERVING']
        else:
            pass

    def get_serving(self):
        return self.model_serving


class ModelServingClient(object):
    """Tensorflow Serving Client

    Attributes:
      serving_url: like localhost:8500
    """
    def __init__(self, serving_url):
        """set host and port
        """
        host, port = serving_url.split(':')
        self.host = host
        self.port = int(port)

    @abc.abstractmethod
    def health_check(self, name: str, configs: dict):
        """
        """
        return NotImplementedError

    @abc.abstractmethod
    def set_inputs(self, inputs_configs: dict) -> dict:
        """format a request tensor

        Arguments:
            request_configs: including version and inputs info
        Returns:
          a reqest type
        """
        return NotImplementedError

    @abc.abstractmethod
    def inference(self, name: str, configs: dict) -> dict:
        """a simple wrapper for predict

        Args:
            name: model serving name
            configs: including inputs and outputs info

        Returns:
          a dict including results
        """
        return NotImplementedError