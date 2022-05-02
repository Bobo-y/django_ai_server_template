#!/usr/bin/env python
# coding:utf-8

import os
import logging
import threading
from django.conf import settings
from utils.utils import load_class_by_code
from utils import exception


logger = logging.getLogger(__name__)
internal_aliyun = True
code_class_map = {}

# load model serving and instanti
TF_SERVING_HOST = os.environ['TF_SERVING_HOST'] if 'TF_SERVING_HOST' in os.environ else None
TRITON_SERVING_HOST = os.environ['TRITON_SERVING_HOST'] if 'TRITON_SERVING_HOST' in os.environ else None
MODELS_SERVING = {'TF_SERVING': None, 'TRITON_SERVING': None}

if TF_SERVING_HOST and ':' in TF_SERVING_HOST:
    from utils.tf_serving_client import TFServingClient
    logger.info(f"TF SERVING HOST set: {TF_SERVING_HOST}")
    MODELS_SERVING['TF_SERVING'] = TFServingClient(TF_SERVING_HOST)
if TRITON_SERVING_HOST and ':' in TRITON_SERVING_HOST:
    from utils.triton_serving_client import TritonServingClient
    logger.info(f"TRITON SERVING HOST set: {TRITON_SERVING_HOST}")
    MODELS_SERVING['TRITON_SERVING'] = TritonServingClient(TRITON_SERVING_HOST)

if MODELS_SERVING['TF_SERVING'] is None and MODELS_SERVING['TRITON_SERVING'] is None:
    logger.warning('Can not load model serving.')


class AIServingClient():
    _instance_lock = threading.Lock()

    def __init__(self):
        self.models_serving = MODELS_SERVING

    def __new__(cls, *args, **kwargs):
        if not hasattr(AIServingClient, "_instance"):
            with AIServingClient._instance_lock:
                if not hasattr(AIServingClient, "_instance"):
                    AIServingClient._instance = object.__new__(cls)
        return AIServingClient._instance
        
    def run_algorithm_list(self):
        """run algorithms
        """
        args = self.metadata
        for algorithm_obj in args['algorithms']:
            temp_results = self.run_algorithm(args['image_path'], algorithm_obj)
            self.json_result[algorithm_obj['algorithmCode']] = list(temp_results)

    def run_algorithm(self, image_path, args):
        try:
            temp_results = self._run_algorithm(image_path, args)
        except Exception as err:
            logger.exception(err)
            raise exception.AIServingError(err)
        return temp_results

    def _run_algorithm(self, image_path, args):
        """run one algorithm

        Args:
            image_path: the path of image file;
            args: dict, args for this algorithm.
        Return:
            status_code: status code in func.
            result: algorithm result.
        Raise:
            CloudAIError
        """

        temp_results = []
        algorithm_code = args['algorithmCode']
        thresh = args['confThreshold'] = args['thresh'] if 'thresh' in args else None

        # load algorithm class
        if algorithm_code not in code_class_map:
            code_class_map[algorithm_code] = load_class_by_code(algorithm_code)(self.models_serving)

        try:
            # call inference function
            result = code_class_map[algorithm_code](image_path, args)
            
            save_path = os.path.join(settings.IMAGE_SAVE_DIR, os.path.basename(image_path))
            category_color_map = result["category_color_map"]
            result_image_path = code_class_map[algorithm_code].draw_result(image_path, result, save_path, category_color_map)
            result['saveImageUrl'] = result_image_path
            temp_results.append(result)
        except Exception as err:
            logger.error('%s', err)
            raise err

        return temp_results
