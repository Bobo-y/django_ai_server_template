# coding:utf-8
import abc
import logging

from utils.image_func import imread, imwrite, cv_draw_box
from utils import exception
from utils.model_serving_client import NodeInfo, ModelServingServer
from utils.parameters import merge_dict

logger = logging.getLogger(__name__)


class AlgorithmBase(object):
    """Algorithm App base class

    Attributes:
      default_args: a dict including default info.
    """
    default_args = {}

    def __init__(self, model_serving=None):
        self.model_serving = ModelServingServer(model_serving).get_serving()

    @classmethod
    def document(cls):
        document = cls.default_args
        return document

    def merge_args(self, input_configs: dict, default_configs: dict) -> dict:
        """ load default configs to input configs
        """
        return merge_dict(input_configs, default_configs)

    @abc.abstractmethod
    def preprocess(self):
        """preprocess images
        """
        return NotImplemented

    @abc.abstractmethod
    def postprocess(self):
        """postprocess detections
        """
        return NotImplemented

    @abc.abstractmethod
    def call(self):
        """main funciton
        """
        return NotImplemented

    def __call__(self, *args, **kwargs) -> dict:
        """work for API
        """
        try:
            response = self.call(*args, **kwargs)
            self.check_response(response)
        except exception.TFServingError as err:
            logger.exception(err)
            raise exception.TFServingError
        except Exception as err:
            logger.exception(err)
            raise exception.AIServingError(err)
        return response

    def check_response(self, response: dict) -> None:
        """Checke response format
        """
        assert isinstance(response, dict), "response type mast a dict"
        assert "prediction" in response, "Can not find 'prediction' in response"
        assert "imageSize" in response, "Can not find 'imageSize' in response"
        assert "title" in response, "Can not find 'title' in response"
        assert "categories" in response, "Can not find 'categories' in response"

    def convert2node(self, node_info_list: list) -> list:
        """convert a list of node info dict to a list of NodeInfo

        Args:
            node_info_list: a list of node info dict
        Returns:
            a list of NodeInfo
        """
        output_list = []
        for node_info_dict in node_info_list:
            node_info = NodeInfo()
            node_info.from_dict(node_info_dict)
            output_list.append(node_info)
        return output_list

    def draw_result(self,
                    image_path: str,
                    results: dict,
                    save_path: str,
                    category_color_map=None):
        """draw detec result to a image.

        Arguments:
            image_path: a image path.
            results: detection results dict.
            save_path: a save path string.
            category_color_map: a map from category to color.
        """
        image = imread(image_path, format="BGR")
        title_index_map = {t: i for i, t in enumerate(results["title"])}
        categories_id_map = {
            i["label_id"]: i["name"]
            for i in results["categories"]
        }

        for bbox in results["prediction"]:
            label_id = int(bbox[int(title_index_map["label_id"])])
            category = categories_id_map[label_id]
            if category_color_map:
                color = category_color_map[category]
            else:
                color = [0, 255, 0]
            sorted_bbox = [
                bbox[title_index_map["xmin"]], bbox[title_index_map["ymin"]],
                bbox[title_index_map["xmax"]], bbox[title_index_map["ymax"]]
            ]
            cv_draw_box(image, sorted_bbox, color=color)

        imwrite(save_path, image, format="BGR")
        return save_path
