import copy
from utils.image_func import imread
from algorithms.algorithm_base import AlgorithmBase
from algorithms.args_tools import title_to_name_map
from algorithms.args_tools import categories_to_id_map
from algorithms.args_tools import categories_to_name_map


class PipelineBase(AlgorithmBase):
    def __init__(self, model_serving):
        super().__init__(model_serving)
        self.det_server = None
        self.cls_server_list = []
        self.crop_name_list = []
    
    def preprocess(self, image_list, params, result_dict_list):
        return None
    
    def postprocess(self, inf_outputs, params, result_dict_list):
        return result_dict_list

    def call(self, image_path, params={}):
        image = imread(image_path, format='RGB')
        if "parameters" in params:
            params = params['parameters']
            response = self.call_image_batch([image], params)[0]
        else:
            response = self.call_image_batch([image], [])[0]
        return response

    def call_image_batch(self, image_list, params=[]):
        if len(params) == 0:
            for i in range(1 + len(self.cls_server_list)):
                params.append({})
        assert len(params) == 1 + len(self.cls_server_list)

        det_outputs = self.det_server.call_image_batch(image_list, params[0])
        cls_image_list, sub_det_data_list = self.det_data_to_cls_data(image_list, det_outputs, self.crop_name_list)
        assert len(self.cls_server_list) == len(cls_image_list[0])
        cls_outputs_list = []
        for cls_images in cls_image_list:
            sub_cls_output = []
            for cls_index in range(len(self.cls_server_list)):
                cls_outputs = self.cls_server_list[cls_index].call_image_batch(cls_images[cls_index], params[1 + cls_index])
                sub_cls_output.append(cls_outputs)  
            cls_outputs_list.append(sub_cls_output)

        outputs = self.merge_cls_to_det_batch(cls_outputs_list, sub_det_data_list)
        return outputs

    def det_data_to_cls_data(self, image_list, det_outputs, crop_name_list):
        return self.crop_image_batch(image_list, det_outputs, crop_name_list)

    def crop_image_batch(self, image_list, det_outputs, crop_name_list):
        """
        Args:
            image_list:
            det_outputs:
            crop_name_list:
        Returns:
            cls_sub_image_list: image batch
            sub_det_data_list:
        """
        cls_sub_image_list = []
        sub_det_data_list = []
        for image, det_output in zip(image_list, det_outputs):
            sub_images, sub_det_data = self.crop_image(image, det_output, crop_name_list)
            cls_sub_image_list.append(sub_images)
            sub_det_data_list.append(sub_det_data)
        return cls_sub_image_list, sub_det_data_list

    def crop_image(self, image, det_output, crop_name_list):
        """
        Args:
        Returns:
            sub_images: 
            sub_det_data_list: 
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
            sub_image = image[ymin:ymax, xmin:xmax]
            # add data to sub
            category_name = id_category_map[int(bbox[label_id_index])]
            for index in range(len(crop_name_list)):
                if category_name in crop_name_list[index]:
                    sub_det_data_list[index]['prediction'].append(bbox)
                    sub_images[index].append(sub_image)

        return sub_images, sub_det_data_list

    def merge_cls_to_det_batch(self, cls_outputs, sub_det_data_list):
        outputs = []
        for image_cls_output, sub_det_output in zip(cls_outputs, sub_det_data_list):
            output = self.merge_mult_cls_to_det(image_cls_output, sub_det_output)
            outputs.append(output)
        return outputs

    def merge_mult_cls_to_det(self, image_cls_output, sub_det_output):
        categories = copy.deepcopy(sub_det_output[0]['categories'])
        category_color_map = copy.deepcopy(sub_det_output[0]['category_color_map'])
        using_time = sub_det_output[0]['usingTime']
        for cls_out, det_out in zip(image_cls_output, sub_det_output):
            det_out = self.merge_cls_to_det(cls_out, det_out, categories, category_color_map)
        det_output = self.merge_sub_det_out(sub_det_output, categories, category_color_map, using_time)
        return det_output

    def merge_sub_det_out(self, sub_det_output, categories, category_color_map, using_time):
        output = copy.deepcopy(sub_det_output[0])
        output['prediction'] = []
        for sub_det in sub_det_output:
            output['prediction'] += sub_det['prediction']
            using_time += sub_det['usingTime']
        output['categories'] = categories
        output['category_color_map'] = category_color_map
        output['usingTime'] = using_time
        return output

    def merge_cls_to_det(self, cls_output, det_output, categories, category_color_map):
        title_index_map = title_to_name_map(det_output['title'])
        det_name_id_map = categories_to_name_map(categories)
        max_index = max(det_name_id_map.values()) + 1 

        det_label_id_index = title_index_map['label_id']
        det_score_index = title_index_map['confidence_score']

        inf_using_time_list = []
        for cls_out, det_bbox in zip(cls_output, det_output['prediction']):
            if len(cls_out['prediction']) == 0:
                continue

            title_index_map = title_to_name_map(cls_out['title'])
            cls_label_id_index = title_index_map['label_id']
            cls_score_index = title_index_map['confidence_score']

            # replace id
            det_bbox[det_label_id_index] = max_index + cls_out['prediction'][0][cls_label_id_index]
            # replace score
            det_bbox[det_score_index] = cls_out['prediction'][0][cls_score_index]
            inf_using_time_list.append(cls_out['usingTime'])

        for cls_category in cls_output[0]['categories']:
            cls_category['label_id'] += max_index
        categories += cls_output[0]['categories']
        det_output['usingTime'] = sum(inf_using_time_list)
        category_color_map.update(cls_output[0]['category_color_map'])
        return det_output
