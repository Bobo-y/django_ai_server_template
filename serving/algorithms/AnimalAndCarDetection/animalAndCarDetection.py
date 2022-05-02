#!/usr/bin/env python3
# coding:utf-8
"""Person and Head detection yolo_v5
"""
import numpy as np
import copy
import logging
import cv2
import torch
import torchvision
import time
from algorithms.object_detection_base import ObjectDetection

logger = logging.getLogger(__name__)


class AnimalAndCarDetection(ObjectDetection):
    """yolo_v5 
    """
    default_args = {
        "confThreshold": 0.4,
        "splitWidth": 1,
        "splitHeight": 1,
        "categories": [
            {"label_id": 1, "name": "car"},
            {"label_id": 2, "name": "animal"},
        ],
        "category_color_map": {
            "animal": [0, 255, 0],
            "car": [255, 0, 0]
        },
        "model_serving": {
            "version": 1,
            "model_spec_name": "animal_car_det",
            "model_spec_signature_name": "serving_default",
            "inputs": [{'node_name': 'images', 'node_type': 'FP32', 'node_shape': [3, 640, 640]}],
            "outputs": [{'node_name': 'output', 'node_type': 'FP32', 'node_shape': [1, 25200, 7]}]
        }
    }

    def xywh2xyxy(self, x):
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def non_max_suppression(self, pred, conf_thres=0.4, iou_thres=0.5, classes=0, agnostic=False):
        """Performs Non-Maximum Suppression (NMS) on inference results

        Returns:
             detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
        """
        prediction = torch.from_numpy(pred.astype(np.float32))
        if prediction.dtype is torch.float16:
            prediction = prediction.float()
        nc = prediction[0].shape[1] - 5
        xc = prediction[..., 4] > conf_thres
        min_wh, max_wh = 2, 4096
        max_det = 100
        time_limit = 10.0
        multi_label = nc > 1
        output = [None] * prediction.shape[0]
        t = time.time()
        for xi, x in enumerate(prediction):
            x = x[xc[xi]]
            if not x.shape[0]:
                continue
            x[:, 5:] *= x[:, 4:5]
            box = self.xywh2xyxy(x[:, :4])
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero().t()
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            if classes:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
            n = x.shape[0]
            if not n:
                continue
            c = x[:, 5:6] * (0 if agnostic else max_wh)
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
            if i.shape[0] > max_det:
                i = i[:max_det]
            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                break
        return output

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        if ratio_pad is None:
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]
        coords[:, [0, 2]] -= pad[0]
        coords[:, [1, 3]] -= pad[1]
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)
        return coords

    def clip_coords(self, boxes, img_shape):
        boxes[:, 0].clamp_(0, img_shape[1])
        boxes[:, 1].clamp_(0, img_shape[0])
        boxes[:, 2].clamp_(0, img_shape[1])
        boxes[:, 3].clamp_(0, img_shape[0])

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if auto:
            dw, dh = np.mod(dw, 64), np.mod(dh, 64)
        elif scaleFill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, ratio, (dw, dh)

    def preprocess_one(self, image, params, result_dict):
        inference_inputs = copy.deepcopy(params['model_serving']['inputs'])
        height, width, depth = image.shape
        result_dict['imageSize'] = {"height": height, "width": width, "depth": depth}
        img = self.letterbox(image, new_shape=640)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = img / 255.0
        img = img.astype(np.float32)
        inference_inputs[0]['node_data'] = img
        return inference_inputs

    def preprocess(self, image_list, params, result_dict_list):
        inputs_list = []
        for image, result_dict in zip(image_list, result_dict_list):
            inf_inputs = self.preprocess_one(image, params, result_dict)
            inputs_list.append(inf_inputs)
        return inputs_list

    def postprocess_one(self, inference_output, params, result_dict):
        detections_bs = inference_output['output']
        image_shape = result_dict['imageSize']
        image_shape = [image_shape['height'], image_shape['width'], image_shape['depth']]
        boxes = self.non_max_suppression(detections_bs)
        outputs = []

        if len(boxes) > 0:
            for i, det in enumerate(boxes):
                if det is not None and len(det):
                    det[:, :4] = self.scale_coords((640, 640), det[:, :4], (image_shape[0], image_shape[1], image_shape[2])).round()

                    for *xyxy, conf, cls in det:
                        x_min = (xyxy[0] / float(image_shape[1]))
                        y_min = (xyxy[1] / float(image_shape[0]))
                        x_max = (xyxy[2] / float(image_shape[1]))
                        y_max = (xyxy[3] / float(image_shape[0]))
                        # x_min = int(xyxy[0])
                        # y_min = int(xyxy[1])
                        # x_max = int(xyxy[2])
                        # y_max = int(xyxy[3])
                        score = conf
                        class_id = int(cls) + 1

                        if score > params['confThreshold']:
                            outputs.append([class_id, float(score), x_min, y_min, x_max, y_max])

        result_dict['title'] = ["label_id", "confidence_score", "xmin", "ymin", "xmax", "ymax"]
        result_dict['prediction'] = outputs
        return result_dict

    def postprocess(self, inference_outputs, params, result_dict_list):
        for inf_output, result_dict in zip(inference_outputs, result_dict_list):
            result_dict = self.postprocess_one(inf_output, params, result_dict)
        return result_dict_list