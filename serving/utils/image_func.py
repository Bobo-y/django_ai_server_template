#!/usr/bin/env python
# coding:utf-8

import datetime
import uuid

namespace = uuid.NAMESPACE_URL
import cv2
import logging

logger = logging.getLogger(__name__)

RESIZE_MODE_MAP = {'inter_area': cv2.INTER_AREA}
internal_aliyun = True


def imread(image_path, format='BGR'):
    try:
        image = cv2.imread(image_path)
        if format.upper() == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        unused_image_shape = image.shape
    except Exception as err:
        logger.error(err)
        raise (f'Can not load image from {image_path}')
    return image


def resize_image(input_image, target_size=416, mode=None):

    img = input_image.copy()
    (rows, cols, _) = img.shape
    if mode:
        img = cv2.resize(img, (int(target_size), int(target_size)), RESIZE_MODE_MAP[mode])
    else:
        img = cv2.resize(img, (int(target_size), int(target_size)))

    scale = [float(target_size) / cols, float(target_size) / rows]

    return img, scale


def imwrite(image_path, image, format='BGR'):
    if format.upper == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        cv2.imwrite(image_path, image)
    except Exception as err:
        logger.exception(err)
        raise(err)
