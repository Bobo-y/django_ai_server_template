#!/usr/bin/env python
# coding:utf-8

import os
import copy
import uuid
import requests
import importlib
import logging
import numpy as np
from six.moves import urllib
from .base64 import decode_base64_image

logger = logging.getLogger(__name__)
namespace = uuid.NAMESPACE_URL


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def load_class_by_code(algorithm_name, package_path='algorithms'):
    """load a algorithm code using algorithm code.
    
    Args:
        algorithm_name: str, the code of algorithm whitch is the same as the algorighm module name;
        package_path: str, the path of loading algorithm modules.
    Returns:
        the main class in algorithm modules
    """
    try:
        ip_module = importlib.import_module('.'+algorithm_name, package=__package__)
    except:
        ip_module = importlib.import_module(package_path+'.'+algorithm_name)
    main_class = getattr(ip_module, 'main_class')
    return main_class


def get_image_id_by_url(image_url: str):
    """Using uuid to generate image id

    Args:
        image_url: an image url.
    Returns:
        a string of image id
    """
    image_id = str(uuid.uuid3(namespace, image_url))
    return image_id


def get_index_image_path(image_id: str, save_dir: str, image_format: str, max_index: int = 100):
    """if file exist, add image index, untill the last index is equal max index.

    Args:
        image_id: an image name;
        save_dir: image save dir;
        image_format: image save format;
        max_index: default 100, the max image index.
    Returns:
        image path
    """
    image_index = 0
    while image_index < max_index:
        image_path = os.path.join(save_dir, f"{image_id}_{image_index}.{image_format}")
        if not os.path.exists(image_path):
            break
        image_index += 1
    return image_path


def get_image_save_path(image_url, save_dir, image_format='jpg'):
    """Using uuid to generate image name

    Args:
        image_url: the image url
        save_dir: a directory to save image.
    Returns:
        a string of image path
    """
    image_id = get_image_id_by_url(image_url)
    image_path = get_index_image_path(image_id, save_dir, image_format=image_format)
    return image_path


def downloadImageWithUrl(image_url: str, save_dir: str, image_format='jpg'):
    """ Download image from an url

    Args:
        image_url: the image url
    Returns:
        image_name: image name
        image_path: the path saved image with the specified url
    Raises:
        FileNotFoundError: can not find the save directory.
        ConnectionError: can not download the image.
    """
    if not os.path.exists(save_dir):
        # raise FileNotFoundError('can not find the save dir.')
        os.makedirs(save_dir)

    try:
        ir = requests.get(image_url, verify=False, timeout=3)
        image_path = get_image_save_path(image_url, save_dir, image_format)
        with open(image_path, 'wb+') as f:
            f.write(ir.content)
    except:
        image_path = get_image_save_path(image_url, save_dir, image_format)
        try:
            image_path, _ = urllib.request.urlretrieve(image_url, image_path)
        except ConnectionError as err:
            logger.error(err)
            raise ConnectionError('can not download image.')

    return image_path


def save_base64_image(image_str, image_save_dir):
    """save base64 image to a file.

    Args:
        image_str: a str of base64 image;
        image_save_dir: a dir to save image;
    Returns:
        image path
    Raises:
        Exception: can not save image.
    """
    image_data = decode_base64_image(image_str)
    image_id = uuid.uuid1()
    image_path = get_index_image_path(image_id, image_save_dir, image_format='jpg')

    try:
        with open(image_path, 'wb') as f:
            f.write(image_data)
    except Exception as err:
        logger.exception(err)
        raise(err)

    return image_path


def decode_image(image_str, image_save_dir):
    imgExt = ['jpg', 'jpeg', 'png', 'bmp']
    if image_str.startswith('http'):
        image_path = downloadImageWithUrl(image_str, image_save_dir)
    elif is_image_file(image_str): 
        assert os.path.exists(image_str), image_str
        image_path = image_str
    else:
        image_path = save_base64_image(image_str, image_save_dir)
    return image_path


def decode_request_data(request_data: dict, image_save_dir: str):
    """decode request json body and save image to local host.
    """
    args = copy.deepcopy(request_data)
    if 'image' in args:
        image = args.pop('image')
        if type(image).__name__=='list':
            image_all = []
            for image_one in image:
                image_all.append(decode_image(image_one, image_save_dir))
            args['image_path'] = image_all
    else:
        args['image_path'] = decode_image(args['image_path'], image_save_dir)
    return args


def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def is_video_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    IMG_EXTENSIONS = ['.mp4', '.avi', '.MP4', '.AVI', '.mov']
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)