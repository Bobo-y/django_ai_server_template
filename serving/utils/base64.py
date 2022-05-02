import os
import re
import base64
import time


def isBase64(input_str: str):
    pattern = "^([A-Za-z0-9+/]{4})*([A-Za-z0-9+/]{4}|[A-Za-z0-9+/]{3}=|[A-Za-z0-9+/]{2}==)$"
    return re.match(pattern, input_str)


def decode_base64_image(image_base64):
    imgExt = ['jpg', 'jpeg', 'png', 'bmp']
    if 'data:image' == image_base64[:10]:
        image_head, image_data = image_base64.split(',')
        image_format = image_head.split('/')[1].split(';')[0]
        if image_format in imgExt:
            pass
        else:
            raise KeyError(f'ERROR: can only accept {imgExt} format image, not {image_format}')
    else:
        match_result = isBase64(image_base64)
        if match_result is None:
            raise KeyError(f'ERROR: {image_base64} is not base64 string')
        else:
            image_data=image_base64
    return base64.decodestring(image_data.encode())


def save_base64image(image_base64: str, image_id: str, save_dir: str, default_format="jpg"):
    imgExt = ['jpg', 'jpeg', 'png', 'bmp']
    if 'data:image' == image_base64[:10]:
        image_head, image_data = image_base64.split(',')
        image_format = image_head.split('/')[1].split(';')[0]
        if image_format in imgExt:
            pass
        else:
            raise KeyError(f'ERROR: can only accept {imgExt} format image, not {image_format}')
    else:
        match_result = isBase64(image_base64)
        if match_result is None:
            raise KeyError(f'ERROR: {image_base64} is not base64 string')
        else:
            image_format = default_format
            image_data=image_base64
    
    image_save_path = os.path.join(
        save_dir, '{}.{}'.format(image_id, image_format))
    
    try:
        with open(image_save_path, 'wb') as f:
            f.write(base64.decodestring(image_data.encode()))
    except Exception as err:
        raise(err)

    return image_save_path
