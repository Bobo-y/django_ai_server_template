import logging
import os

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.decorators import parser_classes
from rest_framework.decorators import renderer_classes
from rest_framework.parsers import JSONParser
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response


logger = logging.getLogger(__name__)
LOG_LEVEL = 'info' if 'LOG_LEVEL' not in os.environ else os.environ['LOG_LEVEL']
log_level = getattr(logging, LOG_LEVEL.upper(), None)
logging.basicConfig(level=log_level)

import traceback
import time
from django.conf import settings
from algorithms.ai_serving_client import AIServingClient
from algorithms.ai_documents import Docs
from utils.utils import decode_request_data
from utils import exception

ai_server = AIServingClient()


@api_view(['POST'])
@parser_classes((JSONParser,))
@renderer_classes((JSONRenderer,))
def post_one_algorithm(request, *args, **kwargs):
    # get request algorithm name
    algorithm_name = kwargs['algorithm_name']
    logger.info(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    assert algorithm_name is not None

    content = {
        'resultCode': 0,
        'errorString': 'success',
        'results': {}
    }
    try:
        # decode request
        image_save_dir = settings.IMAGE_SAVE_DIR
        if not os.path.exists(image_save_dir):
            os.makedirs(image_save_dir)
        begin = time.time()
        args = decode_request_data(request.data, image_save_dir)
        logger.info('download image time {}'.format(time.time() - begin))
        args['algorithmName'] = algorithm_name
        args['metadata'] = None
        begin = time.time()
        if type(args['image_path']).__name__=='list':
            content_all = []
            for path in args['image_path']:
                content['results'] = ai_server.run_algorithm(path, args)
                content_all.append(content)
            content = content_all
            logger.info("model time {}".format(time.time() - begin))
        else:
            content['results'] = ai_server.run_algorithm(args['image_path'], args)
            logger.info("model time {}".format(time.time() - begin))
        _status = status.HTTP_200_OK
    except exception.TFServingError as err:
        _status = status.HTTP_400_BAD_REQUEST
        content['resultCode'] = exception.TFServingError.code
        content['errorString'] = exception.TFServingError.message
        logger.error(err)

    except exception.AIServingError as err:
        _status = status.HTTP_400_BAD_REQUEST
        content['resultCode'] = exception.AIServingError.code
        content['errorString'] = exception.AIServingError.message
        logger.error(err)

    except Exception as err:
        content['resultCode'] = 500
        content['errorString'] = err
        _status = status.HTTP_400_BAD_REQUEST
        logger.error('%s\n%s\n%s', args, kwargs, request.data)
        logger.error(content)
        logger.error(traceback.format_exc())
    return Response(content, status=_status)


@api_view(['GET'])
@renderer_classes([JSONRenderer])
def get_algorithm_docs(request, *args, **kwargs):
    return Response(data=Docs.documents())