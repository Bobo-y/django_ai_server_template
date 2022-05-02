import os
import importlib
import logging
import traceback

logger = logging.getLogger(__name__)


class Docs(object):

    @staticmethod
    def documents():
        algorithms = os.listdir('./algorithms/')
        pipelines = []
        documents = {}
        for alg in algorithms:
            if alg[0].isupper():
                try:
                    pipelines.append(alg)
                    module_name = alg[0].lower() + alg[1:]
                    algorithm = importlib.import_module('algorithms.' + alg + '.' + module_name)
                    func = getattr(getattr(algorithm, alg), 'document')
                    documents[alg] = func()
                except Exception as e:
                    logger.error(e)
                    logger.error(traceback.format_exc())
                    pipelines.remove(alg)
        return documents