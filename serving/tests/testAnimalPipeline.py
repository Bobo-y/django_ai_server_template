import os
import sys
import unittest
ROOT_Dir='..'
sys.path.append(ROOT_Dir)

from algorithms.AnimalPipeline.animalPipeline import AnimalPipeline
# from utils.tf_serving_client import TFServingClient
from utils.triton_serving_client import TritonServingClient


class TestMathFunc(unittest.TestCase):
    def test_dict(self):

        pic_path=os.path.join('COCO_val2014_000000086615.jpg')

        detect(pic_path)
       

def detect(pic_path):
    model_serving =  {'TF_SERVING': None}
    model_serving['TRITON_SERVING'] = TritonServingClient('0.0.0.0:8001')
    a=AnimalPipeline(model_serving)
    res =  a(pic_path)
    print(res)


if __name__ == "__main__":
    unittest.main(verbosity=2) #all_info
    
