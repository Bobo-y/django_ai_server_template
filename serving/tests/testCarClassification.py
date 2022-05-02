import os
import sys
import unittest
ROOT_Dir='..'
sys.path.append(ROOT_Dir)


from algorithms.CarClassification.carClassification import CarClassification
# from utils.tf_serving_client import TFServingClient
from utils.triton_serving_client import TritonServingClient


class TestMathFunc(unittest.TestCase):
    def test_dict(self):
        args = {
            "parameters": [
                {
                    "confThreshold": 0.5,
                    "model_serving": {
                        "version": 1,
                        "model_spec_name": "car_classify",
                        "model_spec_signature_name": ""
                    }
                }],
        }

        pic_path=os.path.join('COCO_train2014_000000427895_0_airplane.jpg')

        detect(pic_path, args)
       

def detect(pic_path,args):
    model_serving =  {'TF_SERVING': None}
    model_serving['TRITON_SERVING'] = TritonServingClient('0.0.0.0:8001')
    a=CarClassification(model_serving)
    res =  a(pic_path,args)
    print(res)


if __name__ == "__main__":
    unittest.main(verbosity=2) #all_info
    
