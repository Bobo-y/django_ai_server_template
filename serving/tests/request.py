import requests
import os


url = "http://127.0.0.1:8211/ai/AnimalAndCarPipeline"
data ={"image_path": '/home/yanglin5/Documents/django_serving/serving/tests/COCO_val2014_000000086615.jpg', 'parameters': [{}, {}, {} ]}
r = requests.post(url=url,json=data)
print(r.content)

