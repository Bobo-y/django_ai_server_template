# This is a server template for deploy AI model with django and tensorflow-serving or triton-server inference backbone, include object detection, classification, etc.
![image](https://user-images.githubusercontent.com/9928596/201560242-e7579d16-e149-402c-8cb8-0d0b836da56a.png)

# deploy demo

1. clone this code

2. start a triton server:
```shell
cd  model_server_demo/triton_server
docker build -t serving-triton:v0.0  .
docker run --runtime=nvidia -p 8000:8000 -p 8001:8001 -p 8002:8002 serving-triton:v0.0
```

3. start a django server (you can move model_server_demo to other dir)
```
  you need change 'ENV TRITON_SERVING_HOST=serving-triton:8001' replace serving-triton by your local ip, such as TRITON_SERVING_HOST=0.0.0.0:8001
```


```shell
cd django_serving

docker build -t algorithms_serving:v0.0 .
docker run -d -p 8211:8211 algorithms_serving:v0.0
```
4. test request

see in serving/tests/request.py

5. more

you can use docker compose to manage algorithms_serving and triton_server
