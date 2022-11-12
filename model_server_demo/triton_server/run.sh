# cd  in this workspace
# step 1: build docker image
docker build -t serving-triton:v0.0  .
# step 2: start triton server, use nvidia-docker 2
docker run --runtime=nvidia -p 8000:8000 -p 8001:8001 -p 8002:8002 serving-triton:v0.0