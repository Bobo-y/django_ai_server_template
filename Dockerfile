FROM tensorflow/tensorflow:latest-gpu AS builder

RUN echo "Asia/Shanghai" > /etc/timezone

RUN mkdir -p /root/workspace
WORKDIR /root/workspace
COPY --chown=root . /root/workspace/django_serving

# python
RUN pip install --upgrade pip \
        && mv django_serving/config/pip.conf /etc/pip.conf \
        && pip install -r django_serving/config/requirements.txt

USER root
WORKDIR /root/workspace/django_serving/serving

# tensorflow server
# ENV TF_SERVING_HOST=serving-tensorflow:8600

# triton server
ENV TRITON_SERVING_HOST=serving-triton:8001


CMD /bin/bash entry_point.sh
