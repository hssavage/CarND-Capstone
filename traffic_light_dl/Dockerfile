FROM gcr.io/tensorflow/tensorflow:latest-gpu-py3
RUN apt-get update
RUN apt-get install -y protobuf-compiler python3-pil python3-lxml python3-tk git nano libsm6 libxext6 libxrender-dev
RUN pip3 --no-cache-dir install jupyter
RUN pip3  --no-cache-dir install matplotlib
RUN pip3  --no-cache-dir install Cython
RUN pip3 --no-cache-dir install ruamel-yaml
RUN pip3 --no-cache-dir install opencv-python

RUN mkdir /tensorflow
WORKDIR /tensorflow
RUN git clone https://github.com/tensorflow/models.git

WORKDIR /

RUN mkdir /coco
WORKDIR /coco
RUN git clone https://github.com/cocodataset/cocoapi.git
WORKDIR /coco/cocoapi/PythonAPI
RUN make && cp -r pycocotools /tensorflow/models/research



WORKDIR /tensorflow/models/research
RUN protoc object_detection/protos/*.proto --python_out=.


RUN echo "export PYTHONPATH=${PYTHONPATH}:`pwd`:`pwd`/slim" >> ~/.bashrc

RUN mkdir /workspace
VOLUME ["/workspace"]
WORKDIR /

#Tensor board
EXPOSE 6006

#IPython
EXPOSE 8888


#sudo docker run --runtime=nvidia -p 6006:6006 -p 8888:8888 -v $PWD:/workspace --name=google_api_training_udacity_final -it jlclemon/google_obj:latest
