FROM 10.133.11.71:8800/accv-train/cuda10.1-cudnn7.5.1-dev-ubuntu16.04-opencv4.1.1-torch1.4.0-openvino2020r2-haier

RUN rm -rf /usr/local/ev_sdk && mkdir -p /usr/local/ev_sdk
COPY ./ /usr/local/ev_sdk

RUN \
    cd /usr/local/ev_sdk && mkdir -p build && rm -rf build/* \
    && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j4 install && rm -rf ../build/*

RUN python3.6 -m pip install numpy
RUN python3.6 -m pip install opencv_python
RUN python3.6 -m pip install loguru
RUN python3.6 -m pip install scikit-image
RUN python3.6 -m pip install tqdm
RUN python3.6 -m pip install Pillow
RUN python3.6 -m pip install thop
RUN python3.6 -m pip install ninja
RUN python3.6 -m pip install tabulate
RUN python3.6 -m pip install tensorboard
RUN python3.6 -m pip install torch==1.8.0
RUN python3.6 -m pip install torchvision==0.9.0
RUN python3.6 -m pip install onnx==1.8.1
RUN python3.6 -m pip install onnxruntime==1.8.0
RUN python3.6 -m pip install onnx-simplifier==0.3.5
RUN python3.6 -m pip install typing-extensions==3.7.4.3
RUN python3.6 -m pip install protobuf==3.8.0


RUN python3.6 -m pip install cython
RUN python3.6 -m pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
ENV AUTO_TEST_USE_JI_PYTHON_API=1