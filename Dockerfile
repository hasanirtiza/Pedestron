FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    build-essential \
    gfortran \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update
    
RUN apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-dev \
    python3.8-distutils && \
    rm -rf /var/lib/apt/lists/*


ENV CUDA_HOME=/usr/local/cuda

RUN curl https://bootstrap.pypa.io/get-pip.py | python3.8 && \
    pip install --upgrade pip && \
    pip install setuptools==58.0.4 wheel Cython "numpy<2.0"

RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

RUN pip install addict==2.4.0 certifi==2022.9.24 charset-normalizer==2.1.1 \
    cycler==0.11.0 idna==3.4 \
    matplotlib==3.3.4 mmcv==0.2.10 \
    mmengine==0.1.0 opencv-python==4.6.0.66 packaging==21.3 \
    Pillow==9.2.0 pycocotools==2.0.5 pyparsing==3.0.9 python-dateutil==2.8.2 \
    PyYAML==6.0 requests==2.28.1 scipy==1.9.1 six==1.16.0 termcolor==2.0.1 \
    terminaltables==3.1.10 typing-extensions==4.3.0 urllib3==1.26.12 yapf==0.32.0

ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

    
RUN git clone https://github.com/mxsrc/Pedestron.git && \
    cd Pedestron && \
    git checkout cu117 && \
    python3.8 setup.py install





