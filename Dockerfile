# CUDA 12.1을 지원하는 Python 3.11 기본 이미지 사용
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.11
ENV PATH /opt/conda/bin:$PATH
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
# CUDA 컴파일 강제 설정 추가
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"

# 기본 시스템 패키지 설치 (개발 도구 추가)
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    python3-dev \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    pkg-config \
    cmake \
    ninja-build \
    g++ \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Miniconda 설치
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Conda 초기화 및 환경 설정
RUN conda init bash && \
    echo "conda activate maskdino" >> ~/.bashrc

# Python 3.11 환경 생성
RUN conda create -n maskdino python=$PYTHON_VERSION -y

# Conda 환경 활성화 및 패키지 설치를 위한 쉘 스크립트
SHELL ["conda", "run", "-n", "maskdino", "/bin/bash", "-c"]

# PyTorch 및 기본 의존성 설치
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# OpenCV 설치
RUN pip install -U opencv-python

RUN pip install numpy==1.23.5

RUN pip install albumentations

RUN pip install argparse

# Detectron2 설치 전 필요한 의존성 설치
RUN pip install cython

# Detectron2 설치 (에러가 발생하던 부분 수정)
RUN pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'

# 나머지 요구사항 설치
RUN pip install 'git+https://github.com/cocodataset/panopticapi.git' && \
    pip install 'git+https://github.com/mcordts/cityscapesScripts.git'

# 작업 디렉토리 설정
WORKDIR /maskdino

# MaskDINO 클론 및 설치
RUN git clone https://github.com/luckycontrol/MaskDINO.git . && \
    pip install -r requirements.txt

# Pillow 버전 수정
RUN pip install Pillow==9.5.0

# Mask 연산 컴파일
RUN cd maskdino/modeling/pixel_decoder/ops && \
    FORCE_CUDA=1 python setup.py build install && \
    sh make.sh

# 기본 작업 디렉토리로 복귀
WORKDIR /maskdino

# 볼륨 마운트 포인트 생성
VOLUME ["/maskdino/datasets", "/maskdino/output", "/maskdino/weights"]

# 기본 실행 명령어 설정
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "maskdino"]
CMD ["python", "train_net.py"]