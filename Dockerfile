# --- Stage 1: 빌드 환경 ---
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04 AS builder

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST=7.5
    

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget \
    g++ gcc && \
    rm -rf /var/lib/apt/lists/*

# Miniconda 설치
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$ARCH" = "aarch64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    wget $MINICONDA_URL -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Conda 환경 활성화 및 패키지 설치
RUN /opt/conda/bin/conda init bash && \
    /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda create -n maskdino python=3.11 -y && \
    conda activate maskdino && \
    conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y && \
    pip install -U opencv-python numpy==1.23.5 albumentations argparse && \
    conda install -c conda-forge setuptools ninja -y && \
    pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git' && \
    pip install 'git+https://github.com/cocodataset/panopticapi.git' 'git+https://github.com/mcordts/cityscapesScripts.git' && \
    git clone https://github.com/luckycontrol/MaskDINO.git && \
    cd MaskDINO && \
    pip install -r requirements.txt && \
    pip install Pillow==9.5.0 && \
    cd maskdino/modeling/pixel_decoder/ops && \
    sh make.sh && \
    find . -name '*.o' -delete && \
    cd / && \
    conda clean -afy "

# 빌드된 파일 복사를 위한 임시 디렉토리 생성 및 파일 복사
RUN mkdir /app && \
    cp -r /MaskDINO /app/MaskDINO && \
    cp -r /opt/conda /app/conda

# --- Stage 2: 실행 환경 ---
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04

# 환경 변수 설정 (필요한 변수만)
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/conda/bin:$PATH \
    CONDA_AUTO_ACTIVATE_BASE=false

# Stage 1에서 빌드된 파일 복사
COPY --from=builder /app/MaskDINO /maskdino
COPY --from=builder /app/conda /opt/conda

# 작업 디렉토리 설정
WORKDIR /maskdino

# 볼륨 설정
VOLUME ["/maskdino/datasets", "/maskdino/output", "/maskdino/weights"]

SHELL ["/bin/bash", "--login", "-c"]