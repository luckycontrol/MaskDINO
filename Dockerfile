# Use CUDA 12.1 with Python 3.11 base image
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04

# Set environment variables with default values
ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/conda/bin:/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" \
    LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}" \
    CPATH="/usr/local/cuda/include:${CPATH:-}" \
    CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    CUDA_ROOT=/usr/local/cuda

# Install system packages
RUN apt-get update && apt-get install -y \
    git wget build-essential python3-dev python3-pip \
    libgl1-mesa-glx libglib2.0-0 pkg-config cmake ninja-build \
    g++ gcc software-properties-common \
    cuda-nvcc-12-1 cuda-cudart-dev-12-1 cuda-command-line-tools-12-1 \
    cuda-runtime-12-1 cuda-libraries-dev-12-1 cuda-minimal-build-12-1 \
    libcublas-dev-12-1 && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
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

# Initialize Conda and create environment
RUN conda init bash && \
    conda create -n maskdino python=3.11 -y

# Activate Conda environment and install packages
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate maskdino && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -U opencv-python numpy==1.23.5 albumentations argparse && \
    pip install -U pip cython 'pyyaml>=5.1' setuptools>=59.5.0 ninja && \
    git clone https://github.com/MaureenZOU/detectron2-xyz.git && \
    cd detectron2-xyz && \
    TORCH_CUDA_ARCH_LIST='6.0 6.1 7.0 7.5 8.0 8.6+PTX' \
    FORCE_CUDA=1 \
    CC=gcc \
    CXX=g++ \
    CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    python setup.py build develop && \
    pip install 'git+https://github.com/cocodataset/panopticapi.git' 'git+https://github.com/mcordts/cityscapesScripts.git' && \
    git clone https://github.com/luckycontrol/MaskDINO.git . && \
    pip install -r requirements.txt && \
    pip install Pillow==9.5.0 && \
    cd maskdino/modeling/pixel_decoder/ops && \
    FORCE_CUDA=1 python setup.py build install && \
    sh make.sh"

# Set working directory and volumes
WORKDIR /maskdino
VOLUME ["/maskdino/datasets", "/maskdino/output", "/maskdino/weights"]

# Set entrypoint and default command
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "maskdino"]
CMD ["python", "train_net.py"]