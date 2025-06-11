# Base image with CUDA 12.6 and MIG tools
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# Set timezone non-interactively
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/UTC /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# Install system dependencies with Python and MIG support
RUN apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y \
    git \
    cmake \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev \
    gcc-12 \
    g++-12 \
    curl \
    python3.10 \
    python3-pip \
    python3-dev \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 \
    && rm -rf /var/lib/apt/lists/*

# Configure CUDA paths and MIG capabilities
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CUDA_VERSION=12.6
ENV CUDA_HOME=/usr/local/cuda
ENV NVIDIA_DISABLE_REQUIRE=1
ENV NVIDIA_VISIBLE_DEVICES=all

# Install PyTorch 2.6.0 with CUDA 12.6 compatibility
RUN pip3 install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0+cu126 \
    torchaudio==2.6.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126  # Updated to cu126

# Install remaining Python dependencies
RUN pip install --no-cache-dir \
    ninja==1.10.2 \
    torchmetrics \
    torch-fidelity \
    matplotlib \
    pandas \
    click \
    requests \
    tqdm \
    pyspng \
    scikit-learn \
    statsmodels \
    seaborn \
    pycryptodome \
    cryptography \
    lpips \
    imageio-ffmpeg==0.4.3

RUN pip install --no-cache-dir \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    datasets

RUN pip install huggingface_hub[hf_xet]

# Configure writable directories for PyTorch extensions
ENV TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
RUN mkdir -p ${TORCH_EXTENSIONS_DIR} && chmod -R 777 ${TORCH_EXTENSIONS_DIR}

ARG CACHEBUST=1

CMD ["true"]