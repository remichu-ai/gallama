#FROM nvcr.io/nvidia/pytorch:23.07-py3
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
LABEL authors="remichu"
WORKDIR /app
# Enable CUDA for llama cpp
ENV CMAKE_ARGS="-DLLAMA_CUDA=on"

# Set timezone to Asia/Singapore (you can change this to any Asian timezone)
ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# setting build related env vars for llama cpp python
ENV CUDA_DOCKER_ARCH=all
ENV GGML_CUDA=1

# Update and install dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    gcc \
    curl \
    ca-certificates \
    sudo \
    ninja-build \
    git \
    wget \
    software-properties-common \
    tzdata \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd \
    && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA for Python 3.12 and install Python 3.12
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.11 get-pip.py \
    && rm get-pip.py

# Set Python 3.12 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

#ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
# Install depencencies
RUN python3 -m pip install --upgrade pip cmake scikit-build
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

ADD requirements.txt /app
RUN pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip install ninja
RUN pip install packaging
RUN pip install flash-attn==2.6.3 --no-build-isolation
# Build from Github currently not working due to it doesnt detect CUDA device
#RUN git clone --branch dev_tp https://github.com/turboderp/exllamav2
#RUN cd exllamav2 && pip install -r requirements.txt && pip install .
RUN pip install -r requirements.txt
RUN pip install https://github.com/turboderp/exllamav2/releases/download/v0.1.8/exllamav2-0.1.8+cu121.torch2.3.1-cp311-cp311-linux_x86_64.whl

# Clean up
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["python"]