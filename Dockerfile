#FROM nvcr.io/nvidia/pytorch:23.07-py3
#FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS base
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base
LABEL authors="remichu"

# Copy all source code to container
COPY . /app
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
    ffmpeg \
    portaudio19-dev \
    espeak-ng \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd \
    && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA for Python 3.12 and install Python 3.12
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.12 get-pip.py \
    && rm get-pip.py

# Set Python 3.12 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

#ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
# Install depencencies
RUN python3 -m pip install --upgrade pip cmake scikit-build samplerate
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

RUN python3 -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
RUN python3 -m pip install ninja
RUN python3 -m pip install packaging
# RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.3cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
RUN python3 -m pip install flash-attn --no-build-isolation
RUN #python3 -m pip install https://github.com/turboderp-org/exllamav2/releases/download/v0.2.8/exllamav2-0.2.8+cu121.torch2.4.0-cp312-cp312-linux_x86_64.whl
RUN python3 -m pip install https://github.com/turboderp-org/exllamav2/releases/download/v0.2.8/exllamav2-0.2.8+cu124.torch2.6.0-cp312-cp312-linux_x86_64.whl

FROM base AS dev

RUN python3 -m pip install -r requirements-docker.txt

# Build from Github currently not working due to it doesnt detect CUDA device
#RUN git clone --branch dev https://github.com/turboderp/exllamav2
#RUN cd exllamav2 && pip install -r requirements.txt && pip install .
#ADD whl /app/whl
#RUN pip install /app/whl/exllamav2-0.2.4-cp312-cp312-linux_x86_64.whl
#RUN pip install git+https://github.com/huggingface/transformers
RUN python3 -m pip install -U transformers
#git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ
#INSTALL_KERNELS=1 pip install git+https://github.com/casper-hansen/AutoAWQ.git
#pip install -vvv --no-build-isolation -e .

# initialize split-lang to download model
RUN python3 /app/docker_initializer.py

# install gallama from source and clean up files for smaller container size
RUN python3 -m pip install . \
    && pip cache purge \
    && rm -rf /root/.cache/pip \
    && rm -rf /app  # Remove source files after installation


# Clean up
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*


ENTRYPOINT ["gallama", "run"]