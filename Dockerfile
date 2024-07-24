#FROM nvcr.io/nvidia/pytorch:23.07-py3
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
LABEL authors="remichu"
WORKDIR /app
ENV CMAKE_ARGS="-DLLAMA_CUDA=on"
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ca-certificates \
    sudo \
    python3 \
    python3-pip \
    ninja-build \
    git \
    wget
#ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
RUN pip install llama-cpp-python
ADD requirements.txt /app
RUN pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip install ninja
RUN pip install packaging
RUN pip install flash-attn==2.5.9.post1 --no-build-isolation
#RUN git clone https://github.com/turboderp/exllamav2
#RUN cd exllamav2 && pip install -r requirements.txt && pip install .
RUN pip install https://github.com/turboderp/exllamav2/releases/download/v0.1.6/exllamav2-0.1.6+cu121.torch2.3.1-cp310-cp310-linux_x86_64.whl
RUN pip install -r requirements.txt
ENTRYPOINT ["python3"]