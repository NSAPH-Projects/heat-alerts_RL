FROM nvidia/cuda:12.1.0-base-ubuntu22.04 

# we need root to install libraries
USER root

# this needs to avoid time zone question
ENV DEBIAN_FRONTEND=noninteractive

# install build essentials
RUN apt update && apt install -y build-essential \
    git \
    wget \
    software-properties-common \
    cmake \
    git \
    wget \
    unzip \
    python3-dev \
    zlib1g \
    zlib1g-dev \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    libglu1-mesa-dev \
    libasio-dev \
    pkg-config \
    python3-tk \
    libsm6 \
    libxext6 \
    libxrender1 \
    libpcre3-dev

# install bash and use as default shell
RUN apt-get install -y bash
SHELL ["/bin/bash", "-c"]

# install mamba forge
RUN arch="$(dpkg --print-architecture)" && \
    if [ "$arch" = "arm64" ]; then \
        wget -q https://github.com/conda-forge/miniforge/releases/download/23.1.0-1/Mambaforge-23.1.0-1-Linux-aarch64.sh -O ~/micromamba.sh; \
    elif [ "$arch" = "amd64" ]; then \
        wget -q https://github.com/conda-forge/miniforge/releases/download/23.1.0-1/Mambaforge-23.1.0-1-Linux-x86_64.sh -O ~/micromamba.sh; \
    else \
        echo "Unsupported architecture"; \
        exit 1; \
    fi
RUN bash ~/micromamba.sh -b -u -p ~/mambaforge
ENV PATH=/root/mambaforge/bin:$PATH

# # install python 3.10, r, pip
RUN mamba install -y -n base -c conda-forge python=3.10 r-base=4.3.0 pip

# # Make RUN commands use the new environment:
SHELL ["mamba", "run", "-n", "base", "/bin/bash", "-c"]

# # Copy env.yaml and update conda environment
COPY env-linux.yaml /tmp/env.yaml
RUN mamba env update -n base -f /tmp/env.yaml

# install d3rlpy without deps
# RUN pip install git+https://github.com/takuseno/d3rlpy@v1.1.1 --no-deps

EXPOSE 6006

# Configure SSHD.
# RUN apt install -y openssh-server
# RUN  echo 'root:password' | chpasswd
# RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config
# RUN service ssh start
# EXPOSE 22
# CMD ["/usr/sbin/sshd","-D"]
