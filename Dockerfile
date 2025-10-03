# Use an official NVIDIA CUDA devel image
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

# Set a non-interactive mode for APT
ENV DEBIAN_FRONTEND=noninteractive

# Install Python3, pip and git
RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    # apt-get install nvidia-cuda-toolkit && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip/setuptools first
RUN pip3 install --no-cache-dir --upgrade pip setuptools

# Install PyTorch (GPU version), plus other Python packages if needed
# Note: Ensure the CU version matches the base image's CUDA version
RUN pip3 install torch torchvision torchaudio

# Copy our Python script into the container
# COPY multiflow.py /app/multiflow.py

# Set the working directory
# WORKDIR /app

# Run the Python script
# CMD ["python3", "multiflow.py"]

RUN pip install jupyter

RUN pip install --no-cache-dir --upgrade pip setuptools

RUN pip install --no-cache-dir tqdm ipython Pillow numba matplotlib scipy pandas h5py numbalsoda

EXPOSE 8888

RUN apt-get update -q\
    && apt-get install -yqq pcmanfm \
        xterm \
        python3-tk \

# ENV DISPLAY=host.docker.internal:0.0