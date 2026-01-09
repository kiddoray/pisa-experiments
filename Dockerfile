# PISA Experiments Docker Image
# Base: NVIDIA CUDA + Ubuntu for GPU support with Blender/PyBullet

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Set up basic environment
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1

# ============================================================================
# EGL/OpenGL Environment for headless GPU rendering (Eevee)
# ============================================================================
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display
# Use NVIDIA EGL for headless rendering
ENV __GLX_VENDOR_LIBRARY_NAME=nvidia
ENV __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
# Blender specific
ENV BLENDER_SYSTEM_PYTHON=/usr/bin/python3
# XDG runtime directory for Wayland/display fallback
ENV XDG_RUNTIME_DIR=/tmp/runtime-root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Basic tools
    wget \
    curl \
    git \
    vim \
    unzip \
    ca-certificates \
    # Python
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    # Blender/OpenGL dependencies
    libxi6 \
    libxxf86vm1 \
    libxfixes3 \
    libxrender1 \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libegl1 \
    libgles2 \
    libxkbcommon0 \
    libsm6 \
    libice6 \
    # NVIDIA EGL/OpenGL support
    libnvidia-gl-535 \
    libnvidia-egl-wayland1 \
    libglvnd0 \
    libglvnd-dev \
    libegl1-mesa \
    libegl-mesa0 \
    # OpenEXR dependencies
    libopenexr-dev \
    openexr \
    # Image/Video processing
    libpng-dev \
    libjpeg-dev \
    ffmpeg \
    # Build tools (for compiling some pip packages)
    build-essential \
    cmake \
    ninja-build \
    # X11/Xvfb for fallback rendering
    xvfb \
    x11-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create XDG runtime directory
RUN mkdir -p /tmp/runtime-root && chmod 700 /tmp/runtime-root

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace/pisa-experiments

# Copy requirements first (for better layer caching)
COPY sim_data/requirements.txt /tmp/sim_requirements.txt
COPY requirements/ /tmp/requirements/

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/sim_requirements.txt

# Install bpy (Blender Python API)
RUN pip install --no-cache-dir bpy==3.6.0 --extra-index-url https://download.blender.org/pypi/

# Install additional requirements if needed
RUN pip install --no-cache-dir -r /tmp/requirements/requirements.txt || true

# Copy the entire project
COPY . /workspace/pisa-experiments/

# Install the project in editable mode (if setup.py exists)
RUN pip install -e . || true

# Create directories for data and output
RUN mkdir -p /workspace/pisa-experiments/sim_output \
    && mkdir -p /workspace/pisa-experiments/data

# Set default command
CMD ["/bin/bash"]
