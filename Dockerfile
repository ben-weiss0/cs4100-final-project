# Use the latest Ubuntu image
FROM ubuntu:20.04

# Metadata
LABEL authors="marcogracie"

# Install dependencies for Python, Conda, and OpenGL
# Install dependencies for Python, Conda, and OpenGL
RUN apt-get update && \
    apt-get install -y \
        wget \
        curl \
        git \
        build-essential \
        libgl1-mesa-glx \
        libglu1-mesa \
        libxi6 \
        libxmu6 \
        libx11-dev \
        libxcursor1 \
        libxi6 \
        libegl1-mesa \
        x11-apps \
        libgles2-mesa && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Add Conda to the system PATH
ENV PATH="/opt/conda/bin:${PATH}"

# Create a new Conda environment with Python and required libraries
RUN conda create -n myenv python=3.10 && \
    conda install -n myenv -c conda-forge \
        pyglet \
        shapely \
        numpy \
        gym \
        box2d-py \
        pygame \
        gymnasium && \
    conda clean -a -y

# Set the Conda environment as the default
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Copy Python files to the container
#COPY /gym_multi_car_racing/__init__.py /app/__init__.py
#COPY /gym_multi_car_racing/multi_car_racing.py /app/multi_car_racing.py

COPY test.py /app/test.py
# Set the working directory
WORKDIR /app

# Define the entrypoint to run the Python file

ENTRYPOINT ["conda", "run", "-n", "myenv", "python", "/app/test.py"]