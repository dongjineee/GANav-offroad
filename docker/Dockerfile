#####################
# Base image #
#####################
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04 as base

# Labels
LABEL maintainer="dongjincho"
LABEL contact="d22g66@naver.com"
LABEL description="ga_nav"

ARG ROS_VERSION="noetic"

# Disable dialog frontend
ARG DEBIAN_FRONTEND=noninteractive

# Select shell
SHELL ["/bin/bash", "-c"]

# ==
# Install ROS
# ==
RUN apt update \
    && apt install --no-install-recommends -y curl gnupg lsb-release git build-essential nano \
    && sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
    && apt update \
    && apt install --no-install-recommends -y ros-${ROS_VERSION}-desktop-full \
    && apt install --no-install-recommends -y \
        python3-pip \
        python3-venv \
        python3-rosdep \
        python3-rosinstall \
        python3-rosinstall-generator \
        python3-wstool \
        python3-catkin-tools \
        python3-osrf-pycommon \
    && apt install python3-tk \
    && rm -f "/etc/ros/rosdep/sources.list.d/20-default.list" \
    && rosdep init \
    && rosdep update \
    && echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
    
    
# ==
# Install python packages
# ==
RUN apt-get update && apt-get install -y python3-venv \
    && cd /root \
    && python3 -m venv env --system-site-packages \
    && source /root/env/bin/activate \
    && pip install --upgrade pip \
    && pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 \
    && pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
    && pip install mmengine \
    && pip install einops prettytable \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ==
# Build_ws
# ==
RUN mkdir -p /root/catkin_ws/src \
    && source /opt/ros/noetic/setup.bash \
    && cd /root/catkin_ws

# Configure environment
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN echo "source /root/env/bin/activate" >> ~/.bashrc

# Define the dev stage for Docker Compose
FROM base as dev

# Set working directory
WORKDIR /root/catkin_ws

