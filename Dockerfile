# 1. Khai báo hệ điều hành nền
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# 1) Setup working directory
WORKDIR /code

# 2) Install Python + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    gcc g++ git build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3) Install Python dependencies
COPY requirements.txt /code/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /code/requirements.txt

# 4) Copy all source code
COPY . /code

# 5) Entry point
RUN chmod +x /code/inference.sh
CMD ["bash", "/code/inference.sh"]


