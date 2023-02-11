FROM rapidsai/rapidsai-core:22.04-cuda11.0-base-ubuntu20.04-py3.9

RUN apt-get update -q
RUN apt-get install -qy make gcc g++
COPY requirements.txt .
RUN pip install -r requirements.txt
WORKDIR /root/work/
