FROM rapidsai/rapidsai-core:22.04-cuda11.0-base-ubuntu20.04-py3.9

RUN apt update -q
RUN apt install -qy make gcc g++
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install wandb jupyterlab optuna
RUN pip install polars lightgbm annoy gensim

WORKDIR /root/work/
