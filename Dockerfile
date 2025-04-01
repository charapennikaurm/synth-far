FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt update
RUN apt install -y tmux nano wget
RUN pip install jupyter
RUN pip install notebook==6.4.4
