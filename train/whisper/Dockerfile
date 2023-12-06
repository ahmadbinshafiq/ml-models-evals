FROM python:3.10

RUN pip install datasets>=2.6.1
RUN pip install -U accelerate==0.20.3
RUN pip install transformers==4.30.1
# RUN pip install git+https://github.com/huggingface/transformers
RUN pip install librosa
RUN pip install evaluate>=0.30
RUN pip install jiwer
RUN pip install gradio tensorboard

WORKDIR /app