FROM ubuntu:20.04

# Setup of enviornment for coco framework
WORKDIR /benchmark
RUN apt-get update && apt-get install -y gcc python3 python3-pip git
RUN apt-get install -y build-essential python-dev python-setuptools
RUN pip install numpy matplotlib scipy six
RUN pip install cocopp
RUN git clone https://github.com/numbbo/coco.git

WORKDIR /benchmark/coco
RUN python3 do.py run-python

# Running benchmark and processing data
WORKDIR /app
COPY . .
#RUN python3 coco_benchmark_pypop.py
#RUN python3 -m cocopp -o output /app/exdata/coco-maes