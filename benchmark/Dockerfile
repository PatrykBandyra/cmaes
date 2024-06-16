FROM ubuntu:20.04

WORKDIR /benchmark
RUN apt-get update && apt-get install -y gcc python3 python3-pip git
RUN apt-get install -y build-essential python-dev python-setuptools
RUN pip install numpy matplotlib scipy six
RUN pip install cocopp
RUN git clone https://github.com/numbbo/coco.git

WORKDIR /benchmark/coco
RUN python3 do.py run-python

WORKDIR /app
RUN cp /benchmark/coco/code-experiments/build/python/example_experiment_for_beginners.py /app
RUN python3 example_experiment_for_beginners.py
RUN python3 -m cocopp -o output /app/exdata/scipy-optimize-fmin
