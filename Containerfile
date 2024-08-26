FROM cupy/cupy:v13.0.0b1 AS cupy
WORKDIR /project
COPY setup.py ./
COPY src ./src
RUN pip3 install --no-cache-dir -e .[experiments] \
	&& pip3 install --no-cache-dir cutensor-cu11
COPY notebooks ./notebooks
ENV TENSORLY_BACKEND=cupy \
    CUPY_ACCELERATORS=cutensor,cub \
    CUPY_CUDA_PER_THREAD_DEFAULT_STREAM=1

FROM pytorch/pytorch AS pytorch
WORKDIR /project
COPY setup.py ./
COPY src ./src
RUN pip3 install --no-cache-dir -e .[experiments] \
	&& pip3 install --no-cache-dir cutensor-cu11
COPY notebooks ./notebooks
ENV TENSORLY_BACKEND=pytorch

FROM python:3.12 AS numpy
WORKDIR /project
COPY setup.py ./
COPY src ./src
RUN pip3 install --no-cache-dir -e .[experiments]
COPY notebooks ./notebooks
ENV TENSORLY_BACKEND=numpy
