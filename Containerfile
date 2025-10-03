FROM cupy/cupy:v13.3.0 AS cupy
WORKDIR /project
COPY setup.py ./
COPY src ./src
RUN pip3 install --no-cache-dir -e .[experiments] \
	&& pip3 install --no-cache-dir cutensor-cu12
COPY notebooks ./notebooks
ENV TENSORLY_BACKEND=cupy \
    CUPY_ACCELERATORS=cutensor,cub \
    CUPY_CUDA_PER_THREAD_DEFAULT_STREAM=1

FROM python:3.11 AS numpy
WORKDIR /project
COPY setup.py ./
COPY src ./src
RUN pip3 install --no-cache-dir -e .[experiments]
COPY notebooks ./notebooks
ENV TENSORLY_BACKEND=numpy
