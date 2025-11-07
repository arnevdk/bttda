FROM cupy/cupy:v13.3.0 AS cupy
WORKDIR /project
COPY setup.py ./
COPY src ./src
RUN pip3 install --no-cache-dir -e .[experiments]
RUN python -m cupyx.tools.install_library --cuda 12.x --library cutensor
COPY notebooks ./notebooks
ENV TENSORLY_BACKEND=cupy \
    CUPY_ACCELERATORS=cutensor,cub

FROM python:3.11 AS numpy
WORKDIR /project
COPY setup.py ./
COPY src ./src
RUN pip3 install --no-cache-dir -e .[experiments]
COPY notebooks ./notebooks
ENV TENSORLY_BACKEND=numpy
