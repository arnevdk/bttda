FROM cupy/cupy:latest AS cupy
WORKDIR /project
COPY setup.py ./
COPY src ./src
RUN python3 -m cupyx.tools.install_library --cuda 12.x --library cutensor
RUN pip3 install --no-cache-dir -e .[experiments]
COPY notebooks ./notebooks
COPY .mne-python.json /root/.mne/mne-python.json
RUN mkdir -p /root/mne_data/MOABB_results
ENV TENSORLY_BACKEND=cupy \
    CUPY_ACCELERATORS=cutensor,cub

FROM python:3.11 AS numpy
WORKDIR /project
COPY setup.py ./
COPY src ./src
RUN pip3 install --no-cache-dir -e .[experiments]
COPY notebooks ./notebooks
COPY .mne-python.json /root/.mne/mne-python.json
RUN mkdir -p /root/mne_data/MOABB_results
ENV TENSORLY_BACKEND=numpy
