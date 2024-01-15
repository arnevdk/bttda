#FROM cupy/cupy:latest
FROM cupy/cupy:v13.0.0b1

WORKDIR /project
COPY setup.py ./
COPY src ./src
RUN pip3 install --no-cache-dir -e .[experiments] \
	&& pip3 install --no-cache-dir cutensor-cu11
COPY notebooks ./notebooks
