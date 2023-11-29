FROM cupy/cupy

WORKDIR /project
COPY setup.py ./
COPY src ./src
RUN pip3 install --no-cache-dir -e .[notebook]
COPY notebooks ./notebooks
CMD jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root
