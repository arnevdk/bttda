#!/usr/bin/bash
podman run --security-opt=label=disable \
           --device=nvidia.com/gpu=all \
           -v ./src:/project/src:Z \
           -v ./notebooks:/project/notebooks:Z \
           -v $HOME/mne_data:/root/mne_data:Z \
           -v $HOME/.jupyter:/root/.jupyter:Z \
           -v $HOME/.ipython:/root/.ipython:Z \
	   -e CUPY_ACCELERATORS=cutensor,cub \
	   -e TENSORLY_BACKEND=cupy \
           -p 8888:8888 \
           -it hoda $1
