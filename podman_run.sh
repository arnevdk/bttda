#!/usr/bin/bash
podman run --security-opt=label=disable \
           --device=nvidia.com/gpu=all \
           -v ./src:/project/src:Z \
           -v ./tests:/project/tests:Z \
           -v ./notebooks:/project/notebooks:Z \
           -v $HOME/mne_data:/root/mne_data:Z \
           -v $HOME/.jupyter:/root/.jupyter:Z \
           -v $HOME/.ipython:/root/.ipython:Z \
	   -p 8888:8888 \
           -it hoda-$1 $2
