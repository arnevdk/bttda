# hoda-bci

Implementation of regularized Toeplitz Higher Order Discriminant Analysis for Brain-Computer Interface classification

# Installation and running the notebooks

## CPU

1. Install the package using pip:
```
pip install git+ssh://git@gitlab.kuleuven.be:compneuro/hoda-bci.git[notebooks]
```
2. Run the notebooks:
```
jupyter notebook
```

## GPU

1. Install [CUDA](https://developer.nvidia.com/cuda-downloads)
2. Install [cupy](https://docs.cupy.dev/en/stable/install.html)
3. Install the package using pip:
```
pip install git+ssh://git@gitlab.kuleuven.be:compneuro/hoda-bci.git[notebooks]
```
4. Run the notebooks:
```
jupyter notebook
```



## GPU in a container

1. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
2. Generate a CDI for your GPUs
2. Build or pull the image defined in `Containerfile` as `hoda`
3. Run the container using a command like the one defined in `podman_run.sh`

# Usage

## TODO:

* [ ] Implement pruning based on mode update
* [ ] Script find optimal number of blocks
* [ ] Script find optimal component selection
* [ ] Implement tenalg as tl backend
* [ ] Move figure scripts
* [ ] Figure out differences between solvers and backends
