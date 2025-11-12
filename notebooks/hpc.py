import math
import os

from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster

TIMEOUT = 12 * 60 * 60


def create_cluster(cluster=None, scale=150):

    if cluster == "local":
        cluster = LocalCluster()
    elif cluster == "wice":
        n_cores = 36 * 2
        n_processes = int(math.sqrt(n_cores))
        cluster = SLURMCluster(
            cores=n_cores,
            processes=n_processes,
            memory="250GB",
            account=os.environ["HPC_GROUP"],
            queue="batch",
            walltime="01:00:00",
            scheduler_options=dict(dashboard_address=":62698"),
            job_extra_directives=[
                "-M wice",
                "-o logs/slurm-%A.log",
                "--export=ALL",
                "--nodes=1",
            ],
        )
        cluster.scale(150 * n_processes)
    elif cluster == "wice_sapphirerapids":
        n_cores = 48 * 2
        n_processes = int(math.sqrt(n_cores))
        cluster = SLURMCluster(
            cores=n_cores,
            processes=n_processes,
            memory="250GB",
            account=os.environ["HPC_GROUP"],
            queue="batch_sapphirerapids",
            walltime="01:00:00",
            scheduler_options=dict(dashboard_address=":62698"),
            job_extra_directives=[
                "-M wice",
                "-o logs/slurm-%A.log",
                "--export=ALL",
                "--nodes=1",
            ],
        )
        cluster.scale(150 * n_processes)
    else:
        raise ValueError(
            "cluster must be one of: 'local', 'wice', 'wice_sapphirerapids'"
        )

    return cluster


def create_client(cluster):
    return Client(cluster)
