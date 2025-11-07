from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import pickle
import os
from dask.distributed import LocalCluster
import math

TIMEOUT = 12*60*60

def create_cluster(cluster=None, scale=150):
     
    if cluster=='local':
        cluster = LocalCluster(
            n_workers=1,
            threads_per_worker=1,
            dashboard_address=':62698'
        ) 
    elif cluster == 'wice':
        n_cores = 36*2
        n_processes = int(math.sqrt(n_cores))
        cluster = SLURMCluster(
            cores=n_cores,
            processes=n_processes,
            memory='250GB',
            account='llonpp',
            queue='batch',
            walltime='01:00:00',
            scheduler_options=dict(
                dashboard_address=':62698'
            ),
            job_extra_directives=[
                '-M wice',
                '-o logs/slurm-%A.log',
                '--export=ALL',
                '--nodes=1',
            ],
        )
        cluster.scale(150*n_processes) 
    elif cluster == 'wice_sapphirerapids':
        n_cores = 48*2
        n_processes = int(math.sqrt(n_cores))
        cluster = SLURMCluster(
            cores=n_cores,
            processes=n_processes,
            memory='250GB',
            account='llonpp',
            queue='batch_sapphirerapids',
            walltime='01:00:00',
            scheduler_options=dict(
                dashboard_address=':62698'
            ),
            job_extra_directives=[
                '-M wice',
                '-o logs/slurm-%A.log',
                '--export=ALL',
                '--nodes=1',
            ],
        )
        cluster.scale(150*n_processes)
    else:
        raise ValueError("cluster must be one of: 'local', 'wice', 'wice_sapphirerapids'")
        
    return cluster


def create_client(cluster):
    return Client(cluster)
