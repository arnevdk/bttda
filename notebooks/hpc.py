from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import pickle
import os
from dask.distributed import LocalCluster

TIMEOUT = 12*60*60

def create_cluster(cluster='cpu', scale=256*2*2):
     
    if cluster=='local':
        cluster = LocalCluster(
            n_workers=1,
            threads_per_worker=1,
        ) 
    elif cluster=='cpu':
        cluster = SLURMCluster(
            cores=96,
            memory="250GB",
            account='llonpp',
            queue='batch_sapphirerapids',
            walltime='08:00:00',
            scheduler_options=dict(
                dashboard_address=':8787'
            ),
            job_extra_directives=[
                '-M wice',
                '-o logs/slurm-%A.log',
                '--nodes=1',
                '--export=ALL',
            ],
            local_directory=os.path.join(os.environ['VSC_SCRATCH'],'.cache'),
            death_timeout=TIMEOUT,
        )
        cluster.scale(scale)
    else:
        raise ValueError
        
    return cluster


def create_client(cluster):
    return Client(cluster)