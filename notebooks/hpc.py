from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import pickle
import os
from dask.distributed import LocalCluster

TIMEOUT = 12*60*60

def create_cluster(cluster='cpu', scale=150, factor=2):
     
    if cluster=='local':
        cluster = LocalCluster(
            n_workers=1,
            threads_per_worker=1,
        ) 
    elif cluster=='cpu':
        cluster = SLURMCluster(
            cores=9*factor,
            #cores=7*factor,
            #cores=6*factor,
            processes=factor,
            memory="32GB",
            account='llonpp',
            queue='batch_sapphirerapids',
            #queue='batch',
            walltime='04:00:00',
            scheduler_options=dict(
                dashboard_address=':8788'
            ),
            job_extra_directives=[
                '-M wice',
                #'-M genius',
                '-o logs/slurm-%A.log',
                '--export=ALL',
            ],
            local_directory=os.path.join(os.environ['VSC_SCRATCH'],'.cache'),
            death_timeout=TIMEOUT,
        )
        cluster.scale(scale*factor)
    else:
        raise ValueError
        
    return cluster


def create_client(cluster):
    return Client(cluster)