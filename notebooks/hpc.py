from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import pickle
import os
from dask.distributed import LocalCluster
import math

TIMEOUT = 12*60*60

def create_cluster(cluster='cpu', scale=150, factor=1):
     
    if cluster=='local':
        cluster = LocalCluster(
            n_workers=1,
            threads_per_worker=1,
            dashboard_address=':62698'
        ) 
    elif cluster == 'gpu':
        cluster = SLURMCluster(
            cores=1,
            processes=1,
            memory='32GB',
            account='llonpp',
            queue='gpu_p100',
            #queue='gpu_p100_debug',
            walltime='01:00:00',
            scheduler_options=dict(
                dashboard_address=':62698'
            ),
            job_extra_directives=[
                '-M genius',
                '--gpus-per-node=1',
                '-o logs/slurm-%A.log',
                '--export=ALL',
                '--nodes=1',
            ],
        )
        cluster.scale(32)
        #cluster.scale(1)
    elif cluster=='batch_sapphirerapids':
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
                dashboard_address=':62698'
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
    elif cluster=='batch':
        cluster = SLURMCluster(
            cores=4,
            processes=1,
            memory="64GB",
            account='llonpp',
            queue='batch',
            walltime='08:00:00',
            scheduler_options=dict(
                dashboard_address=':62698'
            ),
            job_extra_directives=[
                #'-M wice',
                '-M genius',
                '-o logs/slurm-%A.log',
                '--export=ALL',
            ],
            local_directory=os.path.join(os.environ['VSC_SCRATCH'],'.cache'),
            death_timeout=TIMEOUT,
        )
        cluster.scale(scale)
    elif cluster=='batch_sapphirerapids_full_nodes':
        cluster = SLURMCluster(
            cores= 96,
            memory="250GB",
            account='llonpp',
            queue='batch_sapphirerapids',
            walltime='08:00:00',
            scheduler_options=dict(
                dashboard_address=':62698'
            ),
            job_extra_directives=[
                '--nodes=1',
                '-M wice',
                '-o logs/slurm-%A.log',
                '--export=ALL',
            ],
            local_directory=os.path.join(os.environ['VSC_SCRATCH'],'.cache'),
            death_timeout=TIMEOUT,
        )
        cluster.scale(scale*10)
    elif cluster=='batch_full_nodes':
        cluster = SLURMCluster(
            cores=72,
            memory="250GB",
            account='llonpp',
            queue='batch',
            #queue='batch',
            walltime='08:00:00',
            scheduler_options=dict(
                dashboard_address=':62698'
            ),
            job_extra_directives=[
                '-M wice',
                '-o logs/slurm-%A.log',
                '--export=ALL',
                '--nodes=1'
            ],
            local_directory=os.path.join(os.environ['VSC_SCRATCH'],'.cache'),
            death_timeout=TIMEOUT,
        )
        cluster.scale(scale*int(math.sqrt(72)))
    else:
        raise ValueError
        
    return cluster


def create_client(cluster):
    return Client(cluster)