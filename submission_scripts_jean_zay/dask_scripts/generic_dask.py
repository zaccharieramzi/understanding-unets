from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from sklearn.model_selection import ParameterGrid


def train_on_jz_dask(job_name, train_function, *args, **kwargs):
    cluster = SLURMCluster(
        cores=1,
        job_cpu=40,
        memory='40GB',
        job_name=job_name,
        walltime='20:00:00',
        interface='ib0',
        job_extra=[
            f'--gres=gpu:4',
            '--qos=qos_gpu-t3',
            '--distribution=block:block',
            '--hint=nomultithread',
            '--output=%x_%j.out',
        ],
        env_extra=[
            'cd $WORK/understanding-unets',
            '. ./submission_scripts_jean_zay/env_config.sh',
        ],
    )
    cluster.scale(1)

    print(cluster.job_script())

    client = Client(cluster)
    futures = client.submit(
        # function to execute
        train_function,
        *args,
        **kwargs,
        # this function has potential side effects
        pure=True,
    )
    run_id = client.gather(futures)
    print(f'Train run id: {run_id}')

def eval_on_jz_dask(job_name, eval_function, *args, **kwargs):
    cluster = SLURMCluster(
        cores=1,
        job_cpu=40,
        memory='40GB',
        job_name=job_name,
        walltime='20:00:00',
        interface='ib0',
        job_extra=[
            f'--gres=gpu:4',
            '--qos=qos_gpu-t3',
            '--distribution=block:block',
            '--hint=nomultithread',
            '--output=%x_%j.out',
        ],
        env_extra=[
            'cd $WORK/understanding-unets',
            '. ./submission_scripts_jean_zay/env_config.sh',
        ],
    )
    cluster.scale(1)

    print(cluster.job_script())

    client = Client(cluster)
    futures = client.submit(
        # function to execute
        eval_function,
        *args,
        **kwargs,
        # this function has potential side effects
        pure=True,
    )
    metrics_names, eval_res = client.gather(futures)
    print(metrics_names)
    print(eval_res)
    print('Shutting down dask workers')

def train_eval_parameter_grid(job_name, train_function, eval_function, parameter_grid, n_samples_eval=68):
    parameters = list(ParameterGrid(parameter_grid))
    n_parameters_config = len(parameters)
    train_cluster = SLURMCluster(
        cores=1,
        job_cpu=40,
        memory='40GB',
        job_name=job_name,
        walltime='20:00:00',
        interface='ib0',
        job_extra=[
            f'--gres=gpu:4',
            '--qos=qos_gpu-t3',
            '--distribution=block:block',
            '--hint=nomultithread',
            '--output=%x_%j.out',
        ],
        env_extra=[
            'cd $WORK/understanding-unets',
            '. ./submission_scripts_jean_zay/env_config.sh',
        ],
    )
    train_cluster.scale(n_parameters_config)
    client = Client(train_cluster)
    futures = [client.submit(
        # function to execute
        train_function,
        **params,
    ) for params in parameters]
    run_ids = client.gather(futures)
    client.close()
    train_cluster.close()
    # eval
    eval_cluster = SLURMCluster(
        cores=1,
        job_cpu=40,
        memory='80GB',
        job_name=job_name,
        walltime='20:00:00',
        interface='ib0',
        job_extra=[
            f'--gres=gpu:4',
            '--qos=qos_gpu-t3',
            '--distribution=block:block',
            '--hint=nomultithread',
            '--output=%x_%j.out',
        ],
        env_extra=[
            'cd $WORK/fastmri-reproducible-benchmark',
            '. ./submission_scripts_jean_zay/env_config.sh',
        ],
    )
    eval_cluster.scale(n_parameters_config)
    client = Client(eval_cluster)
    for params in parameters:
        params.pop('n_samples')
    futures = [client.submit(
        # function to execute
        eval_function,
        run_id=run_id,
        n_samples=n_samples_eval,
        **params,
    ) for run_id, params in zip(run_ids, parameters)]

    for params, future in zip(parameters, futures):
        metrics_names, eval_res = client.gather(future)
        print('Parameters', params)
        print(metrics_names)
        print(eval_res)
    print('Shutting down dask workers')
    client.close()
    eval_cluster.close()
