"""
Main script for running the experiments with fully connected BNNs.

The focus is to investigate how the sampler behaves within the warmup phase.
Therefore do not forget to add the logging of relevant metrics in the numpyro
code like the step size, acceptance prob etc. The file of interest is utils.py in the
 numpyro library (line 371) as of version 0.13.0.

Run not from root directory but from experiments/fcn_bnns/:
python fcn_bnn_sampling_insights.py
"""
import copy
import itertools
import json
import os
import pickle
import sys
import time
from datetime import datetime

import jax
import yaml
from jax import numpy as jnp
from joblib import (
    Parallel,
    delayed,
    parallel_config,
)
from numpyro.infer import HMC, NUTS

sys.path.append('../..')
import probabilisticml as pml
from src.shallow_bnn_numpyro import gaussian_mlp_from_config  # noqa: E402
from src.utils import add_chain_dimension  # noqa: E402


def load_config():
    """Load the configuration file."""
    config_path = 'config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def save_config(config, path):
    """Save the configuration file."""
    with open(path, 'w') as file:
        yaml.dump(config, file)


def create_exp_saving_dir():
    """Create the directory for saving the experiment results."""
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists('results/fcn_bnns'):
        os.mkdir('results/fcn_bnns')
    if not os.path.exists('results/fcn_bnns/config'):
        os.mkdir('results/fcn_bnns/config')


def experiment_generator(config: dict) -> dict:
    """Generate the experiments (Cartesian Product)."""
    exp_dimensions = []
    for key, value in config.items():
        if isinstance(value, list):
            exp_dimensions.append(value)
        elif isinstance(value, dict):
            exp_dimensions.append(list(value.keys()))
        else:
            exp_dimensions.append([value])

    exp_tuples = list(itertools.product(*exp_dimensions))
    experiments = {
        f'exp{str(i)}|' + '|'.join([str(e) for e in p]): p
        for i, p in enumerate(exp_tuples)
    }
    return experiments


def load_data(dataset: str, seed: int) -> tuple:
    """
    Load the training data.

    Args:
        dataset (str): Name of the dataset to load.
        seed (int): Seed for the random number generator.

    Returns:
        X_train (jax.numpy.ndarray): Input features for training.
        Y_train (jax.numpy.ndarray): Target values for training.
    """
    regr_dataset = pml.data.dataset.DatasetTabular(
        data_path=f'../../data/{dataset}',
        target_indices=[],
        split_spec={'train': 0.8, 'test': 0.2},
        seed=seed,
        standardize=True,
    )
    X_train, Y_train = regr_dataset.get_data(split='train', data_type='jax')

    return X_train, Y_train


def exp_tuple_to_dict(exp: tuple) -> dict:
    """Convert an experiment tuple to a dictionary."""
    return {
        'data': exp[0],
        'activation': exp[1],
        'hidden_structure': [int(d) for d in exp[2].split('-')],
        'n_chains': exp[3],
        'n_samples': exp[4],
        'keep_warmup': exp[5],
        'sampler': exp[6],
        'replications': exp[7],
        'prior_sd': exp[8],
        'prior_dist': exp[9],
    }


def run_experiment(config: dict, exp_name: str, exp: tuple, date: str) -> None:
    """Run a single experiment."""
    start_time = time.time()
    # load the data
    X_train, Y_train = load_data(exp[0], seed=exp[7])
    exp_dict = exp_tuple_to_dict(exp)

    # initialize the model & sampler
    model_config = {
        'dim_input': X_train.shape[1],
        'dim_output': 1,
        'dim_hidden': exp_dict['hidden_structure'],
        'activation': exp_dict['activation'],
        'w_prior': {
            'name': exp_dict['prior_dist'],
            'loc': 0,
            'scale': exp_dict['prior_sd'],
        },
        'b_prior': {
            'name': exp_dict['prior_dist'],
            'loc': 0,
            'scale': exp_dict['prior_sd'],
        },
        'precision_prior': {
            'name': 'HalfNormal',
            'scale': 1,
        },
    }
    model = gaussian_mlp_from_config(model_config)

    mcmc_args = pml.utils.configclasses.ConfigNumPyroSampler(
        num_warmup=config['sampler'][exp_dict['sampler']]['n_warmup'],
        num_samples=config['n_samples'],
        num_chains=1,
    )
    if 'NUTS' in exp_dict['sampler']:
        sampler = NUTS(
            model,
            step_size=config['sampler'][exp_dict['sampler']]['step_size'],
            adapt_step_size=config['sampler'][exp_dict['sampler']]['adapt_step_size'],
        )
    elif 'HMC' in exp_dict['sampler']:
        sampler = HMC(
            model,
            step_size=config['sampler'][exp_dict['sampler']]['step_size'],
            adapt_step_size=config['sampler'][exp_dict['sampler']]['adapt_step_size'],
            trajectory_length=config['sampler'][exp_dict['sampler']][
                'trajectory_length'
            ],
            adapt_mass_matrix=config['sampler'][exp_dict['sampler']][
                'adapt_mass_matrix'
            ],
        )
    else:
        raise ValueError('Sampling Kernel not implemented')

    mcmc_learner = pml.inference.infer_numpyro.sampling.NumPyroMCMCLearner(
        model=model,
        kernel=sampler,
        mcmc_args=copy.deepcopy(mcmc_args),
        verbose=False,
    )
    num_parallel_chains = exp_dict['n_chains']
    rng_key = jax.random.PRNGKey(exp_dict['replications'])
    rng_key_m = jax.random.split(rng_key, num_parallel_chains)

    # define a helper function to get the posterior samples
    def get_mcmc_samples(mcmc_learner, rng_key):
        m = copy.deepcopy(mcmc_learner)
        m.perform_inference(
            rng_key=rng_key,
            X=X_train,
            Y=Y_train,
            collect_warmup=exp_dict['keep_warmup'],
        )
        # print(m.mcmc._states)
        return m.get_samples(with_warmup=exp_dict['keep_warmup'])

    # sequential execution
    samples = []
    for j in range(num_parallel_chains):
        samples.append(
            get_mcmc_samples(
                mcmc_learner=mcmc_learner,
                rng_key=rng_key_m[j],
            )
        )

    # concatenate the posterior samples in the first dimension for each dictionary entry
    posterior_samples = {
        key: jnp.concatenate([samples[j][key] for j in range(num_parallel_chains)])
        for key in samples[0].keys()
    }
    # add chain dimension
    posterior_samples = add_chain_dimension(
        posterior_samples, n_chains=num_parallel_chains
    )

    # save the posterior samples
    with open(f'results/fcn_bnns/{date}/{exp_name}.pkl', 'wb') as fpkl:
        pickle.dump(posterior_samples, fpkl)

    # save the model_config
    with open(f'results/fcn_bnns/{date}/mconfig_{exp_name}.json', 'w') as fjson:
        json.dump(model_config, fjson)

    # log the runtime (in minutes) one can also log the average acceptance rate here
    with open(f'results/fcn_bnns/{date}/runtime.txt', 'a+') as frun:
        frun.write(f'{exp_name}: {(time.time() - start_time) / 60:.2f}\n')

    print(f'{(time.time() - start_time) / 60:.2f} min for {exp_name}')


def main():
    """Run all the experiments."""
    # Setup ------------------------------------------------------
    config = load_config()
    experiments = experiment_generator(config)
    n_chains = config['n_chains']
    n_cpus = os.cpu_count()
    print(f'Number of CPUs: {n_cpus}')
    max_parallel_experiments = (n_cpus - 1) // (n_chains + 1)
    print(
        f'Running {max_parallel_experiments} of {len(experiments)} '
        'experiment(s) in parallel.'
    )
    print(f'Each experiment will run {n_chains} parallel chains.')
    date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    os.makedirs(f'../../results/fcn_bnns/{date}', exist_ok=True)
    save_config(config, path=f'results/fcn_bnns/{date}/config.yaml')

    # Run the experiments in parallel -------------------------------------------
    parallel_experiments = True
    if parallel_experiments:
        with parallel_config(backend='loky', inner_max_num_threads=n_chains):
            Parallel(n_jobs=max_parallel_experiments, prefer='threads')(
                delayed(run_experiment)(config, exp_name, exp, date)
                for exp_name, exp in experiments.items()
            )
    else:
        for exp_name, exp in experiments.items():
            run_experiment(config, exp_name, exp, date)

    print('All experiments have been run.')


if __name__ == '__main__':
    main()
