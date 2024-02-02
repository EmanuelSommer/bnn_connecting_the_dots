"""Utility functions for the analysis of the experiments."""
import copy
import json
import os
import pickle
import re
import sys

import jax
import jax.numpy as jnp
import numpy as np
import probabilisticml as pml
import yaml
from probabilisticml.inspection.prediction import MCMCPredictor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

sys.path.append('../../..')
from src.bnn_numpyro import gaussian_mlp_from_config  # noqa: E402
from src.utils import (  # noqa: E402
    add_chain_dimension,
    flatten_chain_dimension,
)


def load_config(config_path: str) -> tuple[dict, list]:
    """Load the configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_exp_names(path: str = '.') -> tuple[list, list]:
    """Get the experiment names."""
    file_names = os.listdir(path)
    exp_names = [
        file_name.removesuffix('.pkl')
        for file_name in file_names
        if file_name.endswith('.pkl')
    ]
    return exp_names


def extract_exp_info(exp_name: str) -> dict:
    """Extract the experiment information from the experiment name."""
    exp_info = {}
    key_dict = {
        'data': str,
        'activation': str,
        'hidden_structure': str,
        'n_chains': int,
        'n_samples': int,
        'keep_warmup': bool,
        'sampler': str,
        'replications': int,
        'prior_sd': float,
        'prior_dist': str,
    }
    trimmed_exp_name = re.sub(r'^exp[0-9]+\|', '', exp_name)
    for i, (k, v) in enumerate(key_dict.items()):
        exp_info[k] = trimmed_exp_name.split('|')[i]
        if v == int:
            exp_info[k] = int(exp_info[k])
        elif v == float:
            exp_info[k] = float(exp_info[k])
        elif v == bool:
            exp_info[k] = exp_info[k] == 'True'

    return exp_info


def load_samples(exp_name: str, path: str = '', discard_warmup: int = 0) -> dict:
    """Load the posterior samples."""
    path = f'{path}/' if path else path
    sample_path = f'{path}{exp_name}.pkl'
    with open(sample_path, 'rb') as f:
        posterior_samples = pickle.load(f)
        if discard_warmup > 0:
            posterior_samples = {
                k: v[:, discard_warmup:, ...] for k, v in posterior_samples.items()
            }
        posterior_samples_raw = copy.deepcopy(posterior_samples)
        posterior_samples_raw = flatten_chain_dimension(posterior_samples_raw)
    return posterior_samples, posterior_samples_raw


def get_posterior_predictive(
    model,
    posterior_samples_raw: dict,
    X_val: jnp.ndarray,
    n_chains: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Get the posterior predictive samples."""
    rng_key_predict = jax.random.PRNGKey(0)
    # if X_val has more than 1000 rows, we only predict on the first 1000 rows
    predictor = MCMCPredictor(
        model=model,
        rng_key=rng_key_predict,
        samples=posterior_samples_raw,
    )
    preds = (
        predictor.predict(
            X=X_val,
            Y=None,
        )
    ).squeeze()

    preds_chain_dim = add_chain_dimension({'pp': preds}, n_chains=n_chains)['pp']

    return preds_chain_dim, preds


def load_data(
    exp_info: dict,
    data_path: str,
    splittype: str = 'train',
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Load the training data.

    Args:
        exp_info (dict): Experiment Specs dictionary.
        splittype (str): Type of data ("val" or "train") to load. Defaults to "train".

    Returns:
        X_train (jax.numpy.ndarray): Input features for Eval/Train.
        Y_train (jax.numpy.ndarray): Target values for Eval/Train.
    """
    regr_dataset = pml.data.dataset.DatasetTabular(
        data_path=f'{data_path}/{exp_info["data"]}',
        target_indices=[],
        split_spec={'train': 0.8, 'val': 0.2},
        seed=exp_info['replications'],
        standardize=True,
    )
    return regr_dataset.get_data(split=splittype, data_type='jax')


def fit_baselines(
    X_train: jnp.ndarray,
    Y_train: jnp.ndarray,
) -> tuple[LinearRegression, RandomForestRegressor]:
    """Fit the baselines."""
    # Fit the linear regression model
    linear_regr = LinearRegression()
    linear_regr.fit(X_train, Y_train.ravel())

    # Fit the random forest model
    rf_regr = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=0,
    )
    rf_regr.fit(X_train, Y_train.ravel())

    return linear_regr, rf_regr


def evaluate_baselines(
    X_val: jnp.ndarray,
    Y_val: jnp.ndarray,
    linear_regr: LinearRegression,
    rf_regr: RandomForestRegressor,
) -> tuple[float, float]:
    """Evaluate Hold out MSE of the baselines."""
    mse_linear = np.mean((linear_regr.predict(X_val).ravel() - Y_val.ravel()) ** 2)
    mse_rf = np.mean((rf_regr.predict(X_val).ravel() - Y_val.ravel()) ** 2)
    return mse_linear, mse_rf


def load_runtimes() -> dict:
    """Load the runtimes from the runtime.txt file."""
    with open('runtime.txt', 'r') as f:
        runtime = {}
        for line in f.readlines():
            line = line.strip()
            if line:
                id, value = line.split(': ')
                runtime[id] = float(value)
    return runtime


def load_model(exp_name: str, path: str = ''):
    """Initialize the model from the model config file."""
    path = f'{path}/' if path else path
    with open(f'{path}mconfig_{exp_name}.json', 'r') as f:
        model_config = json.load(f)
    return gaussian_mlp_from_config(model_config)


def fit_slope(y: jnp.array) -> jnp.array:
    """
    Fit a least squares slope to the data.

    Provided an array of values, fit a least squares slope to the data. The target
    variable is the array itself, and the predictor variable is the index of the
    array.

    Args:
        x (jnp.array): The target variable to fit the slope to.

    Returns:
        float: The slope of the least squares line.
    """
    x = jnp.arange(y.shape[0]) / y.shape[0]
    A = np.vstack([x, np.ones(len(x))]).T
    slope = jnp.linalg.lstsq(A, y, rcond=None)[0][0]
    return slope
