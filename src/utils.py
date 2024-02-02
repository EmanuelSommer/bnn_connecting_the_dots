"""General utility functions for the module sandbox."""

import copy
import math
from typing import Callable

import jax
import jax.numpy as jnp


def rolling_average(x: jnp.array, window_size: int) -> jnp.array:
    """Calculate the rolling average of a JAX array."""
    return jnp.convolve(x.ravel(), jnp.ones(window_size), 'valid') / window_size


def mse(preds: jnp.array, y: jnp.array) -> tuple[float, float]:
    """Calculate the mean squared error and its standard deviation."""
    return (
        jnp.mean((preds.mean(axis=0).squeeze() - y.squeeze()) ** 2),
        jnp.std((preds.mean(axis=0).squeeze() - y.squeeze()) ** 2),
    )


def apply_statistic_to_dict(samples_dict: dict, statistic: Callable, **kwargs) -> dict:
    """
    Apply a statistic to each entry of a dictionary of samples.

    Args:
    samples_dict (dict): Dictionary of samples.
    statistic (function): Statistic to apply to each entry of the dictionary.
    **kwargs: Keyword arguments to pass to the statistic function.

    Returns:
    dict: Dictionary with the same keys as the input dictionary and the values are the
      statistic applied to the samples.
    """
    return {key: statistic(samples_dict[key], **kwargs) for key in samples_dict.keys()}


def add_chain_dimension(samples_dict: dict, n_chains: int):
    """
    Add a chain dimension to each entry of a dictionary of samples.

    Args:
    samples_dict (dict): Dictionary of samples.
    n_chains (int): Number of chains.

    Returns:
    dict: Dictionary with the same keys as the input dictionary and the values are the
      samples with an additional chain dimension.
    """
    # copy the samples_dict
    samples_d = copy.deepcopy(samples_dict)
    for key in samples_d.keys():
        samples_d[key] = jnp.reshape(
            samples_d[key], (n_chains, -1, *samples_d[key].shape[1:])
        )
    return samples_d


def flatten_chain_dimension(samples_dict: dict):
    """
    Flatten the chain dimension of each entry of a dictionary of samples.

    Args:
    samples_dict (dict): Dictionary of samples.

    Returns:
    dict: Dictionary with the same keys as the input dictionary and the values are the
      samples with the chain dimension flattened.
    """
    # copy the samples_dict
    samples_d = copy.deepcopy(samples_dict)
    for key in samples_d.keys():
        samples_d[key] = samples_d[key].reshape(-1, *samples_d[key].shape[2:])
    return samples_d


def rank_normalization(samples: jnp.array) -> jnp.array:
    """
    Rank normalization of a JAX array.

    Args:
    samples (ndarray): JAX array with the shape (n_samples, ...).

    Returns:
    ndarray: JAX array with the same shape as the input array.
    """
    n_samples = math.prod(samples.shape)
    ranks = jax.scipy.stats.rankdata(samples, axis=None).reshape(samples.shape)
    tmp = (ranks - 0.375) / (n_samples + 0.25)
    return jax.scipy.stats.norm.ppf(tmp)


def vectorized_rank_normalization(samples: jnp.array) -> jnp.array:
    """
    Vectorized rank normalization of a JAX array.

    Args:
    samples (ndarray): JAX array with the shape (n_chains, n_samples, ...).

    Returns:
    ndarray: JAX array with the same shape as the input array.
    """
    # reshape the first two dimensions to one
    n_chains = samples.shape[0]
    flattend_samples = samples.copy().reshape(-1, *samples.shape[2:])
    # apply the rank normalization for each ... dimension separately
    result = jnp.apply_along_axis(rank_normalization, 0, flattend_samples)
    # reshape the result to the original shape
    return jnp.reshape(result, (n_chains, -1, *result.shape[1:]))
