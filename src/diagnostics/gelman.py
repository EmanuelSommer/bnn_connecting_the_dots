"""Gelman-Rubin style MCMC convergence diagnostics."""

import jax
import jax.numpy as jnp

from src.utils import vectorized_rank_normalization


def gelman_split_r_hat(
    samples: jax.Array,
    n_splits: int,
    rank_normalize: bool = True,
) -> jax.Array:
    """
    Calculate the split Gelman-Rubin R-hat statistic for samples of MCMC chains.

    Args:
    samples (ndarray): MCMC chains as a JAX array with the shape
    (n_chains, n_samples, ...).
    n_splits (int): Number of splits of the chains.

    Returns:
    rhat (ndarray): R-hat statistic for each parameter (the other dimensions) as a JAX
    array.
    """
    n_chains = samples.shape[0]
    if (n_chains == 1) & (n_splits == 1):
        raise ValueError('Only one chain, R_hat is not defined')

    # rank normalize the samples
    if rank_normalize:
        samples = vectorized_rank_normalization(samples)

    overall_splits = n_chains * n_splits
    n_samples = samples.shape[1] / n_splits
    if (n_samples % 1) != 0:
        raise ValueError('Number of samples must be divisible by n_splits')
    if n_samples < 50:
        print(
            'Warning! Nbr of samples should be at least 50 times the number of splits'
        )

    # flatten the chains first jnp.reshape(b, (-1, b.shape[-1]))
    chains = samples.copy().reshape(-1, *samples.shape[2:])
    # (n_eval_chains, n_samples, ...)
    chains = jnp.reshape(chains, (overall_splits, -1, *chains.shape[1:]))
    chain_means = jnp.mean(chains, axis=1)
    whithin_chain_var = jnp.mean(jnp.var(chains, axis=1, ddof=1), axis=0)
    between_chain_var = jnp.var(chain_means, axis=0, ddof=1)
    numerator = ((n_samples - 1) / n_samples) * whithin_chain_var + between_chain_var
    rhat = jnp.sqrt(numerator / whithin_chain_var)
    return rhat


def split_chain_r_hat(
    samples: jax.Array,
    n_splits: int,
    rank_normalize: bool = True,
) -> jax.Array:
    """
    Calculate the split chain R-hat statistic for samples of MCMC chains.

    Args:
    samples (ndarray): MCMC chains as a JAX array with the shape
    (n_chains, n_samples, ...).
    n_splits (int): Number of splits of the chains.
    rank_normalize (bool): Whether to rank normalize the samples before calculating

    Returns:
    dict: Dictionary with the following entries:
        split_chain_rhat (ndarray): split R-hat statistic for each chain and parameter
          (the other dimensions) as a JAX array.
        avg_split_chain_rhat (float): Mean of the split chain R-hat statistic over all
          parameters.
        rhat (ndarray): classic R-hat statistic for each parameter
        (the other dimensions) as a JAX array.
        split_normalized_rhat (ndarray): split R-hat normalized R-hat statistic.
    """
    n_chains = samples.shape[0]
    if n_chains == 1:
        raise ValueError('Only one chain, R_hat is not defined')

    # calc the splitrhat for each chain with a number of splits specified by n_splits
    # (n_chains, n_samples, ...)
    chains = samples.copy()
    split_chain_rhat = []
    for i in range(n_chains):
        split_chain_rhat.append(
            gelman_split_r_hat(
                chains[i, jnp.newaxis, ...],
                n_splits=n_splits,
                rank_normalize=rank_normalize,
            )
        )
    split_chain_rhat = jnp.array(split_chain_rhat)

    # then average the splitrhat over the chains for each parameter
    avg_split_chain_rhat = jnp.mean(split_chain_rhat, axis=0)

    # now calculate the rhat for each parameter
    rhat = gelman_split_r_hat(samples.copy(), n_splits=1, rank_normalize=True)

    # divide the rhat by the avg_split_chain_rhat
    split_normalized_rhat = (avg_split_chain_rhat + rhat - 1) / (avg_split_chain_rhat)

    return {
        'split_chain_rhat': split_chain_rhat,
        'avg_split_chain_rhat': avg_split_chain_rhat,
        'rhat': rhat,
        'split_normalized_rhat': split_normalized_rhat,
    }
