"""Effective sample size (ESS) diagnostics for MCMC chains."""

import jax
import jax.numpy as jnp
import numpyro

from src.utils import vectorized_rank_normalization


def effective_sample_size(
    samples: jax.Array, rank_normalization: bool = False
) -> tuple[jax.Array, float]:
    """
    Calculate the effective sample size for a list of MCMC chains.

    Args:
    samples (ndarray): MCMC chains as a JAX array with the shape
      (n_chains, n_samples, ...).
    rank_normalization (bool): Whether to rank normalize the samples before calculating

    Returns:
    ess (ndarray): Effective sample size for each parameter (the other dimensions) as a
      JAX array.
    ess_mean (float): Mean of the effective sample size over all parameters.
    """
    # calculate the statistic for each parameter (the other dimensions) and
    # also in the end return the average of the statistics
    # first reshape the first dimension to (n_chains, n_samples, ...)
    chains = samples.copy()
    n_chains = chains.shape[0]
    if rank_normalization:
        chains = vectorized_rank_normalization(chains)
    ess = []
    for i in range(n_chains):
        ess.append(
            numpyro.diagnostics.effective_sample_size(
                jnp.expand_dims(chains[i, ...], axis=0)
            )
        )
    ess = jnp.array(ess)
    # this result could also be reasonable to return as one could for instance check
    # for chains with low ess (< 50 as recommended in Vethari et al. 2021)
    # sum over the chain ess to get the ess for each parameter
    ess = jnp.sum(ess, axis=0)
    return ess, jnp.mean(ess)
