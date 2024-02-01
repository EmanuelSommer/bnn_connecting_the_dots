"""Custom visualization functions for posterior predictive checks."""

import copy

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('../..')
from probabilisticml.inspection.prediction import MCMCPredictor


def pp_interchain_means(
    posterior_samples_dict: dict,
    model,
    X_val: jax.Array,
    rng_key=None,
    center_means: bool = True,
) -> float:
    """
    Posterior Predictive Interchain means.

    Args:
    posterior_samples_dict (dict): Dictionary of posterior samples for each Parameter.
    model (probabilisticml.training.models.Model): Model object.
    n_chains (int): Number of MCMC chains.
    X_val (jax.Array): Validation data.
    rng_key (jax.random.PRNGKey): Random number generator key.

    Returns:
    pp_interchain_var (float): Posterior Predictive Interchain Variance.
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    post_sample_dict = copy.deepcopy(posterior_samples_dict)
    for key in post_sample_dict.keys():
        post_sample_dict[key] = post_sample_dict[key].reshape(
            -1, *post_sample_dict[key].shape[2:]
        )
    n_chains = posterior_samples_dict[list(posterior_samples_dict.keys())[0]].shape[0]
    predictor = MCMCPredictor(
        model=model,
        rng_key=rng_key,
        samples=post_sample_dict,
    )
    preds = (
        predictor.predict(
            X=X_val,
            Y=None,
        )
    ).squeeze()
    # reshape to (n_chains, n_samples, ...)
    preds = jnp.reshape(preds, (n_chains, -1, *preds.shape[1:]))
    # mean of each chain
    chain_means = jnp.mean(preds, axis=1)

    if center_means:
        # center the means of each chain with the mean of the first chain
        chain_means = chain_means - chain_means.mean(axis=0)

    return chain_means


def visualize_pp_chain_means(
    chain_means: jax.Array, n_data_points: int = 100, show: bool = True
):
    """
    Visualize the Posterior Predictive Interchain means.

    Args:
    chain_means (jax.Array): Posterior Predictive Interchain means.
    """
    n_chains = chain_means.shape[0]
    n_val = chain_means.shape[1]
    alpha = 0.1

    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_val):
        if i < n_data_points:
            sns.lineplot(
                x=[i + 1 for i in range(n_chains)],
                y=chain_means[:, i],
                ax=ax,
                alpha=alpha,
                color='black',
            )
    ax.set_xlabel('Chain')
    ax.set_ylabel('Centered Posterior Predictive Mean')
    ax.set_title('Posterior Predictive Interchain Means')
    # discrete +1 x axis
    ax.set_xticks([i + 1 for i in range(n_chains)])
    if show:
        plt.show()
    return fig, ax
