"""Utility functions for the UI."""
import sys
from typing import (
    List,
    Tuple,
    Union,
)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('../..')
from module_sandbox.diagnostics.ess import effective_sample_size  # noqa: E402
from module_sandbox.diagnostics.gelman import split_chain_r_hat  # noqa: E402


def plot_sample_paths(
    sample_dict: dict, dim1: int, dim2: int, param: str, colors: List[str]
) -> plt.Figure:
    """
    Plot the sample paths of the parameter.

    Args:
        sample_dict: A dictionary containing the sample paths for each chain.
        dim1: The first dimension index.
        dim2: The second dimension index.
        param: The parameter name.
        colors: A list of colors for each chain.

    Returns:
        The matplotlib Figure object containing the plot.
    """
    sns.set_theme(style='whitegrid')
    fig_parameter_samples, ax = plt.subplots(figsize=(10, 6))
    for model in sample_dict.keys():
        sns.lineplot(
            x=range(sample_dict[model].shape[0]),
            y=(
                sample_dict[model][:, dim1, dim2]
                if len(sample_dict[model].shape) == 3
                else sample_dict[model][:, dim1]
            ),
            # color the lines according to the chain
            hue=colors,
            ax=ax,
            legend=False,
        )
    ax.set_xlabel('Sample')
    ax.set_ylabel(f'W1[{dim1}, {dim2}]')
    ax.set_title(f'Sample Paths for {param}[{dim1}, {dim2}]')
    plt.close(fig_parameter_samples)
    return fig_parameter_samples


def calculate_diagnostics(
    posterior_samples: dict,
    param: str,
    good_chains: List[int],
    truncation: Union[int, None] = None,
) -> Tuple[dict, jnp.ndarray]:
    """Calculate the diagnostics for the parameter."""
    if truncation is None:
        truncation = posterior_samples[param].shape[1]
    param_split_chain_rhat = split_chain_r_hat(
        posterior_samples[param][:, :truncation, ...],
        n_splits=4,
        rank_normalize=True,
    )
    ess_values, ess_means = effective_sample_size(
        posterior_samples[param][:, :truncation, ...]
    )
    return param_split_chain_rhat, ess_values


def visualize_rhat(param_split_chain_rhat: dict) -> plt.Figure:
    """Visualize the Rhat values with Histograms."""
    fig_param_rhat, axs = plt.subplots(1, 4, figsize=(20, 5))
    for i, key in enumerate(param_split_chain_rhat.keys()):
        if i < 4:
            sns.histplot(
                x=jnp.ravel(param_split_chain_rhat[key]),
                ax=axs[i],
                bins=50,
            )
            # add the mean as annotation
            axs[i].annotate(
                f'mean: {float(jnp.mean(param_split_chain_rhat[key])):.2f}',
                xy=(0.95, 0.95),
                xycoords='axes fraction',
                fontsize=12,
                ha='right',
                va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
            )
            axs[i].set_title(f'Histogram of {key}')
            axs[i].set_xlabel('')
    plt.close(fig_param_rhat)
    return fig_param_rhat


def visualize_ess(ess_values: jnp.ndarray) -> plt.Figure:
    """Visualize the ESS values."""
    ess_plot, axs = plt.subplots(1, 1, figsize=(10, 5))
    sns.histplot(
        x=jnp.ravel(ess_values),
        bins=50,
    )
    # add the mean as annotation
    axs.annotate(
        f'mean: {float(jnp.mean(ess_values)):.2f}',
        xy=(0.95, 0.95),
        xycoords='axes fraction',
        fontsize=12,
        ha='right',
        va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
    )
    axs.set_title('Histogram of ESS')
    axs.set_xlabel('')
    plt.close(ess_plot)
    return ess_plot


def visualize_pp_rhat(pp_split_chain_rhat: dict) -> plt.Figure:
    """Visualize the function space Rhat values with Histograms."""
    fig_pp_rhat, axs = plt.subplots(1, 4, figsize=(20, 5))
    for i, key in enumerate(pp_split_chain_rhat.keys()):
        if i < 4:
            sns.histplot(
                x=jnp.ravel(pp_split_chain_rhat[key]),
                ax=axs[i],
                bins=50,
                color='violet',
            )
            axs[i].annotate(
                f'mean: {float(jnp.mean(pp_split_chain_rhat[key])):.2f}',
                xy=(0.95, 0.95),
                xycoords='axes fraction',
                fontsize=12,
                ha='right',
                va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
            )
            axs[i].set_title(f'Histogram of {key}')
            axs[i].set_xlabel('')
    plt.close(fig_pp_rhat)
    return fig_pp_rhat
