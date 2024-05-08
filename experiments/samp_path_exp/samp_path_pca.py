"""Explore the path of the samples in the parameter space using PCA."""
import os
import pickle

# from utils import get_flattened_key_shapes
from typing import List, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
import plotly.graph_objects as go # plotly is not installed by default within the requirements.txt!
import plotly.io as pio
from jax.tree_util import tree_flatten
from sklearn.decomposition import PCA
from tqdm import tqdm

pio.templates.default = 'simple_white'


def get_flattened_key_shapes(d: dict) -> Tuple[List[str], List[Tuple[int]]]:
    """
    Recursively get the keys of a dictionary and its subdictionaries.

    Args:
    d: dict

    Returns:
    list: list of keys
    """
    keys = []
    shape = []
    for k, v in d.items():
        if isinstance(v, dict):
            keys.extend([f'{k}.{kk}' for kk in get_flattened_key_shapes(v)[0]])
            shape.extend(get_flattened_key_shapes(v)[1])
        else:
            keys.append(k)
            shape.append(v.shape)
    return keys, shape


RESULT_DIR = '../../results/'
EXP = 'fcn_bnns/'
DATE = '2024-01-16-13-57-52/'
EXPS = [f for f in os.listdir(f'{RESULT_DIR}{EXP}{DATE}') if f.endswith('.pkl')]

RESULT_PATH = f'{RESULT_DIR}{EXP}{DATE}/'
SAVE_DIR = f'{RESULT_DIR}{EXP}{DATE}/pca/'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


def sampling_path_pca(respath: str, exp: str, savedir: str) -> pd.DataFrame:
    """Perform PCA on the samples from the posterior."""
    with open(f'{respath}{exp}', 'rb') as f:
        data = pickle.load(f)
    flat_keys, flat_shapes = get_flattened_key_shapes(data)
    flat_shapes = [np.prod(x[2:]).item() for x in flat_shapes]
    flat_data, _ = tree_flatten(data)
    all_chains_df = []
    for chain in range(flat_data[0].shape[0]):
        one_chain_data = [x[chain, ...] for x in flat_data]
        one_chain_data = jnp.concatenate(
            [
                jnp.reshape(jnp.reshape(x, (-1, *x.shape[1:])), (x.shape[0], -1))
                for x in one_chain_data
            ],
            axis=1,
        )
        assert one_chain_data.shape[1] == sum(flat_shapes)
        # perform a PCA
        pca = PCA(n_components=3)
        pca.fit(one_chain_data)
        explained_variance = sum(pca.explained_variance_ratio_)
        pcomps = np.abs(pca.components_).sum(axis=0)
        pcomp_df = pd.DataFrame(
            {'pcomp': pcomps, 'block': np.repeat(flat_keys, flat_shapes)}
        )
        # mean and std
        pcomp_df['mean'] = pcomp_df.groupby('block')['pcomp'].transform('mean')
        pcomp_df['std'] = pcomp_df.groupby('block')['pcomp'].transform('std')
        pcomp_df['type'] = pcomp_df['block'].str.split('.').str[-1]
        pcomp_df['explained_variance'] = explained_variance
        pcomp_df['chain'] = chain
        all_chains_df.append(pcomp_df)

        # plot 3d scatter plot
        if chain == 0:
            pca_data = pca.transform(one_chain_data)
            fig = go.Figure(
                data=go.Scatter3d(
                    x=pca_data[:, 0],
                    y=pca_data[:, 1],
                    z=pca_data[:, 2],
                    marker=dict(
                        size=2,
                        color=np.arange(pca_data.shape[0]),
                        colorscale='Viridis',
                    ),
                    line=dict(
                        color=np.arange(pca_data.shape[0]),
                        colorscale='Viridis',
                        width=2,
                    ),
                )
            )
            fig.write_image(
                f'{savedir}firstchainpath{exp.split("|")[1].removesuffix(".data")}.png'
            )

    all_chains_df = pd.concat(all_chains_df)
    all_chains_df['dataset'] = exp.split('|')[1].removesuffix('.data')
    all_chains_df['repl'] = int(exp.removesuffix('|1|Normal.pkl')[-1])
    return all_chains_df


all_exp_df = []
for exp in tqdm(EXPS):
    all_exp_df.append(sampling_path_pca(RESULT_PATH, exp, SAVE_DIR))
all_exp_df = pd.concat(all_exp_df)
all_exp_df.to_csv(f'{SAVE_DIR}pca.csv', index=False)
