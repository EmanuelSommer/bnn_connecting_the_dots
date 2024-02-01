"""
Interactive UI for exploring the results of the experiments.

Run with `streamlit run 1_experiment_ui.py` from the experiments/fcn_bnns folder.
"""

import copy
import json
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns
import streamlit as st
import yaml
from numpyro.diagnostics import hpdi

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

sys.path.append('../..')
import probabilisticml as pml
from probabilisticml.inspection.prediction import MCMCPredictor
from experiments.fcn_bnns.utils.ui_utils import (  # noqa: E402
    calculate_diagnostics,
    plot_sample_paths,
    visualize_ess,
    visualize_pp_rhat,
    visualize_rhat,
)

# from src.diagnostics.ess import effective_sample_size  # noqa: E402
from src.diagnostics.gelman import split_chain_r_hat  # noqa: E402
from src.shallow_bnn_numpyro import gaussian_mlp_from_config  # noqa: E402
from src.utils import (  # noqa: E402
    add_chain_dimension,
    flatten_chain_dimension,
    mse,
)
from src.visualization.posterior_predictive import (  # noqa: E402
    pp_interchain_means,
    visualize_pp_chain_means,
)

st.set_page_config(
    page_title='Experiment Explorer',
    page_icon='🔍',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Define the path to the results folder
results_folder = '../../results/fcn_bnns/'

# Get the list of experiment dates (file names)
experiment_dates = os.listdir(results_folder)

# Streamlit app title
st.title('Experiment Viewer 🔍')

# init sidebar
sb = st.sidebar

# Select the experiment --------------------------------------------------------
# Select experiment date
with sb:
    st.subheader('Select Experiment')
    selected_date = st.selectbox(
        'Select Experiment Datetime', sorted(experiment_dates)[::-1]
    )

# Load and display config.yaml
config_path = os.path.join(results_folder, selected_date, 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
st.subheader('Config.yaml')
with st.expander('Configuration YAML of the selected experiment.'):
    st.write(config)

# load the runtime file which is runtime.txt
runtime_path = os.path.join(results_folder, selected_date, 'runtime.txt')
with open(runtime_path, 'r') as f:
    # the file has multiple rows each with the format: id: float
    # we want to get the id and the float value (runtime in minutes)
    # and store it in a dictionary
    runtime = {}
    for line in f.readlines():
        line = line.strip()
        if line:
            id, value = line.split(': ')
            runtime[id] = float(value)

# Display the runtime
with st.expander('Runtime'):
    st.write(runtime)

# Select pickle file
pickle_files = os.listdir(os.path.join(results_folder, selected_date))
pickle_files = [f for f in pickle_files if f.endswith('.pkl')]
exp_ids = [f.removesuffix('.pkl') for f in pickle_files]
exp_info = [f.split('|') for f in exp_ids]

if len(exp_info) == 0:
    st.warning('No experiments found!')
    st.stop()
else:
    exp_number_from_config = np.prod(
        [
            len(v) if isinstance(v, list) or isinstance(v, dict) else 1
            for v in config.values()
        ]
    )
    st.write(f'Number of experiments: {len(exp_info)}/{exp_number_from_config}')

labels = [
    'Exp',
    'Data',
    'Activation',
    'Architecture',
    '#Chains',
    '#Samples',
    'KeepWarmup',
    'Sampler',
    'Replication',
    'Prior Std',
    'Prior Distribution',
    'Warmstart',
]

cols = st.columns(len(exp_info[0]) - 1)
selected_exp = []
for i, col in enumerate(cols):
    selected_exp.append(
        col.selectbox(
            f'{labels[i+1]}',
            sorted(list(set([e[i + 1] for e in exp_info]))),
        )
    )

selected_id = '|'.join(selected_exp)
selected_id = [i for i in exp_ids if selected_id in i][0]

start = st.checkbox('Start the analysis', value=False)
if start:
    # get the data
    shallow_regr_dataset = pml.data.dataset.DatasetTabular(
        data_path='../../data/' + selected_exp[0],
        target_indices=[],
        split_spec={'train': 0.8, 'val': 0.2},
        seed=int(selected_exp[7]),
        standardize=True,
    )
    X_SNN_train, Y_SNN_train = shallow_regr_dataset.get_data(
        split='train', data_type='jax'
    )
    X_SNN_val, Y_SNN_val = shallow_regr_dataset.get_data(split='val', data_type='jax')
    val_threshold = min(400, X_SNN_val.shape[0])
    X_SNN_val = X_SNN_val[:val_threshold, ...]
    Y_SNN_val = Y_SNN_val[:val_threshold, ...]

    # Baseline performance --------------------------------------------------------
    stats_cols = st.columns(2)
    with stats_cols[0]:
        st.subheader('Baseline Performance (RMSE)')
    reg_shallow = LinearRegression().fit(X_SNN_train, Y_SNN_train)
    mse_linear_model = np.mean(
        (reg_shallow.predict(X_SNN_val).ravel() - Y_SNN_val.ravel()) ** 2
    )
    mse_linear_model_train = np.mean(
        (reg_shallow.predict(X_SNN_train).ravel() - Y_SNN_train.ravel()) ** 2
    )

    reg_shallow_rf = RandomForestRegressor(
        max_depth=10, n_estimators=100, random_state=0
    )
    reg_shallow_rf.fit(X_SNN_train, Y_SNN_train.ravel())
    mse_rf = np.mean((reg_shallow_rf.predict(X_SNN_val) - Y_SNN_val.ravel()) ** 2)
    with stats_cols[0]:
        st.write('Linear Model: ', np.round(np.sqrt(mse_linear_model), 3))
        st.write('Linear Model Train: ', np.round(np.sqrt(mse_linear_model_train), 3))
        st.write('Random Forest: ', np.round(np.sqrt(mse_rf), 3))

    # Load the model and samples -------------------------------------------------------
    with open(
        f'../../results/fcn_bnns/{selected_date}/mconfig_{selected_id}.json', 'r'
    ) as f:
        model_config = json.load(f)
    shallow_regr_bnn = gaussian_mlp_from_config(model_config)

    # load the samples
    sample_path = f'../../results/fcn_bnns/{selected_date}/{selected_id}.pkl'
    with open(sample_path, 'rb') as f:
        posterior_samples = pickle.load(f)
        # posterior_samples_raw = posterior_samples
        posterior_samples_raw = copy.deepcopy(posterior_samples)
        posterior_samples_raw = flatten_chain_dimension(posterior_samples_raw)

    with stats_cols[1]:
        st.subheader('Posterior Sample Shapes')
        cols = st.columns(2)
        for w in posterior_samples.keys():
            cols[0].write(w)
            cols[1].write(posterior_samples[w].shape)

with sb:
    st.subheader('Choose Analysis')
    visualize_parameter_traces = st.checkbox('Visualize parameter traces', value=False)
    posterior_predictive = st.checkbox('Assess PP performance', value=False)
    diagnostic_plots = st.checkbox('Visualize diagnostic plots', value=False)
    interchain_means_plot = st.checkbox('Visualize interchain PP means', value=False)
    rmse_over_samples_and_chains = st.checkbox(
        'Visualize RMSE over number of samples and chains', value=False
    )
    visualize_a_prediction = st.checkbox('Visualize a prediction', value=False)
    truncate_samples = st.number_input(
        'Truncate Samples',
        min_value=0,
        max_value=config['n_samples'],
        value=config['n_samples'],
    )


if posterior_predictive & start:
    # posterior predictive analysis -----------------------------------------
    st.cache_data()

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

    n_chains = config['n_chains']
    preds_chain_dim, preds = get_posterior_predictive(
        shallow_regr_bnn,
        posterior_samples_raw,
        X_SNN_val,
        n_chains=n_chains,
    )

    rmse_per_chain = {}
    for i in range(preds_chain_dim.shape[0]):
        rmse_per_chain[f'chain_{i}'] = np.sqrt(mse(preds_chain_dim[i], Y_SNN_val)[0])

    rmse_table = pd.DataFrame(rmse_per_chain, index=['RMSE']).T
    bad_chains = rmse_table[rmse_table['RMSE'] > np.sqrt(mse_linear_model)].index
    bad_chains = bad_chains.str.split('_').str[1].astype(int).values
    bad_chains = bad_chains.tolist()
    if len(bad_chains) == n_chains:
        st.warning('All chains are bad. Can\'t continue.')
        st.stop()
    good_chains = [i for i in range(n_chains) if i not in bad_chains]
    n_samples = config['n_samples']
    good_chains_pred_indices = np.concatenate(
        [np.arange(truncate_samples) + (n_samples * i) for i in good_chains]
    )
    rmse_table = rmse_table.sort_values(by='RMSE', ascending=True)

    # all entries of the df that are > np.sqrt(mse_linear_model) should get a red
    # background
    def color_cells(x):
        """Color the cells of the table."""
        return 'background-color: red' if x > np.sqrt(mse_linear_model) else ''

    rmse_table = rmse_table.style.map(color_cells)
    rmse_table = rmse_table.format('{:.3f}')

    pp_stats_cols = st.columns(2)
    with pp_stats_cols[0]:
        st.subheader('Posterior Predictive Performance (RMSE)')
        st.write(
            (
                f'Number of bad chains: {len(bad_chains)}'
                f' ({len(bad_chains)/n_chains*100:.2f}%)'
            )
        )
        st.write(
            'RMSE: ',
            np.round(np.sqrt(mse(preds[good_chains_pred_indices, :], Y_SNN_val)[0]), 3),
            ' (only good chains)',
        )
        st.table(rmse_table)
        st.write(
            (
                'The below statistics are used to assess whether we learned a constant'
                'function'
            )
        )
        st.write(
            'Mean of the |predictions|: ',
            np.round(jnp.abs(preds_chain_dim).mean(axis=1).mean(axis=1), 3),
        )
        st.write(
            'SD of the predictions: ',
            np.round(preds_chain_dim.std(axis=1).mean(axis=1), 3),
        )

    with pp_stats_cols[1]:
        st.subheader('Calibration (only good chains)')
        for q in [0.5, 0.75, 0.98]:
            hpdi_preds = hpdi(preds[good_chains_pred_indices], q)
            acc_hpdi = jnp.mean(
                (hpdi_preds[0, :] <= Y_SNN_val.squeeze())
                & (hpdi_preds[1, :] >= Y_SNN_val.squeeze())
            )
            st.write(f'Accuracy of {int(q*100)}% HPDI: {acc_hpdi:.2f}')

if rmse_over_samples_and_chains and start and posterior_predictive:
    st.subheader('RMSE over Samples and Chains')
    log_rmse_over_samples_and_chains = st.checkbox(
        'Log Transform the RMSE', value=False
    )

    # define a grid of samples and chains to evaluate the RMSE on

    # use all 1,2,3,..,10, 50, 100, 500 and every other 500 until truncate_samples
    sample_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sample_steps += [50, 100]
    sample_steps += list(range(500, truncate_samples, 500))
    sample_steps += [truncate_samples]
    sample_steps = np.unique(sample_steps)
    sample_steps = sample_steps[sample_steps <= truncate_samples]
    sample_steps = sample_steps.tolist()

    # calculate the rmse for each combination of samples and chains
    rmse_over_samples_and_chains = []
    for n_samples in sample_steps:
        for i, n_chains in enumerate(good_chains):
            rmse_over_samples_and_chains.append(
                np.sqrt(
                    mse(
                        preds_chain_dim[good_chains[: i + 1], :n_samples, ...].reshape(
                            -1, *preds_chain_dim.shape[2:]
                        ),
                        Y_SNN_val,
                    )[0]
                )
            )
    # visualize the rmse over samples and chains using a heatmap
    rmse_over_samples_and_chains = np.array(rmse_over_samples_and_chains).reshape(
        len(sample_steps), len(good_chains)
    )
    rmse_over_samples_and_chains = rmse_over_samples_and_chains[::-1, :]
    if log_rmse_over_samples_and_chains:
        rmse_over_samples_and_chains = np.log(rmse_over_samples_and_chains)
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(
        rmse_over_samples_and_chains,
        xticklabels=good_chains,
        yticklabels=sample_steps[::-1],
        cmap='YlGn_r',
    )
    # find the indices of the minimum value in the heatmap
    min_idx = np.unravel_index(
        np.nanargmin(rmse_over_samples_and_chains),
        rmse_over_samples_and_chains.shape,
    )
    # annotate with a red cross
    plt.scatter(min_idx[1] + 0.5, min_idx[0] + 0.5, marker='x', color='red')
    plt.xlabel('Chain')
    plt.ylabel('Number of Samples (Non-Linear!)')
    plt.title(
        (
            f'{"Log-" if log_rmse_over_samples_and_chains else ""}RMSE over Samples and'
            ' Chains (Lower is better)'
        )
    )
    plt.close(fig)

    rmse_over_samples_and_chains_cols = st.columns([3, 4, 3])
    with rmse_over_samples_and_chains_cols[1]:
        st.pyplot(fig, use_container_width=True)

# Individual parameter traces --------------------------------------------------------
if visualize_parameter_traces & start:
    trace_plot_cols = st.columns([3, 4, 3])
    with sb:
        st.subheader('Choose Parameter')
        param = st.selectbox(
            'Select Parameter for Traceplot', list(posterior_samples.keys())
        )
        dim1 = st.number_input(
            'Select Dimension 1',
            min_value=0,
            max_value=posterior_samples[param].shape[-2] - 1,
            value=0,
        )
        dim2 = st.number_input(
            'Select Dimension 2',
            min_value=0,
            max_value=posterior_samples[param].shape[-1] - 1,
            value=0,
        )

    samples = copy.deepcopy(posterior_samples[param][:, :truncate_samples, ...])
    # display a button to download the samples as a csv
    df_save = (
        pd.DataFrame(samples[:, :, dim1, dim2])
        if len(samples.shape) == 4
        else pd.DataFrame(samples[:, :, dim1])
    )
    # transpose
    df_save = df_save.T

    def convert_df(df):
        """Convert a dataframe to csv."""
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(df_save)

    st.download_button(
        'Press to Download', csv, 'file.csv', 'text/csv', key='download-csv'
    )

    sample_dict = {'model': samples.reshape(-1, *samples.shape[2:])}
    n_chains = posterior_samples[param].shape[0]
    n_samples = posterior_samples[param].shape[1]
    chain_names = [f'chain_{i}' for i in range(n_chains)]
    colors = np.repeat(chain_names, truncate_samples, axis=0)
    fig_parameter_samples = plot_sample_paths(
        sample_dict,
        dim1=dim1,
        dim2=dim2,
        param=param,
        colors=colors,
    )
    with trace_plot_cols[1]:
        st.subheader('Parameter Traces')
        st.pyplot(fig_parameter_samples, use_container_width=True)

if diagnostic_plots and start:
    if not posterior_predictive:
        st.warning('Posterior predictive not yet evaluated!')
        st.stop()

    param_diag = st.selectbox(
        'Select Parameter for Diagnostics', list(posterior_samples.keys())
    )
    # calculate the diagnostics --------------------------------------------------------
    param_split_chain_rhat, ess_values = calculate_diagnostics(
        posterior_samples,
        param_diag,
        good_chains,
        truncate_samples,
    )
    fig_param_rhat = visualize_rhat(param_split_chain_rhat)
    ess_plot = visualize_ess(ess_values)
    pp_split_chain_rhat = split_chain_r_hat(
        preds_chain_dim[good_chains, :truncate_samples, ...],
        n_splits=4,
    )
    fig_pp_rhat = visualize_pp_rhat(pp_split_chain_rhat)
    st.subheader('Diagnostics')
    st.text(
        (
            f'The parameter scale diagnostics (Parameter: {param_diag}) are shown in '
            'blue and the PP diagnostics in pink.'
        )
    )
    diagnostic_cols = st.columns([3, 1])
    with diagnostic_cols[0]:
        st.pyplot(fig_param_rhat, use_container_width=True)
        st.pyplot(fig_pp_rhat, use_container_width=True)
    with diagnostic_cols[1]:
        st.pyplot(ess_plot, use_container_width=True)


if interchain_means_plot and start:
    st.subheader('Interchain PP Means')
    interchain_means_cols = st.columns([3, 4, 3])
    trunc_posterior_samples = {
        k: v[:, :truncate_samples, ...] for k, v in posterior_samples.items()
    }
    interchain_means_normal = pp_interchain_means(
        trunc_posterior_samples, shallow_regr_bnn, X_SNN_val
    )
    fig, ax = visualize_pp_chain_means(interchain_means_normal, 1000, show=False)
    plt.close(fig)
    with interchain_means_cols[1]:
        st.pyplot(fig, use_container_width=True)

if visualize_a_prediction and posterior_predictive and start:
    st.subheader('Prediction')
    pred_obs = st.number_input(
        'Select Observation',
        min_value=0,
        max_value=X_SNN_val.shape[0] - 1,
        value=0,
    )
    fig = plt.figure(figsize=(10, 6))
    # color the chains by their RMSE
    for chain_index in range(preds_chain_dim.shape[0]):
        sns.lineplot(
            x=np.arange(truncate_samples),
            y=preds_chain_dim[chain_index, :truncate_samples, pred_obs],
            alpha=0.8,
            label=f'Chain {chain_index}',
        )
    plt.axhline(
        Y_SNN_val[pred_obs],
        color='black',
        label='True Value',
        alpha=0.8,
        linestyle='--',
    )
    plt.ylim([-2, 2])
    plt.title('Prediction')
    plt.legend()
    plt.close(fig)
    st.pyplot(fig, use_container_width=True)

    # print the mean and sd of each chains prediction
    st.subheader('Prediction Statistics')
    st.write(
        'Mean of the |predictions|: ',
        np.round(preds_chain_dim[:, :truncate_samples, pred_obs].mean(axis=1), 3),
    )
    st.write(
        'SD of the predictions: ',
        preds_chain_dim[:, :truncate_samples, pred_obs].std(axis=1),
    )
