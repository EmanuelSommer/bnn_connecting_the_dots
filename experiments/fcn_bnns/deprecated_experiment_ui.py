"""Interactive UI for exploring the results of the experiments."""

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
import probabilisticml as pml
import seaborn as sns
import streamlit as st
import yaml
from numpyro.diagnostics import hpdi
from probabilisticml.inspection.prediction import MCMCPredictor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

sys.path.append('../..')
from module_sandbox.diagnostics.ess import effective_sample_size  # noqa: E402
from module_sandbox.diagnostics.gelman import split_chain_r_hat  # noqa: E402
from module_sandbox.shallow_bnn_numpyro import gaussian_mlp_from_config  # noqa: E402
from module_sandbox.utils import (  # noqa: E402
    add_chain_dimension,
    flatten_chain_dimension,
    mse,
)
from module_sandbox.visualization.posterior_predictive import (  # noqa: E402
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

# Select pickle file
pickle_files = os.listdir(os.path.join(results_folder, selected_date))
pickle_files = [f for f in pickle_files if f.endswith('.pkl')]
exp_ids = [f.removesuffix('.pkl') for f in pickle_files]
exp_info = [f.split('|') for f in exp_ids]

if len(exp_info) == 0:
    st.warning('No experiments found!')
    st.stop()

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

# Load and display pickle file
pickle_path = os.path.join(results_folder, selected_date, selected_id + '.pkl')
with open(pickle_path, 'rb') as f:
    data = pickle.load(f)

# get the data
shallow_regr_dataset = pml.data.dataset.DatasetTabular(
    data_path='../../data/' + selected_exp[0],
    target_indices=[],
    split_spec={'train': 0.8, 'val': 0.2},
    seed=int(selected_exp[7]),
    standardize=True,
)
X_SNN_train, Y_SNN_train = shallow_regr_dataset.get_data(split='train', data_type='jax')
X_SNN_val, Y_SNN_val = shallow_regr_dataset.get_data(split='val', data_type='jax')


# Baseline performance --------------------------------------------------------
stats_cols = st.columns(2)
with stats_cols[0]:
    st.subheader('Baseline Performance (RMSE)')
# Also fit with a linear model using sklearn for reference
reg_shallow = LinearRegression().fit(X_SNN_train, Y_SNN_train)
mse_linear_model = np.mean(
    (reg_shallow.predict(X_SNN_val).ravel() - Y_SNN_val.ravel()) ** 2
)
mse_linear_model_train = np.mean(
    (reg_shallow.predict(X_SNN_train).ravel() - Y_SNN_train.ravel()) ** 2
)


reg_shallow_rf = RandomForestRegressor(max_depth=20, n_estimators=100, random_state=0)
reg_shallow_rf.fit(X_SNN_train, Y_SNN_train.ravel())
mse_rf = np.mean((reg_shallow_rf.predict(X_SNN_val) - Y_SNN_val.ravel()) ** 2)
with stats_cols[0]:
    st.write('Linear Model: ', np.round(np.sqrt(mse_linear_model), 3))
    st.write('Linear Model Train: ', np.round(np.sqrt(mse_linear_model_train), 3))
    st.write('Random Forest: ', np.round(np.sqrt(mse_rf), 3))

# Load the model and samples --------------------------------------------------------
with open(
    f'../../results/fcn_bnns/{selected_date}/mconfig_{selected_id}.json', 'r'
) as f:
    model_config = json.load(f)
shallow_regr_bnn = gaussian_mlp_from_config(model_config)

# load the samples
sample_path = f'../../results/fcn_bnns/{selected_date}/{selected_id}.pkl'
with open(sample_path, 'rb') as f:
    posterior_samples = pickle.load(f)
    # if selected_exp[5] == "True":
    #     for posterior_samples_key in posterior_samples.keys():
    #         warmup_size = (
    #             posterior_samples[list(posterior_samples.keys())[0]].shape[1] -
    #             int(selected_exp[4])
    #         )
    #         posterior_samples[posterior_samples_key] = (
    #             posterior_samples[posterior_samples_key][:, warmup_size:, ...]
    #         )
    posterior_samples_raw = copy.deepcopy(posterior_samples)
    posterior_samples_raw = flatten_chain_dimension(posterior_samples_raw)

with stats_cols[1]:
    st.subheader('Posterior Sample Shapes')
    cols = st.columns(2)
    for w in posterior_samples.keys():
        cols[0].write(w)
        cols[1].write(posterior_samples[w].shape)


# plot the parameter traces --------------------------------------------------------
# first allow to select the parameter and then the two dimensions
with sb:
    st.subheader('Choose Parameter')
    param = st.selectbox('Select Parameter', list(posterior_samples.keys()))
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

sample_dict = {'model': posterior_samples_raw[param]}
n_chains = posterior_samples[param].shape[0]
n_samples = posterior_samples[param].shape[1]
chain_names = [f'chain_{i}' for i in range(n_chains)]
colors = np.repeat(chain_names, n_samples, axis=0)


def plot_sample_paths(sample_dict, dim1, dim2, param):
    """Plot the sample paths of the parameter."""
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


# calculate the diagnostics --------------------------------------------------------
def calculate_diagnostics(posterior_samples, param):
    """Calculate the diagnostics for the parameter."""
    param_split_chain_rhat = split_chain_r_hat(
        posterior_samples[param], n_splits=4, rank_normalize=True
    )
    ess_values, ess_means = effective_sample_size(posterior_samples[param])
    return param_split_chain_rhat, ess_values


def visualize_rhat(param_split_chain_rhat):
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


def visualize_ess(ess_values):
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


# posterior predictive analysis -----------------------------------------
rng_key_predict = jax.random.PRNGKey(0)
predictor = MCMCPredictor(
    model=shallow_regr_bnn,
    rng_key=rng_key_predict,
    samples=posterior_samples_raw,
)
preds = (
    predictor.predict(
        X=X_SNN_val,
        Y=None,
    )
).squeeze()

preds_chain_dim = add_chain_dimension(
    {'pp': preds}, n_chains=posterior_samples['W1'].shape[0]
)['pp']


rmse_per_chain = {}
for i in range(preds_chain_dim.shape[0]):
    rmse_per_chain[f'chain_{i}'] = np.sqrt(mse(preds_chain_dim[i], Y_SNN_val)[0])
# now visualize the rmse for each chain
fig_rmse, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    x=list(rmse_per_chain.keys()),
    y=list(rmse_per_chain.values()),
    ax=ax,
)

ax.set_xlabel('Chain')
ax.set_ylabel('RMSE')
ax.set_title('RMSE per Chain')
plt.close(fig_rmse)

rmse_table = pd.DataFrame(rmse_per_chain, index=['RMSE']).T
bad_chains = rmse_table[rmse_table['RMSE'] > np.sqrt(mse_linear_model)].index
bad_chains = bad_chains.str.split('_').str[1].astype(int).values
bad_chains = bad_chains.tolist()
if len(bad_chains) == n_chains:
    st.warning('All chains are bad. Can\'t continue.')
    st.stop()
good_chains = [i for i in range(n_chains) if i not in bad_chains]
good_chains_pred_indices = np.concatenate(
    [np.arange(n_samples) + (n_samples * i) for i in good_chains]
)
rmse_table = rmse_table.sort_values(by='RMSE', ascending=True)

# Calculate training RMSE
preds_train = (
    predictor.predict(
        X=X_SNN_train,
        Y=None,
    )
).squeeze()
preds_train_chain_dim = add_chain_dimension(
    {'pp': preds_train}, n_chains=posterior_samples['W1'].shape[0]
)['pp']
rmse_per_chain_train = {}
for i in range(preds_train_chain_dim.shape[0]):
    rmse_per_chain_train[f'chain_{i}'] = np.sqrt(
        mse(preds_train_chain_dim[i], Y_SNN_train)[0]
    )
# no visualization
rmse_table_train = pd.DataFrame(rmse_per_chain_train, index=['RMSE']).T
# sort like the validation table and append there
rmse_table_train = rmse_table_train.loc[rmse_table.index]
# add as an additional column
rmse_table['RMSE Train'] = rmse_table_train['RMSE']


# all entries of the df that are > np.sqrt(mse_linear_model) should get a red background
def color_cells(x):
    """Color the cells of the table."""
    return 'background-color: red' if x > np.sqrt(mse_linear_model) else ''


rmse_table = rmse_table.style.applymap(color_cells)
rmse_table = rmse_table.format('{:.3f}')


def visualize_pp_rhat(pp_split_chain_rhat):
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


pp_stats_cols = st.columns(2)
with pp_stats_cols[0]:
    st.subheader('Posterior Predictive Performance (RMSE)')
    # note that only the RMSE of the good chains is taken into account so first filter
    # add a note about the number and percentage of bad chains
    st.write(
        f'Number of bad chains: {len(bad_chains)} ({len(bad_chains)/n_chains*100:.2f}%)'
    )
    st.write(
        'RMSE: ',
        np.round(np.sqrt(mse(preds[good_chains_pred_indices, :], Y_SNN_val)[0]), 3),
        ' (only good chains)',
    )
    st.write(
        'RMSE Train: ',
        np.round(np.sqrt(mse(preds_train, Y_SNN_train)[0]), 3),
        ' (all chains)',
    )
    st.table(rmse_table)

with pp_stats_cols[1]:
    st.subheader('Calibration (only good chains)')
    for q in [0.5, 0.75, 0.98]:
        hpdi_preds = hpdi(preds[good_chains_pred_indices], q)
        acc_hpdi = jnp.mean(
            (hpdi_preds[0, :] <= Y_SNN_val.squeeze())
            & (hpdi_preds[1, :] >= Y_SNN_val.squeeze())
        )
        st.write(f'Accuracy of {int(q*100)}% HPDI: {acc_hpdi:.2f}')

# plot the overall rmse only taking the first n samples into account
rmse_over_n_samples = {}
rmse_over_n_samples_chainwise = {f'chain_{i}': {} for i in good_chains}
steps = [i + 1 for i in range(19)] + list(range(20, n_samples + 1, 10))
for n in steps:
    rmse_over_n_samples[n + 1] = np.sqrt(
        mse(
            (
                preds_chain_dim[good_chains, : (n + 1), :].reshape(
                    -1, preds_chain_dim.shape[-1]
                )
            ),
            Y_SNN_val,
        )[0]
    )
    for i in good_chains:
        rmse_over_n_samples_chainwise[f'chain_{i}'][n + 1] = np.sqrt(
            mse(preds_chain_dim[i, : (n + 1), :], Y_SNN_val)[0]
        )

fig_rmse_over_n_samples, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(
    x=list(rmse_over_n_samples.keys()),
    y=list(rmse_over_n_samples.values()),
    ax=ax,
    color='blue',
)
# also plot the individual chains
for i, c in enumerate(good_chains):
    sns.lineplot(
        x=list(rmse_over_n_samples_chainwise[f'chain_{c}'].keys()),
        y=list(rmse_over_n_samples_chainwise[f'chain_{c}'].values()),
        ax=ax,
        color='grey',
        label=list(rmse_over_n_samples_chainwise.keys())[i],
        alpha=0.5,
    )

ax.set_xlabel('Number of Samples')
ax.set_ylabel('RMSE')
ax.set_title('RMSE of the pooled samples in blue')
plt.close(fig_rmse_over_n_samples)

# visualizations --------------------------------------------------------
with sb:
    st.subheader('Choose Visualizations')
    visualize_parameter_traces = st.checkbox('Visualize parameter traces', value=False)
    diagnostic_plots = st.checkbox('Visualize diagnostic plots', value=False)
    interchain_means_plot = st.checkbox('Visualize interchain PP means', value=False)
    rmse_over_n_samples_plot = st.checkbox(
        'Visualize RMSE over number of samples', value=False
    )

trace_plot_cols = st.columns([3, 4, 3])
# with trace_plot_cols[0]:
#     st.pyplot(fig_rmse, use_container_width=True)
if visualize_parameter_traces:
    fig_parameter_samples = plot_sample_paths(
        sample_dict, dim1=dim1, dim2=dim2, param=param
    )
    with trace_plot_cols[1]:
        st.subheader('Parameter Traces')
        st.pyplot(fig_parameter_samples, use_container_width=True)

if diagnostic_plots:
    param_split_chain_rhat, ess_values = calculate_diagnostics(posterior_samples, param)
    fig_param_rhat = visualize_rhat(param_split_chain_rhat)
    ess_plot = visualize_ess(ess_values)
    pp_split_chain_rhat = split_chain_r_hat(preds_chain_dim, n_splits=4)
    fig_pp_rhat = visualize_pp_rhat(pp_split_chain_rhat)
    st.subheader('Diagnostics')
    diagnostic_cols = st.columns([3, 1])
    with diagnostic_cols[0]:
        st.pyplot(fig_param_rhat, use_container_width=True)
        st.pyplot(fig_pp_rhat, use_container_width=True)
    with diagnostic_cols[1]:
        st.pyplot(ess_plot, use_container_width=True)


if interchain_means_plot:
    st.subheader('Interchain PP Means')
    interchain_means_cols = st.columns([3, 4, 3])
    interchain_means_normal = pp_interchain_means(
        posterior_samples, shallow_regr_bnn, X_SNN_val
    )
    fig, ax = visualize_pp_chain_means(interchain_means_normal, 1000, show=False)
    plt.close(fig)
    with interchain_means_cols[1]:
        st.pyplot(fig, use_container_width=True)

if rmse_over_n_samples_plot:
    st.subheader('RMSE over Number of Samples')
    rmse_over_n_samples_cols = st.columns([3, 4, 3])
    with rmse_over_n_samples_cols[1]:
        st.pyplot(fig_rmse_over_n_samples, use_container_width=True)
