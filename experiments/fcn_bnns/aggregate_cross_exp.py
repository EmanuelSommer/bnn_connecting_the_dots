"""
Aggregate Cross Experiment Results.

Script to aggregate some main results from the experiments to
identify roughly the best performing models.
The results are saved in an aggregated_data.csv file for further exploration.

The Script should be run from the results/fcn_bnns/datetime folder as follows:

`python ../../../experiments/fcn_bnns/aggregate_cross_exp.py`
"""
import sys

import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpyro.diagnostics import hpdi
from tqdm import tqdm

sys.path.append('../../..')
from experiments.fcn_bnns.utils.analysis_utils import (  # noqa: E402; load_config,
    evaluate_baselines,
    extract_exp_info,
    fit_baselines,
    get_exp_names,
    get_posterior_predictive,
    load_data,
    load_model,
    load_runtimes,
    load_samples,
)
from module_sandbox.utils import mse  # noqa: E402

CONFIG_PATH = 'config.yaml'
DATA_PATH = '../../../data'


def aggregate_one_exp(exp_name: str) -> dict:
    """Aggregate the data from one experiment."""
    exp_info = extract_exp_info(exp_name)
    n_chains = int(exp_info['n_chains'])
    n_samples = int(exp_info['n_samples'])
    res_dict = {k: v for k, v in exp_info.items()}
    # Data
    X_train, Y_train = load_data(exp_info, splittype='train', data_path=DATA_PATH)
    X_val, Y_val = load_data(exp_info, splittype='val', data_path=DATA_PATH)
    val_threshold = min(400, X_val.shape[0])
    X_val = X_val[:val_threshold, :]
    Y_val = Y_val[:val_threshold, :]
    # Baselines
    linear_regr, rf_regr = fit_baselines(X_train, Y_train)
    mse_linear, mse_rf = evaluate_baselines(X_val, Y_val, linear_regr, rf_regr)
    res_dict['rmse_linear'] = np.sqrt(mse_linear)
    res_dict['rmse_rf'] = np.sqrt(mse_rf)
    posterior_samples, posterior_samples_raw = load_samples(exp_name)
    model = load_model(exp_name)
    preds_chain_dim, preds = get_posterior_predictive(
        model, posterior_samples_raw, X_val, exp_info['n_chains']
    )
    # determine the number of good chains (mse < than linear baseline)
    rmse_per_chain = {}
    for i in range(preds_chain_dim.shape[0]):
        rmse_per_chain[f'chain_{i}'] = np.sqrt(mse(preds_chain_dim[i], Y_val)[0])

    rmse_table = pd.DataFrame(rmse_per_chain, index=['RMSE']).T
    bad_chains = rmse_table[rmse_table['RMSE'] > np.sqrt(mse_linear)].index
    bad_chains = bad_chains.str.split('_').str[1].astype(int).values
    bad_chains = bad_chains.tolist()
    good_chains = [i for i in range(n_chains) if i not in bad_chains]
    if len(good_chains) > 0:
        good_chains_pred_indices = np.concatenate(
            [np.arange(n_samples) + (n_samples * i) for i in good_chains]
        )
        good_chains_pred_indices_100 = np.concatenate(
            [np.arange(100) + (n_samples * i) for i in good_chains]
        )
    res_dict['n_bad_chains'] = len(bad_chains)
    res_dict['n_good_chains'] = len(good_chains)

    if len(good_chains) == 0:
        res_dict['rmse_good_chains'] = np.nan
        res_dict['rmse_good_chains_100'] = np.nan
        res_dict['acc_90hpdi'] = np.nan
        res_dict['acc_90hpdi_100'] = np.nan
        return res_dict

    # RMSE
    res_dict['rmse_good_chains'] = np.sqrt(
        mse(preds[good_chains_pred_indices, :], Y_val)[0]
    )
    res_dict['rmse_good_chains_100'] = np.sqrt(
        mse(preds[good_chains_pred_indices_100, :], Y_val)[0]
    )

    # Calibration
    hpdi_preds = hpdi(preds[good_chains_pred_indices], 0.90)
    acc_hpdi = jnp.mean(
        (hpdi_preds[0, :] <= Y_val.squeeze()) & (hpdi_preds[1, :] >= Y_val.squeeze())
    )
    res_dict['acc_90hpdi'] = acc_hpdi
    hpdi_preds_100 = hpdi(preds[good_chains_pred_indices_100], 0.90)
    acc_hpdi_100 = jnp.mean(
        (hpdi_preds_100[0, :] <= Y_val.squeeze())
        & (hpdi_preds_100[1, :] >= Y_val.squeeze())
    )
    res_dict['acc_90hpdi_100'] = acc_hpdi_100

    return res_dict


def aggregate_cross_exp():
    """Aggregate the data from all experiments."""
    # Load the config file
    # config = load_config()
    # load run
    runtimes = load_runtimes()
    # get the experiment names
    exp_names = get_exp_names()
    res = []
    for exp_name in tqdm(exp_names):
        res.append(aggregate_one_exp(exp_name))
    aggregated_data = pd.DataFrame(res)
    # add the runtimes information
    aggregated_data['exp_name'] = exp_names
    aggregated_data['runtime'] = aggregated_data['exp_name'].map(runtimes)
    aggregated_data.to_csv('aggregated_data.csv', index=False)


if __name__ == '__main__':
    aggregate_cross_exp()
