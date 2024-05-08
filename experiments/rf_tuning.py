"""HPO for Random Forest baselines."""
# %%
import copy

import numpy as np
import optuna # not installed by default within the requirements.txt!
import pandas as pd
import probabilisticml as pml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# %%
def hpo_one_dataset(
    X_train: np.array,
    Y_train: np.array,
    X_test: np.array,
    Y_test: np.array,
    n_trials: int = 100,
) -> dict:
    """Perform HPO for Random Forest on one dataset."""
    X_train = copy.deepcopy(X_train)
    Y_train = copy.deepcopy(Y_train)
    X_test = copy.deepcopy(X_test)
    Y_test = copy.deepcopy(Y_test)
    X_train_sub, X_val, Y_train_sub, Y_val = train_test_split(
        X_train, Y_train, test_size=0.2, random_state=0
    )

    def objective(trial):
        """Objective function for HPO."""
        n_estimators = trial.suggest_int('n_estimators', 10, 500, log=True)
        max_depth = trial.suggest_int('max_depth', 1, 32)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=0,
        )

        model.fit(X_train_sub, Y_train_sub)
        Y_pred = model.predict(X_val)
        mse = mean_squared_error(Y_val, Y_pred)
        rmse = np.sqrt(mse)
        return rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    model = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        random_state=0,
    )
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    return {'best_params': best_params, 'best_rmse': rmse}


# %%

# loop over datasets and replications
replications = [1, 2, 3]
datasets = [
    'airfoil.data',
    'bikesharing.data',
    'concrete.data',
    'energy.data',
    'protein.data',
    'yacht.data',
]
n_trials = {
    'airfoil.data': 100,
    'bikesharing.data': 100,
    'concrete.data': 100,
    'energy.data': 100,
    'protein.data': 100,
    'yacht.data': 100,
}
results = []
for dataset in tqdm(datasets):
    for rep in replications:
        # load data
        regr_dataset = pml.data.dataset.DatasetTabular(
            data_path=f'../data/{dataset}',
            target_indices=[],
            split_spec={'train': 0.8, 'test': 0.2},
            standardize=True,
            seed=rep,
        )
        X_train, Y_train = regr_dataset.get_data(split='train', data_type='numpy')
        X_test, Y_test = regr_dataset.get_data(split='test', data_type='numpy')
        Y_test = Y_test.squeeze()
        Y_train = Y_train.squeeze()
        result = hpo_one_dataset(
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            Y_test=Y_test,
            n_trials=n_trials[dataset],
        )
        print(f"Dataset: {dataset}, Replication: {rep}, RMSE: {result['best_rmse']}")
        result['dataset'] = dataset
        result['replication'] = rep
        results.append(result)

# %%
results_df = pd.DataFrame(results)
# unpack best_params into columns
results_df = pd.concat(
    [
        results_df.drop(['best_params'], axis=1),
        results_df['best_params'].apply(pd.Series),
    ],
    axis=1,
)
# %%
results_df = (
    results_df.groupby('dataset')
    .agg(
        {
            'best_rmse': ['mean', 'std'],
            'n_estimators': ['mean', 'std'],
            'max_depth': ['mean', 'std'],
            'min_samples_split': ['mean', 'std'],
            'min_samples_leaf': ['mean', 'std'],
        }
    )
    .apply(lambda x: np.round(x, 3))
    .reset_index()
)
results_df.columns = [
    'dataset',
    'best_rmse_mean',
    'best_rmse_std',
    'n_estimators_mean',
    'n_estimators_std',
    'max_depth_mean',
    'max_depth_std',
    'min_samples_split_mean',
    'min_samples_split_std',
    'min_samples_leaf_mean',
    'min_samples_leaf_std',
]
# format as described above
results_df['Best RMSE'] = (
    results_df['best_rmse_mean'].astype(str)
    + ' ± '
    + results_df['best_rmse_std'].astype(str)
)
results_df['n_estimators'] = (
    results_df['n_estimators_mean'].astype(str)
    + ' ± '
    + results_df['n_estimators_std'].astype(str)
)
results_df['max_depth'] = (
    results_df['max_depth_mean'].astype(str)
    + ' ± '
    + results_df['max_depth_std'].astype(str)
)
results_df['min_samples_split'] = (
    results_df['min_samples_split_mean'].astype(str)
    + ' ± '
    + results_df['min_samples_split_std'].astype(str)
)
results_df['min_samples_leaf'] = (
    results_df['min_samples_leaf_mean'].astype(str)
    + ' ± '
    + results_df['min_samples_leaf_std'].astype(str)
)
results_df = results_df[
    [
        'dataset',
        'Best RMSE',
        'n_estimators',
        'max_depth',
        'min_samples_split',
        'min_samples_leaf',
    ]
]
# %%
results_df['dataset'] = results_df['dataset'].str.replace('.data', '')
# %%
results_df.to_csv('rf_tuning_results.csv', index=False)
# %%
# markdown to clipboard
print(results_df.to_markdown(index=False))
# %%
