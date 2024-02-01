"""Main script for running the experiments with fully connected Gaussian BNNs."""

import copy
import itertools
import os
import sys
import time
from typing import (
    Dict,
    Final,
    List,
    Union,
)

import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from pytorch_lightning.callbacks import EarlyStopping

# from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset

# import wandb

sys.path.append('../..')
import probabilisticml as pml

ENSEMBLE_SIZE: Final = 12
MAX_EPOCHS: Final = 5000
WEIGHT_DECAY: Final = [0.01, 0.001, 0.0001]
BATCH_SIZE: Final = [32, 64]
VAL_SIZE: Final = [0.1]  # [0., 0.1]



def load_config(config_path: str) -> tuple[dict, list]:
    """Load the configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def save_config(config, path):
    """Save the configuration file."""
    with open(path, 'w') as file:
        yaml.dump(config, file)



def load_data(dataset: str, seed: int, val_size: float) -> tuple:
    """
    Load the training data.

    Args:
        dataset (str): Name of the dataset to load.
        seed (int): Seed for the random number generator.

    Returns:
        X_train (jax.numpy.ndarray): Input features for training.
        Y_train (jax.numpy.ndarray): Target values for training.
    """
    regr_dataset = pml.data.dataset.DatasetTabular(
        data_path=f'data/{dataset}',
        target_indices=[],
        split_spec={'train': 0.8 - val_size, 'val': val_size, 'test': 0.2},
        seed=seed,
        standardize=True,
    )
    X_train, Y_train = regr_dataset.get_data(split='train', data_type='jax')
    X_val, Y_val = regr_dataset.get_data(split='val', data_type='jax')

    return X_train, Y_train, X_val, Y_val



def exp_tuple_to_dict(exp: tuple) -> dict:
    """Convert an experiment tuple to a dictionary."""
    return {
        'data': exp[0],
        'activation': exp[1],
        'hidden_structure': exp[2],
        'replications': exp[3],
    }



def experiment_generator(config: dict) -> dict:
    """Generate the experiments (Cartesian Product)."""
    exp_dimensions = []
    for key, value in config.items():
        if isinstance(value, list):
            exp_dimensions.append(value)
        elif isinstance(value, dict):
            exp_dimensions.append(list(value.keys()))
        else:
            exp_dimensions.append([value])

    exp_tuples = list(itertools.product(*exp_dimensions))
    experiments = {
        f'exp{str(i)}|' + '|'.join([str(e) for e in p]): p
        for i, p in enumerate(exp_tuples)
    }
    return experiments


def init_weights(layer: nn.Module) -> None:
    """Create checkpoint with network(s) to be loaded in learning."""
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    """Simple MLP network."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        activation: nn.modules.activation,
        dropout_ratio: float,
    ) -> None:
        """Instantiate MLP."""
        super().__init__()
        hidden_id = '_'.join([str(x) for x in hidden_sizes])
        self.model_id = f'MLP_{input_size}_{hidden_id}_2'
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.net = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_sizes[0]))
        for i, o in zip(hidden_sizes, hidden_sizes[1:] + [2]):
            self.net.append(activation())
            self.net.append(torch.nn.Linear(i, o))
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define forward pass."""
        x = self.net(x)
        return self.dropout(x)


class NNLearner(pl.LightningModule):
    """Vanilla network training."""

    def __init__(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        model: nn.Module,
        training_specs: Dict,
        seed: int,
    ) -> None:
        """Set up learner object."""
        super().__init__()
        self.seed = seed
        self.model = model
        self.data_train = TensorDataset(x_train, y_train)
        self.data_val = TensorDataset(x_val, y_val)

        self.optimizer = optim.Adam(
            params=self.model.parameters(), weight_decay=training_specs['weight_decay']
        )
        self.scheduler = None
        self.training_specs = training_specs

    def on_fit_start(self) -> None:
        """Set global seed."""
        pl.seed_everything(seed=self.seed)

    def train_dataloader(self) -> DataLoader:
        """Set up data loader for training."""
        return DataLoader(
            self.data_train,
            batch_size=self.training_specs.get('batch_size'),
            shuffle=True,
            num_workers=1,
        )

    def val_dataloader(self) -> Union[DataLoader, None]:
        """Set up data loader for validation."""
        if len(self.data_val.tensors[0]) == 0:
            return self.train_dataloader()
        else:
            return DataLoader(
                self.data_val,
                batch_size=self.data_val.tensors[0].shape[0],
                num_workers=1,
            )

    def configure_optimizers(self) -> Dict:
        """Set up optimization-related objects."""
        return {'optimizer': self.optimizer}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define standard forward pass."""
        return self.model(x)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Define training routine."""
        x, y = batch
        outputs = self.model(x)
        mean_pred = outputs[:, 0]
        std_pred = torch.exp(outputs[:, 1])
        loss = torch.nn.functional.gaussian_nll_loss(mean_pred, y.squeeze(), std_pred)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        """Define validation routine."""
        if len(self.data_val.tensors[1]) > 0:
            x, y = batch
            outputs = self.model(x)
            mean_pred = outputs[:, 0]
            std_pred = torch.exp(outputs[:, 1])
            loss_val = torch.nn.functional.gaussian_nll_loss(
                mean_pred, y.squeeze(), std_pred
            )
            self.log('loss_val', loss_val)


class SGDEnsemble:
    """Ensemble of SGD trained models."""

    def __init__(
        self, base_learner: nn.Module, ensemble_size: int, ckpt: str = ''
    ) -> None:
        """Instantiate ensemble."""
        self.ensemble_size = ensemble_size
        self.base_learner = base_learner
        self.ckpt = ckpt
        self.weights = []

    def train(
        self,
        x: torch.tensor,
        y: torch.tensor,
        x_val: torch.tensor,
        y_val: torch.tensor,
        log_at_epoch: list,
        training_specs: dict,
    ) -> None:
        """Train the ensemble."""
        if len(self.ckpt) == 0 and len(log_at_epoch) > 0:
            raise ValueError('Logging requires path to checkpoint')

        for idx in range(self.ensemble_size):
            bl = copy.deepcopy(self.base_learner)
            bl.apply(init_weights)
            nn_learner = NNLearner(
                x_train=x,
                y_train=y,
                x_val=x_val,
                y_val=y_val,
                model=bl,
                training_specs=training_specs,
                seed=idx,
            )
            # logger = WandbLogger(name=f'{training_specs["exp_identifier"]}|{idx}')
            if len(y_val) > 0:
                callback = [EarlyStopping('loss_val', patience=30, check_finite=False)]
            else:
                callback = []
            trainer = pl.Trainer(
                max_epochs=training_specs['max_epochs'],
                # logger=logger,
                num_sanity_val_steps=0,
                deterministic=True,
                callbacks=callback,
            )
            print(f'---> Training ensemble member {idx + 1}...')
            trainer.fit(nn_learner)
            # wandb.finish()

            stdict = nn_learner.model.state_dict()
            weight_keys_in = list(stdict.keys())
            weight_keys_out = []
            for i in range(len(weight_keys_in) // 2):
                weight_keys_out.append(f'W{i + 1}')
                weight_keys_out.append(f'b{i + 1}')
            final_weights = {}
            for i, o in zip(weight_keys_in, weight_keys_out):
                final_weights[o] = bl.state_dict()[i].data.numpy()
            np.savez(os.path.join(self.ckpt, f'{idx}.npz'), **final_weights)
            torch.save(stdict, os.path.join(self.ckpt, f'stdict_{idx}.pt'))

    def predict(self, x: torch.tensor):
        """Predict with the ensemble."""
        ensemble_prediction = []
        for idx in range(self.ensemble_size):
            bl = self.base_learner
            bl.load_state_dict(self.weights[idx])
            prediction = bl(x)
            ensemble_prediction.append(prediction)
        return torch.stack(tuple(ensemble_prediction))


def run_experiment(
    exp: tuple, path: str, training_specs: dict, val_size: float
) -> None:
    """Run a single experiment."""
    start_time = time.time()
    # load the data
    X_train, Y_train, X_val, Y_val = load_data(exp[0], seed=exp[3], val_size=val_size)
    exp_dict = exp_tuple_to_dict(exp)
    if training_specs['batch_size'] == -1:
        training_specs['batch_size'] = X_train.shape[0]

    if exp_dict['activation'] == 'relu':
        activation = nn.ReLU
    elif exp_dict['activation'] == 'tanh':
        activation = nn.Tanh
    else:
        raise ValueError(f'Activation {exp_dict["activation"]} not supported.')

    # initialize the model
    base_learner = MLP(
        input_size=X_train.shape[1],
        hidden_sizes=[int(d) for d in exp_dict['hidden_structure'].split('-')],
        activation=activation,
        dropout_ratio=0.0,
    )
    deep_ensemble = SGDEnsemble(
        base_learner=base_learner,
        ensemble_size=ENSEMBLE_SIZE,
        ckpt=path,
    )
    deep_ensemble.train(
        x=torch.from_numpy(np.array(X_train)),
        y=torch.from_numpy(np.array(Y_train)),
        x_val=torch.from_numpy(np.array(X_val)),
        y_val=torch.from_numpy(np.array(Y_val)),
        log_at_epoch=[],
        training_specs=training_specs,
    )

    print(
        f'{(time.time() - start_time) / 60:.2f} min '
        f'for {training_specs["exp_identifier"]}'
    )


def main():
    """Run all the experiments."""
    # Setup ------------------------------------------------------
    main_path = 'results/de/'
    os.makedirs(main_path, exist_ok=True)
    config = load_config('experiments/fcn_ensembles/config.yaml')
    experiments = experiment_generator(config)
    training_configs = itertools.product(WEIGHT_DECAY, BATCH_SIZE, VAL_SIZE)

    # Run the experiments
    for exp_name, exp in experiments.items():
        for wd, bs, vs in training_configs:
            expd = exp_tuple_to_dict(exp)
            isval = 'val' if vs > 0 else 'noval'
            exp_identifier = (
                f'{expd["data"]}|{expd["hidden_structure"]}|{expd["activation"]}|'
                + f'wd{str(wd)}|bs{str(bs)}|{isval}|{expd["replications"]}|'
            )
            training_specs = {
                'max_epochs': MAX_EPOCHS,
                'weight_decay': wd,
                'batch_size': bs,
                'exp_identifier': exp_identifier,
            }
            dir_name = os.path.join(main_path, exp_identifier)
            os.makedirs(dir_name, exist_ok=True)
            run_experiment(exp, dir_name, training_specs, vs)

    print('All experiments have been run.')


if __name__ == '__main__':
    main()
