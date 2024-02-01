"""Main script for running the experiments with fully connected Gaussian BNNs."""

import copy
import itertools
import os
import sys
import time
from typing import Final, List

import numpy as np
import probabilisticml as pml
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

sys.path.append('../..')


ENSEMBLE_SIZE: Final = 12
NUM_EPOCHS: Final = 5000


# TODO outsource to utils
def load_config(config_path: str) -> tuple[dict, list]:
    """Load the configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def save_config(config, path):
    """Save the configuration file."""
    with open(path, 'w') as file:
        yaml.dump(config, file)


# TODO outsource to utils
def load_data(dataset: str, seed: int) -> tuple:
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
        split_spec={'train': 0.8, 'test': 0.2},
        seed=seed,
        standardize=True,
    )
    X_train, Y_train = regr_dataset.get_data(split='train', data_type='jax')

    return X_train, Y_train


# TODO outsource to utils
def exp_tuple_to_dict(exp: tuple) -> dict:
    """Convert an experiment tuple to a dictionary."""
    return {
        'data': exp[0],
        'activation': exp[1],
        'hidden_structure': [int(d) for d in exp[2].split('-')],
        'replications': exp[3],
    }


# TODO outsource to utils
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
        num_epochs: int,
        x: torch.tensor,
        y: torch.tensor,
        criterion: torch.nn.modules.loss,
        log_at_epoch: list,
    ) -> None:
        """Train the ensemble."""
        if len(self.ckpt) == 0 and len(log_at_epoch) > 0:
            raise ValueError('Logging requires path to checkpoint')

        for idx in range(self.ensemble_size):
            bl = copy.deepcopy(self.base_learner)
            torch.manual_seed(idx)
            # random.seed(idx)
            bl.apply(init_weights)
            opt = optim.Adam(bl.parameters(), weight_decay=0.01)
            with tqdm(total=num_epochs, desc='Training Progress') as pbar:
                for epoch in range(num_epochs):
                    # Forward pass
                    outputs = bl(x)
                    mean_pred = outputs[:, 0]
                    std_pred = torch.exp(outputs[:, 1])
                    loss = criterion(mean_pred, y.squeeze(), std_pred)

                    if (
                        torch.isnan(loss).any()
                        or torch.isinf(loss).any()
                        or loss.item() < -1e6
                    ):
                        print('Loss exploded, breaking')
                        break
                    # Backward pass and optimization
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    pbar.update(1)
                    pbar.set_postfix_str('Loss: {:.4f}'.format(loss.item()))

                # Weights
                weight_keys_in = list(bl.state_dict().keys())
                weight_keys_out = []
                for i in range(len(weight_keys_in) // 2):
                    weight_keys_out.append(f'W{i + 1}')
                    weight_keys_out.append(f'b{i + 1}')
                final_weights = {}
                for i, o in zip(weight_keys_in, weight_keys_out):
                    final_weights[o] = bl.state_dict()[i].data.numpy()
                np.savez(os.path.join(self.ckpt, f'{idx}.npz'), **final_weights)
                torch.save(bl.state_dict(), os.path.join(self.ckpt, f'stdict_{idx}.pt'))

    def predict(self, x: torch.tensor):
        """Predict with the ensemble."""
        ensemble_prediction = []
        for idx in range(self.ensemble_size):
            bl = self.base_learner
            bl.load_state_dict(self.weights[idx])
            prediction = bl(x)
            ensemble_prediction.append(prediction)
        return torch.stack(tuple(ensemble_prediction))


def run_experiment(exp_name: str, exp: tuple, path: str) -> None:
    """Run a single experiment."""
    start_time = time.time()
    # load the data
    X_train, Y_train = load_data(exp[0], seed=exp[3])
    exp_dict = exp_tuple_to_dict(exp)

    if exp_dict['activation'] == 'relu':
        activation = nn.ReLU
    elif exp_dict['activation'] == 'tanh':
        activation = nn.Tanh
    else:
        raise ValueError(f'Activation {exp_dict["activation"]} not supported.')

    # initialize the model
    base_learner = MLP(
        input_size=X_train.shape[1],
        hidden_sizes=[16, 16],
        activation=activation,
        dropout_ratio=0.0,
    )
    deep_ensemble = SGDEnsemble(
        base_learner=base_learner,
        ensemble_size=ENSEMBLE_SIZE,
        ckpt=path,
    )
    deep_ensemble.train(
        num_epochs=NUM_EPOCHS,
        x=torch.from_numpy(np.array(X_train)),
        y=torch.from_numpy(np.array(Y_train)),
        criterion=nn.GaussianNLLLoss(),
        log_at_epoch=[],
    )

    print(f'{(time.time() - start_time) / 60:.2f} min for {exp_name}')


def main():
    """Run all the experiments."""
    # Setup ------------------------------------------------------
    # date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # main_path = f'results/ensembles/{date}/'
    # os.makedirs(main_path, exist_ok=True)
    # config = load_config('experiments/fcn_ensembles/config.yaml')
    # experiments = experiment_generator(config)
    # save_config(config, path=os.path.join(main_path, 'config.yaml'))
    main_path = 'results/de_notune/'
    os.makedirs(main_path, exist_ok=True)
    config = load_config('experiments/fcn_ensembles/config.yaml')
    experiments = experiment_generator(config)

    # Run the experiments
    for exp_name, exp in experiments.items():
        expd = exp_tuple_to_dict(exp)
        exp_identifier = (
            f'{expd["data"]}|{expd["hidden_structure"]}|' + f'{expd["activation"]}|'
        )
        dir_name = os.path.join(main_path, exp_identifier)
        os.makedirs(dir_name, exist_ok=True)
        run_experiment(exp, exp, dir_name)

    print('All experiments have been run.')


if __name__ == '__main__':
    main()
