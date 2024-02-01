"""Generic data classes."""

from typing import List, Union

import jax.numpy as jnp
import numpy as np


class DatasetTabular:
    """Class for loading and splitting tabular datasets."""

    def __init__(
        self,
        data_path: str,
        target_indices: list = [],
        split_spec: dict = {'train': 0.8, 'val': 0.1, 'test': 0.1},
        seed: int = 0,
        standardize: bool = False,
    ):
        """
        Load and split tabular datasets.

        Args:
        - data_path (str): The path to the dataset file.
        - target_indices (list): A list of indices indicating which columns of
        the dataset correspond to the target variable. Per default, the last
        column is assumed to be the target variable.
        - split_spec (dict): A dictionary specifying the relative or absolute
        sizes of the splits like train, validation, and test.
        - seed (int): The seed for reproducibility.
        - standardize (bool): Whether to standardize the data.
        """
        self.data_path = data_path
        self.seed = seed
        self.target_indices = target_indices
        self.standardize = standardize
        self.split_relative = True if list(split_spec.values())[0] < 1 else False
        self.split_spec = split_spec
        self.data_features, self.data_target = self.load_data()
        if standardize:
            self.data_features = (
                self.data_features - self.data_features.mean(axis=0)
            ) / self.data_features.std(axis=0)
            self.data_target = (
                self.data_target - self.data_target.mean(axis=0)
            ) / self.data_target.std(axis=0)
        self.split_indices = self.split_data()

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data_features)

    def load_data(self):
        """
        Load the dataset from the specified file.

        Returns:
        - data_features (numpy.ndarray): An array containing the features of the
          dataset.
        - data_target (numpy.ndarray): An array containing the target
          variable(s) of the dataset.
        """
        if self.data_path.endswith('.npy'):
            data = np.load(self.data_path)
        elif self.data_path.endswith('.csv'):
            data = np.loadtxt(self.data_path, delimiter=',')
        elif self.data_path.endswith('.data'):
            data = np.genfromtxt(self.data_path, delimiter=' ')
        else:
            raise NotImplementedError(
                'Only .npy and .csv files are supported at this time.'
            )
        if len(self.target_indices) == 0:
            self.target_indices = [data.shape[1] - 1]
        data_features = data[
            :, [i for i in range(data.shape[1]) if i not in self.target_indices]
        ]
        data_target = data[:, self.target_indices]
        return data_features, data_target

    def get_data(
        self, split: Union[List[str], str, None] = None, data_type: str = 'numpy'
    ) -> tuple:
        """
        Return the specified split(s) of the dataset.

        Args:
        - split (list): A list of strings indicating which splits to return.
        Defaults to all splits.
        - data_type (str): The type of data to return. Must be either "numpy" or
          "jax".

        Returns:
        - data_features (numpy.ndarray or jax.numpy.ndarray): An array
        containing the features of the specified split(s) of the dataset.
        - data_target (numpy.ndarray or jax.numpy.ndarray): An array
        containing the target variable(s) of the specified split(s) of the dataset.
        """
        split = split if split is not None else list(self.split_spec.keys())
        split = split if isinstance(split, list) else [split]
        requested_indices = [self.split_indices[split_key] for split_key in split]
        requested_indices = [
            index for sublist in requested_indices for index in sublist
        ]

        if data_type == 'numpy':
            return (
                self.data_features[requested_indices].copy(),
                self.data_target[requested_indices].copy(),
            )
        elif data_type == 'jax':
            return jnp.array(self.data_features[requested_indices].copy()), jnp.array(
                self.data_target[requested_indices].copy()
            )
        else:
            raise NotImplementedError(
                'Only numpy and jax data types are supported at this time.'
            )

    def split_data(self) -> dict:
        """
        Split the dataset into predefined sets like train, validation, and test.

        Returns:
        - split_indices (dict): A dictionary containing the indices (lists) of
        the train, validation, and test sets.
        """
        split_dimensions = self.split_spec.keys()
        split_indices = {}
        # set a random seed for reproducibility
        np.random.seed(self.seed)

        if self.split_relative:
            split_spec = {
                split_key: int(self.split_spec[split_key] * len(self.data_features))
                for split_key in split_dimensions
            }
            split_sum = sum(split_spec.values())
            dataset_len = len(self.data_features)
            if split_sum < dataset_len:
                split_spec['train'] += dataset_len - split_sum
            elif split_sum > dataset_len:
                split_spec['train'] -= split_sum - dataset_len
        else:
            split_spec = self.split_spec
        assert sum(split_spec.values()) == len(
            self.data_features
        ), 'Split spec does not add up to length of data.'

        all_shuffled_indices = np.random.permutation(len(self.data_features))
        split_start = 0
        for split_key in split_spec.keys():
            split_end = split_start + split_spec[split_key]
            split_indices[split_key] = all_shuffled_indices[split_start:split_end]
            split_start = split_end

        return split_indices
