"""Custom shallow Bayesian neural network model using NumPyro."""

import copy
from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpyro
import sys
sys.path.append('..')
import probabilisticml as pml
from jax import vmap
from numpy.typing import ArrayLike
from numpyro.distributions import (
    BernoulliProbs,
    Distribution,
    HalfNormal,
    Laplace,
    Normal,
)
from numpyro.handlers import substitute, trace

class GaussianMLP:
    """
    Gaussian multi-layer perceptron with fixed priors.

    Args:
        dim_input (int): The number of input features.
        dim_hidden (list[int]): The number of hidden units per layer.
        dim_output (int): The number of output units.
        activation (str): The activation function. Defaults to 'relu'.
        w_prior (Distribution): The prior distribution of the weights.
        b_prior (Distribution): The prior distribution of the biases.
        precision_prior (Distribution): The prior distribution of the precision
            parameter.
        target_name (str, optional): The name of the target variable. Defaults to
            'y'.
    """

    def __call__(self, X: jnp.array, Y: Optional[jnp.array] = None, **kwargs):
        """Call the model."""
        mu_sigma = self._get_mu_sigma(X, **kwargs)
        mu = mu_sigma[:, : self.dim_output]
        sigma = mu_sigma[:, self.dim_output :]
        sigma = jnp.exp(sigma).clip(min=1e-6, max=1e6)
        numpyro.sample(name='y', fn=Normal(mu, sigma), obs=Y, **kwargs)

    def __init__(
        self,
        dim_input: int,
        dim_hidden: list[int],
        dim_output: int,
        activation: Optional[str] = None,
        w_prior: Distribution = Normal(0, 1),
        b_prior: Optional[Distribution] = None,
        precision_prior: Distribution = HalfNormal(1),
        target_name: str = 'y',
        evaluate: bool = False,
    ) -> None:
        """Initialize the model."""
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        if isinstance(dim_hidden, int):
            self.dim_hidden = [dim_hidden]
        self.dim_output = dim_output
        self.activation = activation
        if activation == 'tanh':
            self._activation = jnp.tanh
        elif activation == 'relu':
            self._activation = lambda x: x * (x > 0)
        else:
            self._activation = lambda x: x

        self.w_prior = w_prior
        self.b_prior = b_prior
        self.precision_prior = precision_prior
        self.target_name = target_name
        self.evaluate = evaluate

    def _get_mu_sigma(
        self,
        X: ArrayLike,
        *args,
        **kwargs,
    ) -> jnp.array:
        """Get mean & var estimate of the Gaussian likelihood."""
        # loop over the hidden layers
        hidden = X
        site_counter = 1  # Counter for generating unique site names
        for hidden_neurons in self.dim_hidden:
            W_i = numpyro.sample(
                f'W{site_counter}',
                fn=self.w_prior,
                sample_shape=(hidden.shape[1], hidden_neurons),
            )
            if self.b_prior is not None:
                b = numpyro.sample(
                    f'b{site_counter}',
                    fn=self.b_prior,
                    sample_shape=(hidden_neurons,),
                )
            else:
                b = 0

            hidden = self._activation(hidden @ W_i + b)
            site_counter += 1

        # output layer
        W = numpyro.sample(
            f'W{site_counter}',
            fn=self.w_prior,
            sample_shape=(hidden.shape[1], self.dim_output * 2),
        )
        if self.b_prior is not None:
            b = numpyro.sample(
                f'b{site_counter}',
                fn=self.b_prior,
                sample_shape=(self.dim_output * 2,),
            )
        else:
            b = 0
        output = hidden @ W + b
        if self.evaluate:
            output = numpyro.param('musigma', output)
        return output

    def get_conditional_mu_sigma(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        posterior_samples: dict,
        rolling=False,
    ) -> tuple[jnp.array, jnp.array]:
        """Get the conditional mean and variance of output layer."""
        self.evaluate = True

        def single_prediction(samples):
            """Get a single conditional value of the intermediate state."""
            model_trace = trace(substitute(self._get_mu_sigma, samples)).get_trace(X)
            return model_trace['musigma']['value']

        multi_pred = vmap(single_prediction)(posterior_samples)
        mu = multi_pred[:, :, : self.dim_output]
        sigma = multi_pred[:, :, self.dim_output :]
        sigma = jnp.exp(sigma).clip(min=1e-6, max=1e6)
        self.evaluate = False
        if rolling:
            # rolling average over the first dimension
            max_rolling = 10000
            if mu.shape[0] > max_rolling:
                print(
                    (
                        f'Warning: rolling average over {max_rolling} post. samples.'
                        'Truncating due to potential memory issues.'
                    )
                )
                mu = mu[:max_rolling, ...]
                sigma = sigma[:max_rolling, ...]
            mu = jnp.cumsum(mu, axis=0) / jnp.arange(1, mu.shape[0] + 1).reshape(
                -1, 1, 1
            )
            sigma = jnp.cumsum(sigma, axis=0) / jnp.arange(
                1, sigma.shape[0] + 1
            ).reshape(-1, 1, 1)
        return mu, sigma

    def get_lppd(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        posterior_samples: dict,
        rolling=False,
    ) -> jnp.array:
        """Get the log pointwise predictive density."""
        mu, sigma = self.get_conditional_mu_sigma(X, Y, posterior_samples, rolling)
        return Normal(mu, sigma).log_prob(Y).squeeze()


def gaussian_mlp_from_config(model_config: dict) -> GaussianMLP:
    """
    Create a Gaussian MLP model from a configuration dictionary.

    The configuration dictionary should be serializable to JSON and as such the
    distributions
    will not be in there but rather dictionaries with the name, location and scale
    parameters of the distributions. Supported distributions are Normal, Laplace,
    HalfNormal and BernoulliProbs.

    Args:
        model_config (dict): Dictionary with the following entries:
            dim_input (int): The number of input features.
            dim_hidden (list[int]): The number of hidden units per layer.
            dim_output (int): The number of output units.
            activation (str): The activation function. Defaults to 'relu'.
            w_prior (dict): The prior distribution of the weights.
            b_prior (dict): The prior distribution of the biases.
            precision_prior (dict): The prior distribution of the precision
                parameter.
            target_name (str, optional): The name of the target variable. Defaults to
                'y'.

    Returns:
        GaussianMLP: The Gaussian MLP model.

    Examples:
        >>> model_config = {
        ...     'dim_input': 2,
        ...     'dim_hidden': [3, 4],
        ...     'dim_output': 1,
        ...     'activation': 'relu',
        ...     'w_prior': {
        ...         'name': 'Normal',
        ...         'loc': 0.0,
        ...         'scale': 1.0,
        ...     },
        ...     'b_prior': None,
        ...     'precision_prior': {
        ...         'name': 'HalfNormal',
        ...         'scale': 1.0,
        ...     },
        ...     'target_name': 'y',
        ... }
        >>> model = gaussian_mlp_from_config(model_config)
    """

    def _get_prior_from_dict(prior_dict: dict) -> Distribution:
        if prior_dict['name'] == 'Normal':
            return Normal(loc=prior_dict['loc'], scale=prior_dict['scale'])
        elif prior_dict['name'] == 'Laplace':
            return Laplace(loc=prior_dict['loc'], scale=prior_dict['scale'])
        elif prior_dict['name'] == 'HalfNormal':
            return HalfNormal(scale=prior_dict['scale'])
        else:
            raise ValueError('Prior distribution not supported.')

    model_config = copy.deepcopy(model_config)
    # convert the distributions to numpyro distributions
    for key in model_config.keys():
        if key.endswith('_prior'):
            model_config[key] = _get_prior_from_dict(model_config[key])

    return GaussianMLP(**model_config)
