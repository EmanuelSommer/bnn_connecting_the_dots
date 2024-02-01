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


class ShallowGaussianNN(pml.inference.infer_numpyro.likelihood.GaussianModel):
    """
    Shallow (1 Layer) neural Bayesian Regression (Gaussian).

    In particular the priors of the first layer can be specified in different ways:
    1. A single prior distribution for all weights
    2. A list of two prior distributions for the weights results in a Hadamard product
    parametrization of the first layer weights. If group_lasso is True, the a group
    Hadamard product parametrization is used otherwise all weights are overparameterized
    with a Hadamard product of the two priors.
    3. A single float value results in a natural Bayesian lasso prior for the first
    layer weights with a Gamma prior on the dispersion parameter of a Normal
    distribution. The float is the hyperprior of the Gamma distribution.

    Args:
        dim_input (int): The number of input features.
        dim_hidden (int): The number of hidden units.
        precision_prior (Distribution): The prior distribution of the precision
            parameter.
        w1_prior (Union[Distribution, list[Distribution]], float): The prior
            distribution of the weights of the first layer. The different options
            are described above.
        w2_prior (Distribution): The prior distribution of the weights of the
            second layer.
        b1_prior (Optional[Distribution], optional): The prior distribution of the
            bias of the first layer. Defaults to None.
        b2_prior (Optional[Distribution], optional): The prior distribution of the
            bias of the second layer. Defaults to None.
        activation (Optional[str], optional): The activation function. Defaults to
            None (thus the linear activation is used). The options are 'tanh' and
            'relu'.
        target_name (str, optional): The name of the target variable. Defaults to
            'y'.
        group_lasso (bool, optional): Whether to use group lasso regularization.
            Defaults to False.
    """

    def __init__(
        self,
        dim_input: int,
        dim_hidden: int,
        precision_prior: Distribution,
        w1_prior: Union[Distribution, list[Distribution]],
        w2_prior: Distribution,
        b1_prior: Optional[Distribution] = None,
        b2_prior: Optional[Distribution] = None,
        activation: Optional[str] = None,
        target_name: str = 'y',
        group_lasso: bool = False,
    ) -> None:
        """Initialize the model."""
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = 1
        self.w1_prior = w1_prior
        self.hadamard_parametrization = False
        self.natural_bayesian_lasso = False

        if activation == 'tanh':
            self.activation = jnp.tanh
        elif activation == 'silu':
            self.activation = jax.nn.silu
        elif activation == 'relu':
            self.activation = lambda x: x * (x > 0)
        else:
            self.activation = lambda x: x

        if isinstance(w1_prior, list):
            self.hadamard_parametrization = True
            self.u_prior = w1_prior[0]
            self.v_prior = w1_prior[1]
        if isinstance(w1_prior, float):
            self.natural_bayesian_lasso = True
        self.b1_prior = b1_prior
        self.w2_prior = w2_prior
        self.b2_prior = b2_prior
        self.precision_prior = precision_prior
        self.target_name = target_name
        self.group_lasso = group_lasso

    def _get_mu(
        self,
        X: ArrayLike,
        *args,
        **kwargs,
    ) -> jnp.array:
        """Get mean estimate of the Gaussian likelihood."""
        # hidden layer
        if self.hadamard_parametrization:
            hidden = self._hadamard_layer(X)
        elif self.natural_bayesian_lasso:
            hidden = self._natural_bayesian_lasso_layer(X)
        else:
            hidden = self._fully_connected_layer(X)

        # output layer
        W2 = numpyro.sample(
            'W2',
            fn=self.w2_prior,
            sample_shape=(self.dim_hidden, self.dim_output),
        )

        if self.b2_prior is not None:
            b2 = numpyro.sample(
                'b2',
                fn=self.b1_prior,
                sample_shape=(self.dim_output,),
            )
        else:
            b2 = 0

        return hidden @ W2 + b2

    def _get_sigma(self, *args, **kwargs) -> jnp.array:
        """Get variance estimate of the Gaussian likelihood."""
        return numpyro.sample(
            name='sigma', fn=self.precision_prior, sample_shape=(self.dim_output,)
        )

    def _natural_bayesian_lasso_layer(self, X: ArrayLike) -> jnp.array:
        dispersion_parameters = numpyro.sample(
            'lambda',
            fn=numpyro.distributions.Gamma(1, self.w1_prior),
            sample_shape=(self.dim_input,),
        )

        W1 = numpyro.sample(
            'W1',
            fn=Normal(0, dispersion_parameters),
            sample_shape=(self.dim_hidden,),
        )
        W1 = jnp.transpose(W1)

        if self.b1_prior is not None:
            b1 = numpyro.sample(
                'b1',
                fn=self.b1_prior,
                sample_shape=(self.dim_hidden,),
            )
        else:
            b1 = 0

        return self.activation(jnp.dot(X, W1) + b1)

    def _hadamard_layer(self, X: ArrayLike) -> jnp.array:
        V = numpyro.sample(
            'V',
            fn=self.v_prior,
            sample_shape=(self.dim_input, self.dim_hidden),
        )

        if self.b1_prior is not None:
            b1 = numpyro.sample(
                'b1',
                fn=self.b1_prior,
                sample_shape=(self.dim_hidden,),
            )
        else:
            b1 = 0

        if self.group_lasso:
            U = numpyro.sample(
                'U',
                fn=self.u_prior,
                sample_shape=(self.dim_input,),
            )
            return self.activation(jnp.multiply(X, U) @ V + b1)
        else:
            U = numpyro.sample(
                'U',
                fn=self.u_prior,
                sample_shape=(self.dim_input, self.dim_hidden),
            )
            return self.activation(X @ (jnp.multiply(U, V)) + b1)

    def _fully_connected_layer(self, X: ArrayLike) -> jnp.array:
        W1 = numpyro.sample(
            'W1',
            fn=self.w1_prior,
            sample_shape=(self.dim_input, self.dim_hidden),
        )

        if self.b1_prior is not None:
            b1 = numpyro.sample(
                'b1',
                fn=self.b1_prior,
                sample_shape=(self.dim_hidden,),
            )
        else:
            b1 = 0

        return self.activation(X @ W1 + b1)


def shallow_gaussian_regr_from_config(model_config: dict) -> ShallowGaussianNN:
    """
    Create a shallow Gaussian regression model from a configuration dictionary.

    The configuration dictionary should be serializable to JSON and as such the
    distributions
    will not be in there but rather dictionaries with the name, location and scale
    parameters of the distributions. Supported distributions are Normal, Laplace,
    HalfNormal and BernoulliProbs.

    Args:
        model_config (dict): Dictionary with the following entries:
            dim_input (int): The number of input features.
            dim_hidden (int): The number of hidden units.
            precision_prior (Distribution): The prior distribution of the precision
                parameter.
            w1_prior (Union[dict, list[dict]], float): The prior distribution
                of the weights of the first layer. The different options are described
                above.
            w2_prior (dict): The prior distribution of the weights of the
                second layer.
            b1_prior (Optional[dict], optional): The prior distribution of the
                bias of the first layer. Defaults to None.
            b2_prior (Optional[dict], optional): The prior distribution of the
                bias of the second layer. Defaults to None.
            activation (Optional[str], optional): The activation function. Defaults to
                None (thus the linear activation is used). The options are 'tanh' and
                'relu'.
            target_name (str, optional): The name of the target variable. Defaults to
                'y'.
            group_lasso (bool, optional): Whether to use group lasso regularization.
                Defaults to False.

    Returns:
        ShallowGaussianNN: The shallow Gaussian regression model.

    Examples:
        >>> model_config = {
        ...     'dim_input': 2,
        ...     'dim_hidden': 3,
        ...     'precision_prior': {
        ...         'name': 'HalfNormal',
        ...         'scale': 1.0,
        ...     },
        ...     'w1_prior': {
        ...         'name': 'Normal',
        ...         'loc': 0.0,
        ...         'scale': 1.0,
        ...     },
        ...     'w2_prior': {
        ...         'name': 'Normal',
        ...         'loc': 0.0,
        ...         'scale': 1.0,
        ...     },
        ...     'b1_prior': None,
        ...     'b2_prior': None,
        ...     'activation': 'tanh',
        ...     'target_name': 'y',
        ...     'group_lasso': False,
        ... }
        >>> model = shallow_gaussian_regr_from_config(model_config)
    """

    def _get_prior_from_dict(prior_dict: dict) -> Distribution:
        if prior_dict['name'] == 'Normal':
            return Normal(loc=prior_dict['loc'], scale=prior_dict['scale'])
        elif prior_dict['name'] == 'Laplace':
            return Laplace(loc=prior_dict['loc'], scale=prior_dict['scale'])
        elif prior_dict['name'] == 'HalfNormal':
            return HalfNormal(scale=prior_dict['scale'])
        elif prior_dict['name'] == 'BernoulliProbs':
            return BernoulliProbs(probs=prior_dict['probs'])
        else:
            raise ValueError('Prior distribution not supported.')

    model_config = copy.deepcopy(model_config)
    # convert the distributions to numpyro distributions
    for key in model_config.keys():
        if key.endswith('_prior'):
            if isinstance(model_config[key], dict):
                model_config[key] = _get_prior_from_dict(model_config[key])
            elif isinstance(model_config[key], list):
                for i in range(len(model_config[key])):
                    model_config[key][i] = _get_prior_from_dict(model_config[key][i])

    return ShallowGaussianNN(**model_config)


# now a simple Gaussian multi-layer perceptron
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
