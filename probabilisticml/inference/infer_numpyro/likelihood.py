"""Likelihood models for NumPyro."""
from abc import ABC, abstractmethod
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np
import numpyro
from numpy.typing import ArrayLike
from numpyro.distributions import Distribution, Normal


class GaussianModel(ABC):
    """Base class for Bayesian regression models."""

    def __call__(self, X: np.array, Y: Optional[np.array] = None, **kwargs) -> Any:
        """Call the model."""
        mu = self._get_mu(X, **kwargs)
        sigma = self._get_sigma(**kwargs)
        numpyro.sample(name='y', fn=Normal(mu, sigma), obs=Y, **kwargs)

    @abstractmethod
    def _get_mu(self, *args, **kwargs) -> jnp.array:
        """Get mean estimate of the Gaussian likelihood."""
        pass

    @abstractmethod
    def _get_sigma(self, *args, **kwargs) -> jnp.array:
        """Get variance estimate of the Gaussian likelihood."""
        pass


class GaussianLinearModel(GaussianModel):
    """Bayesian linear regression model."""

    def __init__(
        self,
        weight_priors: list[Distribution],
        precision_prior: Distribution,
        target_name: str = 'y',
    ) -> None:
        """Initialize the model."""
        self.weight_priors = weight_priors
        self.precision_prior = precision_prior
        self.target_name = target_name

    def _get_mu(
        self, X: ArrayLike, weights: Optional[np.array] = None, **kwargs
    ) -> jnp.array:
        """Get mean estimate of the Gaussian likelihood."""
        if weights is None:
            p = len(self.weight_priors)
            weights = []
            for j, prior in zip(range(p), self.weight_priors):
                # include **kwargs to allow calling the model with rng_key
                w = numpyro.sample(name=f'w_{j}', fn=prior, **kwargs)
                weights.append(w)
        return X @ jnp.array(weights)  # TODO check if allows for intercept

    def _get_sigma(self, **kwargs) -> jnp.array:
        """Get variance estimate of the Gaussian likelihood."""
        return numpyro.sample(name='sigma', fn=self.precision_prior, **kwargs)
        # TODO check how to implement heteroskedastic noise


class GaussianNN(GaussianModel):
    """Bayesian deep regression model."""

    def __init__(self):
        """Initialize the model."""
        pass

    def _get_mu(self, *args, **kwargs) -> jnp.array:
        """Get mean estimate of the Gaussian likelihood."""
        pass

    def _get_sigma(self, *args, **kwargs) -> jnp.array:
        """Get variance estimate of the Gaussian likelihood."""
        pass
