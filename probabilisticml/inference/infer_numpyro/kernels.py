"""Create NumPyro-like kernels."""
from abc import ABC, abstractmethod
from typing import Any

from jax.random import PRNGKey
from numpyro.infer.mcmc import MCMCKernel


class NumPyroKernel(MCMCKernel, ABC):
    """Base class to enable creation of NumPyro-like kernels."""

    # States returned by `init` and `sample` can be any class that is registered as
    # a pytree. As of 2023/10/11, there is no way to make type hints for pytrees.

    @abstractmethod
    def init(
        self,
        rng_key: PRNGKey,
        num_warmup: int,
        init_params: tuple,
        model_args,
        model_kwargs,
    ) -> Any:  # can be any class that is registered as a pytree
        """Initialize the MCMCKernel and return an initial state."""
        pass

    @abstractmethod
    def sample(self, state: Any, model_args, model_kwargs) -> Any:
        """Given the current state, return the next state."""
        pass
