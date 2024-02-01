"""Perform prediction given posterior samples."""

from abc import ABC, abstractmethod
from typing import Any

from jax import tree_flatten, vmap
from jax.random import PRNGKey, split
from numpy.typing import ArrayLike
from numpyro import handlers


class Predictor(ABC):
    """Base class for probabilistic predictions."""

    @abstractmethod
    def predict(self, *args, **kwargs) -> ArrayLike:
        """Generate (approximate) posterior samples."""
        pass


class MCMCPredictor(Predictor):
    """MCMC-based probabilistic predictor."""

    def __init__(self, model: ABC, rng_key: PRNGKey, samples: dict) -> None:
        """Initialize the predictor."""
        self.model = model
        self.rng_key = rng_key
        self.samples = samples

    def predict_fn(self, *args, **kwargs) -> ArrayLike:
        """Predict the output for the test data. (vectorized function)."""

        def single_prediction(rng, samples):
            model_trace = handlers.trace(
                handlers.seed(handlers.condition(self.model, samples), rng)
            ).get_trace(*args, **kwargs)
            return model_trace['y']['value']

        num_samples = tree_flatten(self.samples)[0][0].shape[0]
        rngs = split(self.rng_key, num_samples)
        return vmap(single_prediction)(rngs, self.samples)

    def predict(self, *args, **kwargs) -> ArrayLike:
        """Predict the output for the test data."""
        return self.predict_fn(*args, **kwargs)


class EnsemblePredictor(Predictor):
    """Ensemble-based probabilistic predictor."""

    def __init__(self, model: Any, samples: dict) -> None:
        """Initialize the predictor."""
        self.model = model
        self.samples = samples

    def predict(self, *args, **kwargs) -> ArrayLike:
        """Predict the output for the test data."""
        pass
        # for s in self.samples:
        #     map s to model
        #     predict
