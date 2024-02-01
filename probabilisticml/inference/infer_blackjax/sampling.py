"""Probabilistic learners based on MCMC with NumPyro."""
from typing import Any, Callable

import jax
from jax.random import PRNGKey

from probabilisticml.inference.problearner import ProbLearner


def inference_loop(rng, init_state, step_fn, n_iter):
    """Perform inference using a given step function."""
    ks = jax.random.split(rng, n_iter)

    def one_step(state, key):
        sts, inf = step_fn(key, state)
        return sts, (sts, inf)

    _, (states_step, info_step) = jax.lax.scan(one_step, init_state, ks)
    return states_step, info_step


class BJXMCMCLearner(ProbLearner):
    """Learner producing posterior samples through MCMC with blackJAX."""

    def __init__(
        self,
        sample_key: PRNGKey,
        init_fn: Callable,
        step_fn: Callable,
        init_state: Any,
        num_warmup: int,
        num_samples: int,
    ) -> None:
        """Initialize the learner."""
        self.sample_key = sample_key
        self.init_fn = init_fn
        self.step_fn = step_fn
        self.init_state = init_state
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.posterior_samples = None
        self.inference_performed = False
        self.states = None
        self.info = None

    def perform_inference(self, *args, **kwargs) -> None:
        """Generate (approximate) posterior samples."""
        states, info = inference_loop(
            self.sample_key,
            self.init_fn(self.init_state),
            self.step_fn,
            self.num_warmup + self.num_samples,
        )
        self.states = states
        self.info = info
        self.posterior_samples = states.position
        self.inference_performed = True

    def get_samples(self) -> dict:
        """Get the weight samples."""
        if self.posterior_samples is None:
            raise ValueError(
                'Posterior samples have not been generated yet. '
                'Call `perform_inference()` first.'
            )
        return self.posterior_samples
