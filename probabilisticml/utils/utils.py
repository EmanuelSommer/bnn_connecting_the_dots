"""General utility functions."""
from typing import Union

import jax.numpy as jnp
import numpy as np
import numpyro

ArrayType = Union[np.array, jnp.array]


def trace_model(model, rng_seed=1, **args):
    """
    Show the shapes of the values in the trace of a model.

    Traces a given model with the given training data and random seed,
    and prints the shapes of the resulting values.

    Args:
    - model: a callable that takes input data and returns a numpyro distribution
    - rng_seed: an integer random seed
    - args: keyword arguments to pass to the model

    Returns: None
    """
    with numpyro.handlers.seed(rng_seed=rng_seed):
        trace = numpyro.handlers.trace(model).get_trace(**args)
    print(numpyro.util.format_shapes(trace))
