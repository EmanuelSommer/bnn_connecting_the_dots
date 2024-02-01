"""Probabilistic learners based on MCMC with NumPyro."""
import time
from typing import Any, Optional

import jax.numpy as jnp
from jax.random import PRNGKey
from numpyro.infer import MCMC

from probabilisticml.inference.infer_numpyro.kernels import NumPyroKernel
from probabilisticml.inference.problearner import ProbLearner
from probabilisticml.utils.configclasses import ConfigNumPyroSampler


class NumPyroMCMCLearner(ProbLearner):
    """Learner producing posterior samples through MCMC with NumPyro."""

    def __init__(
        self,
        kernel: NumPyroKernel,
        model: Any,
        mcmc_args: ConfigNumPyroSampler,
        mcmc: Optional[MCMC] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the learner."""
        self.model = model
        self.verbose = verbose
        self.inference_performed = False
        self.kernel = kernel
        self.mcmc = (
            MCMC(self.kernel, **mcmc_args.parse_dict()) if mcmc is None else mcmc
        )
        self.posterior_samples = None
        self.warmup_samples = None

    def perform_inference(
        self,
        rng_key: PRNGKey,
        init_params: Any = None,
        collect_warmup: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Perform Bayesian/MAP inference."""
        start = time.time()
        if collect_warmup:
            self.mcmc.warmup(
                rng_key,
                init_params=init_params,
                collect_warmup=collect_warmup,
                *args,
                **kwargs,
            )
            self.warmup_samples = self.mcmc.get_samples()
            self.mcmc.run(rng_key, *args, **kwargs)
        else:
            self.mcmc.run(rng_key, init_params=init_params, *args, **kwargs)
        if self.verbose:
            self.mcmc.print_summary()
            print(f'\nMCMC elapsed time: {time.time() - start}')
        self.posterior_samples = self.mcmc.get_samples()
        self.inference_performed = True

    def get_samples(self, with_warmup: bool = False) -> dict:
        """Get the weight samples."""
        if self.posterior_samples is None:
            raise ValueError(
                'Posterior samples have not been generated yet. '
                'Call `perform_inference()` first.'
            )
        if not with_warmup:
            return self.posterior_samples

        full_post_samples = {}
        for key in self.posterior_samples.keys():
            full_post_samples[key] = jnp.concatenate(
                [self.warmup_samples[key], self.posterior_samples[key]], axis=0
            )

        return full_post_samples
