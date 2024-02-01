"""Define config classes for different objects."""
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class ConfigNumPyroSampler:
    """Class for handling arguments to NumPyro MCMC."""

    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 1
    thinning: int = 1
    postprocess_fn: Optional[Callable] = None
    progress_bar: bool = True
    jit_model_args: bool = False

    def parse_dict(self) -> dict:
        """Parse class attributes to a dictionary."""
        return {
            'num_warmup': self.num_warmup,
            'num_samples': self.num_samples,
            'num_chains': self.num_chains,
            'thinning': self.thinning,
            'postprocess_fn': self.postprocess_fn,
            'progress_bar': self.progress_bar,
            'jit_model_args': self.jit_model_args,
        }
