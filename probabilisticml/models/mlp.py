"""Define simple MLP architectures."""
import flax.linen as nn

from probabilisticml.utils.utils import ArrayType


class SingleLayerRegressionMLP(nn.Module):
    """Simple MLP with one hidden layer."""

    @nn.compact
    def __call__(self, x: ArrayType, n_hidden: int = 16) -> ArrayType:
        """Foo."""
        x = nn.Dense(features=n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x
