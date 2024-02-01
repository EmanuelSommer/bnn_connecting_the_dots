"""Base class for generating (approximate) posterior samples."""
from abc import ABC, abstractmethod


class ProbLearner(ABC):
    """Base class for probabilistic learners."""

    @abstractmethod
    def perform_inference(self, *args, **kwargs) -> None:
        """Generate (approximate) posterior samples."""
        pass

    @abstractmethod
    def get_samples(self) -> dict:
        """Get the weight samples."""
        pass
