from abc import abstractmethod, ABC

from torch import Tensor


class Metric(ABC):

    @abstractmethod
    def compute(self, data: Tensor) -> Tensor:
        pass

    def __call__(self, data: Tensor) -> Tensor:
        return self.compute(data)
