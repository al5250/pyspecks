from abc import abstractmethod, ABC

from numpy import ndarray

from pyspecks.dataloading import Image
from pyspecks.metrics import Metric


class Embedder(ABC):

    @abstractmethod
    def embed(self, image: Image, metric: Metric) -> ndarray:
        pass

    def __call__(self, image: Image, metric: Metric) -> ndarray:
        return self.embed(image, metric)
