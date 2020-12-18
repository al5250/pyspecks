from abc import abstractmethod, ABC

from numpy import ndarray


class ClusterAlg(ABC):

    @abstractmethod
    def cluster(self, data: ndarray) -> ndarray:
        pass

    def __call__(self, data: ndarray) -> ndarray:
        return self.cluster(data)
