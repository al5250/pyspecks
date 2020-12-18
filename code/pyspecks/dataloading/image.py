from abc import abstractmethod, ABC

from numpy import ndarray


class Image(ABC):

    @property
    @abstractmethod
    def data(self) -> ndarray:
        pass

    @property
    @abstractmethod
    def labels(self) -> ndarray:
        pass
