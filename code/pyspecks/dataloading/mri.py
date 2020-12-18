from abc import ABC

from numpy import ndarray
import numpy as np

from pyspecks.dataloading.image import Image


class MRImage(ABC):

    def __init__(self, path: str, axis: str, normalize: bool = True, noise: float = 0.):
        volume, labels = np.load(path)
        dim_z, dim_y, dim_x = volume.shape
        if axis == 'transverse':
            data = volume[dim_z // 2, :, :]
            labels = labels[dim_z // 2, :, :]
        elif axis == 'coronal':
            data = volume[:, dim_y // 2, :]
            labels = labels[:, dim_y // 2, :]
        elif axis == 'sagittal':
            data = volume[:, :, dim_x // 2]
            labels = labels[:, :, dim_x // 2]
        else:
            raise ValueError()
        data = np.flip(data)
        labels = np.flip(labels)
        if normalize:
            data = data / np.max(data)
        data = data + np.random.normal(0, noise, size=data.shape)
        data = data.clip(min=0, max=1)
        self._data = data
        self._labels = labels

    @property
    def data(self) -> ndarray:
        return self._data

    @property
    def labels(self) -> ndarray:
        return self._labels
