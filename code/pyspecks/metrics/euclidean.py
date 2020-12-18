from typing import Optional

from numpy import ndarray
import torch
from torch import Tensor
import pdb
import numpy as np

from pyspecks.metrics import Metric


class EuclideanDistance(Metric):

    def __init__(self, radius: Optional[int] = None, normalize: float = 0., keep_frac: Optional[float] = None):
        if radius is not None:
            assert radius > 0
        self.radius = radius
        self.normalize = normalize
        self.keep_frac = keep_frac

    def compute(self, data: Tensor) -> Tensor:
        if self.radius is not None:
            size_x, size_y = data.shape
            size_xy = size_x * size_y
            base_idx = torch.arange(size_xy, device=data.device)
            base_idx = base_idx.view(size_x, size_y)
            indices1 = []
            indices2 = []
            values = []

            for r in range(1, self.radius + 1):
                for i in range(0, r + 1):
                    j = r - i

                    idx1 = base_idx.narrow(dim=0, start=0, length=size_x-i)
                    idx2 = base_idx.narrow(dim=0, start=i, length=size_x-i)
                    data1 = data.narrow(dim=0, start=0, length=size_x-i)
                    data2 = data.narrow(dim=0, start=i, length=size_x-i)

                    idx1 = idx1.narrow(dim=1, start=0, length=size_y-j)
                    idx2 = idx2.narrow(dim=1, start=j, length=size_y-j)
                    data1 = data1.narrow(dim=1, start=0, length=size_y-j)
                    data2 = data2.narrow(dim=1, start=j, length=size_y-j)

                    indices1.append(idx1.reshape(-1))
                    indices2.append(idx2.reshape(-1))
                    values.append(((data1.reshape(-1) - data2.reshape(-1)) ** 2) + self.normalize * r)

                for i in range(1, r):
                    j = r - i

                    idx1 = base_idx.narrow(dim=0, start=0, length=size_x-i)
                    idx2 = base_idx.narrow(dim=0, start=i, length=size_x-i)
                    data1 = data.narrow(dim=0, start=0, length=size_x-i)
                    data2 = data.narrow(dim=0, start=i, length=size_x-i)

                    idx1 = idx1.narrow(dim=1, start=j, length=size_y-j)
                    idx2 = idx2.narrow(dim=1, start=0, length=size_y-j)
                    data1 = data1.narrow(dim=1, start=j, length=size_y-j)
                    data2 = data2.narrow(dim=1, start=0, length=size_y-j)

                    indices1.append(idx1.reshape(-1))
                    indices2.append(idx2.reshape(-1))
                    values.append(((data1.reshape(-1) - data2.reshape(-1)) ** 2) + self.normalize * r)

                    # # Vertical differences
                    # idx1 = base_idx.narrow(dim=0, start=0, length=size_x-i)
                    # idx2 = base_idx.narrow(dim=0, start=i, length=size_x-i)
                    # val = (data.narrow(dim=0, start=0, length=size_x-i) - data.narrow(dim=0, start=i, length=size_x-i)) ** 2
                    #
                    # # Horizontal differences
                    # idx1 = base_idx.narrow(dim=1, start=0, length=size_y-i)
                    # idx2 = base_idx.narrow(dim=1, start=i, length=size_y-i)
                    # val = (data.narrow(dim=1, start=0, length=size_y-i) - data.narrow(dim=1, start=i, length=size_y-i)) ** 2
                    # indices1.append(idx1.reshape(-1))
                    # indices2.append(idx2.reshape(-1))
                    # values.append(val.reshape(-1))

                # for j in [i, -i]:
                #     for d in [0, 1]:
                #         # idx = torch.roll(base_idx, shifts=j, dims=d)
                #         # val = (data - torch.roll(data, shifts=j, dims=d)) ** 2
                #         indices.append(idx.view(-1))
                #         values.append(val.view(-1))

            indices1 = torch.cat(indices1, dim=0)
            indices2 = torch.cat(indices2, dim=0)
            indices = torch.stack([indices1, indices2], dim=0)
            values = torch.cat(values, dim=0)

            # Remove large distances --> implicitly set to infinity
            if self.keep_frac is not None:
                threshold = np.quantile(values.cpu().numpy(), self.keep_frac)
                mask = (values < threshold)
                values = values[mask]
                indices = indices[:, mask]

            # Append symmetric transpose and diagonal
            indices = torch.cat([indices, torch.flip(indices, dims=(0,)), base_idx.view(-1).expand((2, size_xy))], dim=1)
            values = torch.cat([values, values, torch.zeros_like(data).view(-1)], dim=0)

            # Create sparse tensor
            dist = torch.sparse.FloatTensor(indices, values, (size_xy, size_xy))
        else:
            dist = (data.view(-1, 1) - data.view(1, -1)) ** 2
        return dist
