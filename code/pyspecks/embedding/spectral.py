from typing import Optional

from numpy import ndarray
import torch
import pdb

from pyspecks.dataloading import Image
from pyspecks.metrics import Metric
from pyspecks.embedding import Embedder
from pyspecks.utils import sparse_dim_multiply, sparse_add_identity


class SpectralEmbedder(Embedder):

    def __init__(
        self,
        k: int,
        gamma: float = 1.,
        use_random_walk: bool = False,
        device: Optional[str] = None,
        seed: Optional[int] = None
    ):
        self.k = k
        self.gamma = gamma
        self.use_random_walk = use_random_walk
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        if seed is not None:
            torch.manual_seed(seed)

    @torch.no_grad()
    def embed(self, image: Image, metric: Metric) -> ndarray:
        data = torch.tensor(image.data, device=self.device, dtype=torch.float32)
        size_x, size_y = data.size()
        M = metric(data)
        M /= -self.gamma
        if M.is_sparse:
            M._values().exp_()
            print(f'Created adjacency matrix of size {M.size()}.')

            deg = torch.sparse.sum(M, dim=0).to_dense()
            N = M

            if self.use_random_walk:
                deg_inv = 1 / deg
                N = sparse_dim_multiply(N, deg_inv, dim=1)
                N *= -1
                N = sparse_add_identity(N)
                print(f'Created random walk Laplacian matrix of size {N.size()}.')

            else:
                deg_neg_half = deg ** (-0.5)

                N = sparse_dim_multiply(N, deg_neg_half, dim=0)
                N = sparse_dim_multiply(N, deg_neg_half, dim=1)
                N *= -1
                N = sparse_add_identity(N)
                print(f'Created normalized Laplacian matrix of size {N.size()}.')

            print(f'Num non-zero: {len(N._values())}.')

        else:
            M.exp_()
            print(f'Created adjacency matrix of size {M.size()}.')

            deg = torch.sum(M, dim=0)
            N = M

            if self.use_random_walk:
                deg_inv = 1 / deg
                N *= deg_inv.view(-1, 1)
                N *= -1
                idx = torch.arange(size_x * size_y)
                N[idx, idx] += 1
                print(f'Created random walk Laplacian matrix of size {N.size()}.')
            else:
                deg_neg_half = deg ** (-0.5)
                N *= deg_neg_half
                N *= deg_neg_half.view(-1, 1)
                N *= -1
                idx = torch.arange(size_x * size_y)
                N[idx, idx] += 1
                print(f'Created normalized Laplacian matrix of size {N.size()}.')

        # Eigendecomposition with top k vectors
        evals, evecs = torch.lobpcg(N, largest=False, k=self.k)
        evecs = evecs.cpu().numpy()
        evecs = evecs.reshape(size_x, size_y, self.k)

        print(f'Finished eigendecomposition with feature matrix of size {evecs.shape}.')

        # eigen_gap = evals[-1] - evals[-2]
        # print(f'Eigengap: {eigen_gap}.')

        return evecs
