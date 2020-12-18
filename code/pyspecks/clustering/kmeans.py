from numpy import ndarray
from sklearn.cluster import KMeans as SKMeans
from typing import Optional

from pyspecks.clustering.cluster_alg import ClusterAlg


class KMeans(ClusterAlg):

    def __init__(self, k: int, n_init: int, seed: Optional[int] = None):
        self.alg = SKMeans(n_clusters=k, n_init=n_init, random_state=seed)

    def cluster(self, data: ndarray) -> ndarray:
        size_x, size_y, dim = data.shape
        data = data.reshape(size_x * size_y, dim)
        out = self.alg.fit(data)
        labels = out.labels_.reshape(size_x, size_y)
        return labels
