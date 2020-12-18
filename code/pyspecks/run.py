from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter

from pyspecks.utils import score_clustering, compute_smoothness


@hydra.main(config_path='../configs', config_name='basic_spectral')
def run(config: DictConfig) -> None:
    writer = SummaryWriter()

    t = time.time()
    image = instantiate(config.image)
    embedder = instantiate(config.embedder)
    cluster_alg = instantiate(config.cluster_alg)
    metric = instantiate(config.metric)

    embedding = embedder(image, metric)
    pred_clusters = cluster_alg(embedding)
    score, pred_clusters, targ_clusters = score_clustering(pred_clusters, image.labels)
    time_elapsed = time.time() - t

    writer.add_scalar('Clustering Score', score)
    writer.add_scalar('Time Elapsed', time_elapsed)
    writer.add_image('Predicted Labels', pred_clusters / pred_clusters.max(), dataformats='HW')
    writer.add_image('True Labels', targ_clusters / targ_clusters.max(), dataformats='HW')
    writer.close()

    pred_smooth = compute_smoothness(pred_clusters)
    targ_smooth = compute_smoothness(targ_clusters)

    print(f"Clustering Score: {score}")
    print(f"Smoothness Score: {pred_smooth} | {targ_smooth}")
    print(f"Running Time: {time_elapsed} seconds")
    print()


if __name__ == "__main__":
    run()
