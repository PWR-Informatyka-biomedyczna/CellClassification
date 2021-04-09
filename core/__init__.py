import os


if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')
if not os.path.exists('./metrics'):
    os.mkdir('./metrics')
if not os.path.exists('./logs'):
    os.mkdir('./logs')


from core.dataset.dataset import Data
from core.train_model import train
from core.visualize import plot_results
