import torch
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from collections import deque


class MetricWriter:

    def __init__(self):
        self._device = torch.device('cpu')
        self._metrics = {
            'train_loss': deque(),
            'train_acc': deque(),
            'train_f1': deque(),
            'val_loss': deque(),
            'val_acc': deque(),
            'val_f1': deque()
        }

    def add_metrics(self, metrics):
        for k, m in metrics.items():
            self._metrics[k].append(m)

    def get_last_epoch_info(self):
        return ''.join([f' {key}: {item[-1]:.5f}' for key, item in self._metrics.items()])

    @property
    def metrics(self):
        return self._metrics

    @property
    def last_val_loss(self):
        return self._metrics['val_loss'][-1]


class MetricWriterKFold:

    def __init__(self, len_train, len_val):
        self._device = torch.device('cpu')
        self._len_train = len_train
        self._len_val = len_val

        # loss
        self._metrics = {
            'train_loss': deque(),
            'train_acc': deque(),
            'train_f1': deque(),
            'val_loss': deque(),
            'val_acc': deque(),
            'val_f1': deque()
        }

        # helpers
        self._train_loss_cur = 0
        self._val_loss_cur = 0
        self._train_pred = deque()
        self._val_pred = deque()
        self._train_target = deque()
        self._val_target = deque()

    def add_loss(self, loss, train=True):
        if train:
            self._train_loss_cur += loss
        else:
            self._val_loss_cur += loss

    def add_target_pred(self, pred, y, train=True, argmax_dim=1):
        pred = torch.argmax(pred, dim=argmax_dim).to(self._device)
        target = y.to(self._device)
        if train:
            self._train_pred.extend(pred)
            self._train_target.extend(target)
        else:
            self._val_pred.extend(pred)
            self._val_target.extend(target)

    def calculate_metrics_fold(self):
        self._metrics['train_loss'].append(self._train_loss_cur / self._len_train)
        self._metrics['val_loss'].append(self._val_loss_cur / self._len_val)
        self._metrics['train_acc'].append(accuracy_score(self._train_target, self._train_pred))
        self._metrics['val_acc'].append(accuracy_score(self._val_target, self._val_pred))
        self._metrics['train_f1'].append(f1_score(self._train_target, self._train_pred, average='weighted'))
        self._metrics['val_f1'].append(f1_score(self._val_target, self._val_pred, average='weighted'))

        self._train_loss_cur = 0
        self._val_loss_cur = 0
        self._train_pred.clear()
        self._val_pred.clear()
        self._train_target.clear()
        self._val_target.clear()

    def get_last_epoch_info(self):
        return ''.join([f' {key}: {item[-1]:.5f}' for key, item in self._metrics.items()])

    @property
    def metrics(self):
        result = {}
        for k, m in self._metrics.items():
            result[k] = np.mean(m)
        return result
