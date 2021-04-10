import torch
from sklearn.metrics import accuracy_score, f1_score


from collections import deque


class MetricWriter:

    def __init__(self, train_data_len, val_data_len):
        self._device = torch.device('cpu')
        self._metrics = {
            'train_loss': deque(),
            'train_acc': deque(),
            'train_f1': deque(),
            'val_loss': deque(),
            'val_acc': deque(),
            'val_f1': deque()
        }
        self._train_data_len = train_data_len
        self._val_data_len = val_data_len

        self._train_loss = 0
        self._val_loss = 0
        self._train_cur_pred = deque()
        self._train_cur_target = deque()
        self._val_cur_pred = deque()
        self._val_cur_target = deque()

    def add_target_pred(self, pred, target, train=True, argmax_dim=1):
        pred = torch.argmax(pred, dim=argmax_dim).to(self._device)
        target = torch.argmax(target, dim=argmax_dim).to(self._device)
        if train:
            self._train_cur_pred.extend(pred)
            self._train_cur_target.extend(target)
        else:
            self._val_cur_pred.extend(pred)
            self._val_cur_target.extend(target)

    def add_loss(self, loss, train=True):
        if train:
            self._train_loss += loss
        else:
            self._val_loss += loss

    def calculate_metrics(self):
        self._metrics['train_loss'].append(self._train_loss / self._train_data_len)
        self._metrics['val_loss'].append(self._val_loss / self._val_data_len)
        self._metrics['train_acc'].append(accuracy_score(self._train_cur_target, self._train_cur_pred))
        self._metrics['val_acc'].append(accuracy_score(self._val_cur_target, self._val_cur_pred))
        self._metrics['train_f1'].append(f1_score(self._train_cur_target, self._train_cur_pred))
        self._metrics['val_f1'].append(f1_score(self._val_cur_target, self._val_cur_pred))
        self._train_loss = 0
        self._val_loss = 0
        self._train_cur_pred.clear()
        self._train_cur_target.clear()
        self._val_cur_pred.clear()
        self._val_cur_target.clear()

    def get_last_epoch_info(self):
        return ''.join([f' {key}: {item[-1]:.5f}' for key, item in self._metrics.items()])

    @property
    def metrics(self):
        return self._metrics

    @property
    def last_val_loss(self):
        return self._metrics['val_loss'][-1]
