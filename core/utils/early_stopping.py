import numpy as np


class EarlyStopper:

    def __init__(self, patience, delta=0):
        self._patience = patience
        self._patience_step = 0
        self._delta = delta
        self._best_loss = np.Inf

    def stop(self, loss):
        loss = -loss
        if self._best_loss:
            self._best_loss = loss
        elif loss < self._best_loss + self._delta:
            self._patience_step += 1
            if self._patience >= self._patience:
                return True
        else:
            self._best_loss = loss
            self._patience_step = 0
        return False
