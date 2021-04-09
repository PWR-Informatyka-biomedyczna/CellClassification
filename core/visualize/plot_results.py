from datetime import datetime
import logging


import matplotlib.pyplot as plt
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


def plot_metric(name, val, ax):
    ax.set_title(name)
    ax.plot(range(len(val)), val)
    ax.set_ylim(-0.01, 1.01)


def plot_results(metrics, figsize=(10, 10), save=True):
    n_rows = 2
    n_cols = len(metrics) // n_rows
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    iterator = iter(metrics)
    for i in range(n_rows):
        for j in range(n_cols):
            name = next(iterator)
            val = metrics[name]
            plot_metric(name, val, ax[i, j] if n_cols > 1 else ax[i])
    fig.tight_layout()
    if save:
        fig.savefig(f'Metrics{datetime.now().strftime("%d.%m.%Y-%H.%M.%S")}.png')
    plt.show()
