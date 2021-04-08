import matplotlib.pyplot as plt


def plot_results(metrics, figsize=(10, 10), save=True):
    n_rows = 2
    n_cols = len(metrics) // n_rows
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    for n, label, val in enumerate(metrics.items()):
        i = n * n_rows

