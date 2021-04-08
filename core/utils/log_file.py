import logging


def log_file(msg, verbose=True):
    if verbose:
        print(msg)
    logging.log(msg)
