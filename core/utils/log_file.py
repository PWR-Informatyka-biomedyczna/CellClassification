import logging
from datetime import datetime


from core.torchsummary import summary_string, summary


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
filename = f'Training{datetime.now().strftime("%d.%m.%Y-%H.%M.%S")}.txt'
f = open(filename, 'w+')
logging.basicConfig(filename=filename, level=logger.level)


def log_msg(msg, verbose=True):
    if verbose:
        print(msg)
    logger.debug(f'{datetime.now()} : {msg}')


def log_model_summary(model, input_size, verbose=True):
    model_summary = summary_string(model, input_size)
    if verbose:
        summary(model, input_size)
    log_msg(model_summary)
