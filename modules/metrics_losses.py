import numpy as np


def prob_log_loss(y, p):
    """
    """
    y = np.array(
        [y for i in range(p.shape[0])]
    )
    log_loss = (y*np.log(p) + (1 - y) * np.log(1 - p))
    return -log_loss


def prob_mae(y, y_hat):
    """
    """
    y = np.array(
        [y for i in range(y_hat.shape[0])]
    )
    mae = abs(y - y_hat)
    return mae


def prob_mse(y, y_hat):
    """
    """
    y = np.array(
        [y for i in range(y_hat.shape[0])]
    )
    mse = (y - y_hat)**2
    return mse


def prob_r2(y, y_hat):
    """
    """
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - y_hat.mean(axis=1))**2)
    r2 = 1 - (ss_res / ss_tot)
    return r2
