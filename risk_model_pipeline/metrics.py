import numpy as np


def calc_ks(y_true, y_pred) -> float:
    y = np.asarray(y_true)
    pred = np.asarray(y_pred)
    order = np.argsort(pred)

    y_sorted = y[order]
    bad = y_sorted.sum()
    good = len(y_sorted) - bad
    if bad == 0 or good == 0:
        return np.nan

    cum_bad_rate = np.cumsum(y_sorted) / bad
    cum_good_rate = np.cumsum(1 - y_sorted) / good
    return float(np.max(np.abs(cum_bad_rate - cum_good_rate)))
