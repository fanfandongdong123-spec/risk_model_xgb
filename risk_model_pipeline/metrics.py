import numpy as np


def calc_regression_metrics(y_true, y_pred) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    error = pred - y
    total_variance = np.sum((y - y.mean()) ** 2)
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1 - np.sum(error ** 2) / total_variance)
        if len(y) > 1 and total_variance > 0 else np.nan,
    }


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
