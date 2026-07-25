
import numpy as np
from scipy import stats


def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, eps=1e-8):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def max_ae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.max(np.abs(y_true - y_pred)))


def directional_accuracy(y_true, y_pred, y_last_obs):
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    y_a = np.asarray(y_last_obs)
    actual_direction = np.sign(y_t - y_a)
    pred_direction = np.sign(y_p - y_a)
    return float(np.mean(actual_direction == pred_direction) * 100)


def compute_all_metrics(y_true, y_pred, y_last_obs_list=None):
    assert len(y_true) == len(y_pred), \
        f"Length mismatch: {len(y_true)} vs {len(y_pred)}"

    result = {
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
        'MaxAE': max_ae(y_true, y_pred),
        'N': len(y_true),
    }

    if y_last_obs_list is not None:
        assert len(y_true) == len(y_last_obs_list), \
            f"y_last_obs length mismatch: {len(y_true)} vs {len(y_last_obs_list)}"
        result['DA'] = directional_accuracy(y_true, y_pred, y_last_obs_list)

    return result


def wilcoxon_test(errors_a, errors_b, model_a="A", model_b="B"):
    errors_a, errors_b = np.asarray(errors_a), np.asarray(errors_b)
    assert len(errors_a) == len(errors_b), "Both models need the same number of test samples"
    stat, p_value = stats.wilcoxon(errors_a, errors_b)
    winner = model_a if np.mean(errors_a) < np.mean(errors_b) else model_b
    return {
        'statistic': float(stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'better_model': winner,
    }
