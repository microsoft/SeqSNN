import re
import time
from numba import njit, prange

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
    r2_score,
    precision_recall_curve,
)
from sklearn.metrics import auc as area_under_curve


EPS = 1e-5


def printt(s=None):
    if s is None:
        print()
    else:
        print(str(s), end="\t")


def format_time(t):
    return time.strftime("%m%d%H%M%S", time.localtime(t))


def nan_weighted_avg(vals, weights, axis=None):
    assert vals.shape == weights.shape
    vals = vals.copy()
    weights = weights.copy()
    is_valid = np.logical_and(~np.isnan(vals), ~np.isnan(weights))
    if not np.any(is_valid):
        return np.nan
    weights[~is_valid] = 0
    vals[~is_valid] = 0
    return (vals * weights).sum(axis=axis) / weights.sum(axis=axis)


def z_score_mask(ser, mask):
    ser = ser.copy()
    mean = ser[mask].mean()
    std = ser[mask].std()
    return (ser[mask] - mean) / std


# loss and metric functions


class K:
    """backend kernel"""

    @staticmethod
    def sum(x, axis=0, keepdims=True):
        if isinstance(x, np.ndarray):
            return x.sum(axis=axis, keepdims=keepdims)
        if isinstance(x, torch.Tensor):
            return x.sum(dim=axis, keepdim=keepdims)
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def clip(x, min_val, max_val):
        if isinstance(x, np.ndarray):
            return np.clip(x, min_val, max_val)
        if isinstance(x, torch.Tensor):
            return torch.clamp(x, min_val, max_val)
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def mean(x, axis=0, keepdims=True):
        # print(x.max())
        if isinstance(x, np.ndarray):
            return x.mean(axis=axis, keepdims=keepdims)
        if isinstance(x, torch.Tensor):
            return x.mean(dim=axis, keepdim=keepdims)
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def seq_mean(x, keepdims=True):
        if isinstance(x, torch.Tensor):
            return x.mean()
        if isinstance(x, np.ndarray):
            return x.mean()
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def std(x, axis=0, keepdims=True):
        if isinstance(x, np.ndarray):
            return x.std(axis=axis, keepdims=keepdims)
        if isinstance(x, torch.Tensor):
            return x.std(dim=axis, unbiased=False, keepdim=keepdims)
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def median(x, axis=0, keepdims=True):
        # NOTE: numpy will average when size is even,
        # but tensorflow and pytorch don't average
        if isinstance(x, np.ndarray):
            return np.median(x, axis=axis, keepdims=keepdims)
        if isinstance(x, torch.Tensor):
            return torch.median(x, dim=axis, keepdim=keepdims)[0]
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def shape(x):
        if isinstance(x, np.ndarray):
            return x.shape
        if isinstance(x, torch.Tensor):
            return list(x.shape)
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def cast(x, dtype="float"):
        if isinstance(x, np.ndarray):
            return x.astype(dtype)
        if isinstance(x, torch.Tensor):
            return x.type(getattr(torch, dtype))
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def maximum(x, y):
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            return np.minimum(x, y)
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            return torch.max(x, y)
        if isinstance(x, torch.Tensor):
            return torch.clamp(x, max=y)
        if isinstance(y, torch.Tensor):
            return torch.clamp(y, max=x)
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def auc(y, p):
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(p, torch.Tensor):
            p = p.detach().cpu().numpy()
        if isinstance(y, np.ndarray) and isinstance(p, np.ndarray):
            y, p = y.reshape(-1), p.reshape(-1)
            assert len(y) == len(p), "Shapes of labels and predictions not match."
            return roc_auc_score(y, p)
        if isinstance(y, list) and isinstance(p, list):
            y, p = np.array(y), np.array(p)
            assert len(y) == len(p), "Shapes of labels and predictions not match."
            return roc_auc_score(y, p)
        raise NotImplementedError("unsupported data type %s or %s" % (type(y), type(p)))

    @staticmethod
    def accuracy(y, p):
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(p, torch.Tensor):
            p = p.detach().cpu().numpy()
        if isinstance(y, np.ndarray) and isinstance(p, np.ndarray):
            y = y.reshape(-1)
            p = p.reshape(len(y), -1)
            if p.shape[-1] == 1:
                p = p.squeeze(-1)
            if len(p.shape) == 2:
                p = np.argmax(p, axis=1)
            assert len(y) == len(p), "Shapes of labels and predictions not match."
            return accuracy_score(y, p)
        if isinstance(y, list) or isinstance(p, list):
            assert len(y) == len(p), "Shapes of labels and predictions not match."
            y, p = np.array(y), np.array(p)
            return accuracy(y, p)
        raise NotImplementedError("unsupported data type %s or %s" % (type(y), type(p)))

    @staticmethod
    def bce(y, p, reduce=True):
        assert type(y) == type(p), "Type of label and prediction not match."
        if isinstance(y, torch.Tensor):
            y = y.reshape(-1).to(torch.float)
            p = p.reshape(-1)
            assert len(p) == len(y)
            loss = torch.nn.BCELoss(reduction="mean" if reduce else "none")
            return loss(p, y)
        if isinstance(y, np.ndarray):
            y = y.reshape(-1)
            p = p.reshape(len(y), -1)
            if p.shape[-1] == 1:
                p = p.squeeze(-1)
            return log_loss(y, p)
        raise NotImplementedError("unsupported data type %s or %s" % (type(y), type(p)))

    @staticmethod
    def cross_entropy(y, p, reduce=True):
        assert type(y) == type(p), "Type of label and prediction not match."
        if isinstance(y, torch.Tensor):
            y = y.reshape(-1).to(torch.long)
            p = p.reshape(len(y), -1).squeeze(-1)
            assert len(p) == len(y)
            loss = torch.nn.NLLLoss(reduction="mean" if reduce else "none")
            return loss(p, y)
        if isinstance(y, np.ndarray):
            y = y.reshape(-1)
            p = p.reshape(len(y), -1)
            if p.shape[-1] == 1:
                p = p.squeeze(-1)
            p = np.exp(p)
            return log_loss(y, p)
        raise NotImplementedError("unsupported data type %s or %s" % (type(y), type(p)))

    @staticmethod
    def nll(y, p, reduce=True):
        assert type(y) == type(p), "Type of label and prediction not match."
        if isinstance(y, torch.Tensor):
            loss = torch.nn.NLLLoss(reduction="mean" if reduce else "none")
            return loss(p, y)
        p = np.exp(p)
        p = np.transpose(p, (0, 2, 1))
        p = p.reshape(-1, p.shape[-1])
        y = y.reshape(-1)
        return log_loss(y, p)

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def mauc(y, p):
        aucs = 0
        ratios = 0
        _, m = y.shape
        for t in prange(m):
            auc, ratio = fast_auc(y[:, t], p[:, t])
            aucs += auc
            ratios += ratio
        # print(ratios)
        return aucs / ratios

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def dauc(y, p):
        aucs = 0
        ratios = 0
        n, _ = y.shape
        for i in prange(n):
            auc, ratio = fast_auc(y[i, :], p[i, :])
            aucs += auc
            ratios += ratio
        print(ratios)
        return aucs / ratios

    @staticmethod
    def mauprc(y, p):
        aucs = 0
        ratios = 0
        _, m = y.shape
        for t in prange(m):
            auc, ratio = fast_auprc(y[:, t], p[:, t])
            aucs += auc
            ratios += ratio
        # print(ratios)
        return aucs / ratios

    @staticmethod
    def r2_score(y, p):
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(p, torch.Tensor):
            p = p.detach().cpu().numpy()
        if isinstance(y, np.ndarray) and isinstance(p, np.ndarray):
            y = y.reshape(-1)
            p = p.reshape(len(y), -1)
            if p.shape[-1] == 1:
                p = p.squeeze(-1)
            return r2_score(y, p)
        if isinstance(y, list) and isinstance(p, list):
            y, p = np.array(y), np.array(p)
            return K.r2_score(y, p)
        raise NotImplementedError("unsupported data type %s or %s" % (type(y), type(p)))

    @staticmethod
    def ap(y, p):
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(p, torch.Tensor):
            p = p.detach().cpu().numpy()
        if isinstance(y, np.ndarray) and isinstance(p, np.ndarray):
            y, p = y.reshape(-1), p.reshape(-1)
            assert len(y) == len(p), "Shapes of labels and predictions not match."
            return average_precision_score(y, p)
        if isinstance(y, list) and isinstance(p, list):
            assert len(y) == len(p), "Shapes of labels and predictions not match."
            return average_precision_score(y, p)
        raise NotImplementedError("unsupported data type %s or %s" % (type(y), type(p)))

    @staticmethod
    def auprc(y, p):
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(p, torch.Tensor):
            p = p.detach().cpu().numpy()
        if isinstance(y, np.ndarray) and isinstance(p, np.ndarray):
            y, p = y.reshape(-1), p.reshape(-1)
            assert len(y) == len(p), "Shapes of labels and predictions not match."
            precision, recall, _ = precision_recall_curve(y, p)
            return area_under_curve(recall, precision)
        if isinstance(y, list) and isinstance(p, list):
            assert len(y) == len(p), "Shapes of labels and predictions not match."
            precision, recall, _ = precision_recall_curve(y, p)
            return area_under_curve(recall, precision)
        raise NotImplementedError("unsupported data type %s or %s" % (type(y), type(p)))


# Add Static Methods
def generic_ops(method):
    def wrapper(x, *args):
        if isinstance(x, np.ndarray):
            return getattr(np, method)(x, *args)
        if isinstance(x, torch.Tensor):
            return getattr(torch, method)(x, *args)
        raise NotImplementedError("unsupported data type %s" % type(x))

    return wrapper


for method in [
    "abs",
    "log",
    "sqrt",
    "exp",
    "log1p",
    "tanh",
    "cosh",
    "squeeze",
    "reshape",
    "zeros_like",
]:
    setattr(K, method, staticmethod(generic_ops(method)))

# Functions


def zscore(x, axis=0):
    mean = K.mean(x, axis=axis)
    std = K.std(x, axis=axis)
    return (x - mean) / (std + EPS)


def robust_zscore(x, axis=0):
    med = K.median(x, axis=axis)
    mad = K.median(K.abs(x - med), axis=axis)
    x = (x - med) / (mad * 1.4826 + EPS)
    return K.clip(x, -3, 3)


def batch_corr(x, y, axis=0, keepdims=True):
    x = zscore(x, axis=axis)
    y = zscore(y, axis=axis)
    return (x * y).mean()


def robust_batch_corr(x, y, axis=0, keepdims=True):
    x = robust_zscore(x, axis=axis)
    y = robust_zscore(y, axis=axis)
    return batch_corr(x, y)


@njit
def fast_auc(y_true, y_prob):
    mask = np.logical_not(np.isnan(y_true))
    ratio = np.sum(mask) / mask.size
    y_true = np.extract(mask, y_true)
    y_prob = np.extract(mask, y_prob)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += 1 - y_i
        auc += y_i * nfalse
    auc /= nfalse * (n - nfalse)
    return auc * ratio, ratio


def fast_auprc(y, p):
    mask = np.logical_not(np.isnan(y))
    ratio = np.sum(mask) / mask.size
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    if isinstance(p, torch.Tensor):
        p = p.detach().cpu().numpy()
    if isinstance(y, np.ndarray) and isinstance(p, np.ndarray):
        y, p = y.reshape(-1), p.reshape(-1)
        assert len(y) == len(p), "Shapes of labels and predictions not match."
        return average_precision_score(y, p) * ratio, ratio
    if isinstance(y, list) and isinstance(p, list):
        assert len(y) == len(p), "Shapes of labels and predictions not match."
        return average_precision_score(y, p) * ratio, ratio
    raise NotImplementedError("unsupported data type %s or %s" % (type(y), type(p)))


def auc(y, preds):
    return K.auc(y, preds)


def mauc(y, preds):
    return K.mauc(y, preds)


def dauc(y, preds):
    return K.dauc(y, preds)


def auprc(y, preds):
    return K.auprc(y, preds)


def ap(y, preds):
    return K.ap(y, preds)


def mauprc(y, preds):
    return K.mauprc(y, preds)


def r2(y, preds):
    return K.r2_score(y, preds)


def sequence_mse(y_true, y_pred):
    loss = (y_true - y_pred) ** 2
    # return torch.mean(loss)

    return K.seq_mean(loss, keepdims=False)


def sequence_mae(y_true, y_pred):
    loss = torch.abs(y_true - y_pred)
    # return torch.mean(loss)
    return K.seq_mean(loss, keepdims=False)


def sequence_mase(y_true, y_pred):
    if isinstance(y_true, np.ndarray):
        loss = (y_true - y_pred) ** 2 + np.abs(y_true - y_pred)
    else:
        loss = (y_true - y_pred) ** 2 + (y_true - y_pred).abs()
    return K.seq_mean(loss, keepdims=False)


def single_mase(y_true, y_pred):
    if isinstance(y_true, np.ndarray):
        loss = (y_true - y_pred) ** 2 + np.abs(y_true - y_pred)
    else:
        loss = (y_true - y_pred) ** 2 + (y_true - y_pred).abs()
    return K.mean(loss, keepdims=False)


def single_mae(y_true, y_pred):
    if isinstance(y_true, np.ndarray):
        loss = np.abs(y_true - y_pred)
        return np.nanmean(loss)
    loss = (y_true - y_pred).abs()
    return loss.mean()


def rrse(y_true, y_pred):
    if isinstance(y_true, np.ndarray):
        y_bar = y_true.mean(axis=0)
        loss = np.sqrt(((y_pred - y_true) ** 2).sum()) / np.sqrt(
            ((y_true - y_bar) ** 2).sum()
        )
        return np.nanmean(loss)
    y_bar = y_true.mean(dim=0)
    loss = torch.sqrt(((y_pred - y_true) ** 2).sum()) / torch.sqrt(
        ((y_true - y_bar) ** 2).sum()
    )
    return loss.mean()


def mape(y_true, y_pred, log=False):
    if isinstance(y_true, np.ndarray):
        if log:
            y_true = np.exp(y_true)
            y_pred = np.exp(y_pred)
        loss = np.abs(y_true - y_pred) / y_true
    else:
        if log:
            y_true = y_true.exp()
            y_pred = y_pred.exp()
        loss = (y_true - y_pred).abs()
    return loss.mean()


def mape_log(y_true, y_pred, log=True):
    if isinstance(y_true, np.ndarray):
        if log:
            y_true = np.exp(y_true)
            y_pred = np.exp(y_pred)
        loss = np.abs(y_true - y_pred) / y_true
    else:
        if log:
            y_true = torch.exp(y_true)
            y_pred = torch.exp(y_pred)
        loss = (y_true - y_pred).abs()
    return K.mean(loss, keepdims=False)


def single_mse(y_true, y_pred):
    if isinstance(y_true, np.ndarray):
        loss = (y_true - y_pred) ** 2
        return np.nanmean(loss)
    mask = ~torch.isnan(y_true)
    y_pred = torch.masked_select(y_pred, mask)
    y_true = torch.masked_select(y_true, mask)
    loss = (y_true - y_pred) ** 2
    loss = loss.mean()

    return loss


def bce(y_true, y_pred, reduce=True):
    return K.bce(y_true, y_pred, reduce)


def cross_entropy(y_true, y_pred, reduce=True):
    return K.cross_entropy(y_true, y_pred, reduce)


def accuracy(y_true, y_pred):
    return K.accuracy(y_true, y_pred)


def nll(y_true, y_pred, reduce=True):
    return K.nll(y_true, y_pred, reduce)


def outside_cross_entropy(y_true, y_pred, reduce=True):
    # https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/ops/nn_impl.py#L142
    y = K.cast(y_true > 0, "float")
    p = y_pred
    loss = K.maximum(p, 0) - p * y + K.log1p(K.exp(-K.abs(p)))
    if reduce:
        return K.mean(loss, keepdims=False)
    return loss


def neg_wrapper(func):
    def wrapper(*args, **kwargs):
        return -1 * func(*args, **kwargs)

    return wrapper


def get_loss_fn(loss_fn):
    # reflection: legacy name
    if loss_fn == "mse":
        return single_mse
    if loss_fn == "single_mse":
        return single_mse
    if loss_fn == "outside_bce":
        return outside_cross_entropy
    if loss_fn == "mase":
        return sequence_mase
    if loss_fn == "mae":
        return single_mae
    if loss_fn.startswith("label"):
        return single_mse
    if loss_fn == "cross_entropy":
        return cross_entropy
    # if loss_fn == 'mape_log':
    #     return partial(mape, log=True)

    # return function by name
    try:
        return eval(loss_fn)  # dangerous eval
    except Exception:
        pass
    # return negative function by name
    try:
        return neg_wrapper(eval(re.sub("^neg_", "", loss_fn)))
    except Exception:
        raise NotImplementedError("loss function %s is not implemented" % loss_fn)


def get_metric_fn(eval_metric):
    # reflection: legacy name
    if eval_metric == "corr":
        return neg_wrapper(robust_batch_corr)  # more stable
    if eval_metric == "mse":
        return single_mse
    if eval_metric == "mae":
        return single_mae
    if eval_metric in ["rse", "rrse"]:
        return rrse
    # return function by name
    # if eval_metric == 'mape_log':
    #     return partial(mape, log=True)
    try:
        return eval(eval_metric)  # dangerous eval
    except Exception:
        pass
    # return negative function by name
    try:
        return neg_wrapper(eval(re.sub("^neg_", "", eval_metric)))
    except Exception:
        raise NotImplementedError("metric function %s is not implemented" % eval_metric)


def test():
    pass


if __name__ == "__main__":
    test()
