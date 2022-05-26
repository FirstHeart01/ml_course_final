from numbers import Number
from torch.nn.functional import one_hot
import numpy as np
import torch
import torch.nn as nn


def accuracy_torch(pred, target):
    num = pred.size(0)  # 图片个数，即batch
    _, pred_label = torch.max(pred, 1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.expand_as(pred_label))
    correct = correct.reshape(-1).float().sum(0, keepdims=True).item()
    res = (correct * 100 / num)
    # pred_score, pred_label = pred.topk(1, dim=1)
    # pred_label = pred_label.t()
    # correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
    # correct = correct[:1].reshape(-1).float().sum(0, keepdims=True)
    # res.append(correct.mul_(100./num))
    return res


def accuracy(pred, target):
    assert isinstance(pred, (torch.Tensor, np.ndarray)), \
        f'The pred should be torch.Tensor or np.ndarray ' \
        f'instead of {type(pred)}.'
    assert isinstance(target, (torch.Tensor, np.ndarray)), \
        f'The target should be torch.Tensor or np.ndarray ' \
        f'instead of {type(target)}.'

    # torch version is faster in most situations.
    to_tensor = (lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x)
    pred = to_tensor(pred)
    target = to_tensor(target)

    res = accuracy_torch(pred, target)

    return res
