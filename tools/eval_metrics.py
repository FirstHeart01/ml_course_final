from torch.nn.functional import one_hot
from numbers import Number
import numpy as np
import torch

from .accuracy import accuracy


def calculate_confusion_matrix(pred, target):
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    assert (
            isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor)), \
        (f'pred and target should be torch.Tensor or np.ndarray, '
         f'but got {type(pred)} and {type(target)}.')
    num_classes = pred.size(1)
    pred_label = torch.argmax(pred, dim=1).flatten()
    target_label = target.flatten()
    assert len(pred_label) == len(target_label)

    with torch.no_grad():
        indices = num_classes * target_label + pred_label
        matrix = torch.bincount(indices, minlength=num_classes ** 2)
        matrix = matrix.reshape(num_classes, num_classes)
    return matrix.detach().cpu().numpy()


def precision_recall_f1(pred, target, average_mode='macro'):
    allowed_average_mode = ['macro', 'none']
    if average_mode not in allowed_average_mode:
        raise ValueError(f'Unsupport type of averaging {average_mode}.')
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    assert isinstance(pred, torch.Tensor), \
        f'pred should be torch.Tensor or np.ndarray, but got {type(pred)}.'
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).long()
    assert isinstance(target, torch.Tensor), \
        f'target should be torch.Tensor or np.ndarray, ' \
        f'but got {type(target)}.'

    num_classes = pred.size(1)  # size 0为图片数量，size 1 为类别数量
    pred_score, pred_label = torch.topk(pred, k=1)
    pred_score = pred_score.flatten()
    pred_label = pred_label.flatten()

    gt_positive = one_hot(target.flatten(), num_classes)

    pred_positive = one_hot(pred_label, num_classes)
    class_correct = (pred_positive & gt_positive).sum(0).detach().cpu().numpy()
    precision = class_correct / np.maximum(pred_positive.sum(0).detach().cpu().numpy(), 1.) * 100
    recall = class_correct / np.maximum(gt_positive.sum(0).detach().cpu().numpy(), 1.) * 100
    f1_score = 2 * precision * recall / np.maximum(
        precision + recall,
        torch.finfo(torch.float32).eps)
    precisions = []
    recalls = []
    f1_scores = []
    if average_mode == 'macro':
        precision = float(precision.mean())
        recall = float(recall.mean())
        f1_score = float(f1_score.mean())
    elif average_mode == 'none':
        precision = precision
        recall = recall
        f1_score = f1_score
    else:
        raise ValueError(f'Unsupport type of averaging {average_mode}.')
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1_score)

    return precisions, recalls, f1_scores


def evaluate(results, gt_labels, metric='accuracy', metric_options=None):
    if metric_options is None:
        metric_options = {'topk': (1,)}
    if isinstance(metric, str):
        metrics = [metric]
    else:
        metrics = metric
    allowed_metrics = [
        'accuracy', 'precision', 'recall', 'f1_score', 'confusion'
    ]
    eval_results = {}
    num_imgs = len(results)
    assert len(gt_labels) == num_imgs, 'dataset testing results should ' \
                                       'be of the same length as gt_labels.'

    invalid_metrics = set(metrics) - set(allowed_metrics)  # 判断metrics是否存在
    if len(invalid_metrics) != 0:
        raise ValueError(f'metric {invalid_metrics} is not supported.')

    # 不考虑topk和阈值
    topk = metric_options.get('topk')
    thrs = metric_options.get('thrs')
    average_mode = metric_options.get('average_mode', 'macro')

    if 'accuracy' in metrics:
        acc = accuracy(results, gt_labels)
        eval_results_ = {'accuracy': acc}
        eval_results.update(
            {k: v for k, v in eval_results_.items()}
        )

    if 'confusion' in metrics:
        confusion_matrix = calculate_confusion_matrix(results, gt_labels)
        eval_results['confusion'] = confusion_matrix

    precision_recall_f1_keys = ['precision', 'recall', 'f1_score']
    if len(set(metrics) & set(precision_recall_f1_keys)) != 0:
        precision_recall_f1_values = precision_recall_f1(results, gt_labels, average_mode)
        for key, values in zip(precision_recall_f1_keys,
                               precision_recall_f1_values):
            if key in metrics:
                eval_results[key] = values
    return eval_results
