import os
import time

import joblib
import numpy as np
import torch
import sys
import types
import importlib

from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from tqdm import tqdm
from numpy import mean
from terminaltables import AsciiTable

from tools.accuracy import accuracy
from tools.eval_metrics import evaluate
import torch.nn as nn

"""
读取配置文件
"""


def file2dict(filename):
    (path, file) = os.path.split(filename)
    abspath = os.path.abspath(os.path.expanduser(path))
    sys.path.insert(0, abspath)
    mod = importlib.import_module(file.split('.')[0])
    sys.path.pop(0)
    cfg_dict = {
        name: value
        for name, value in mod.__dict__.items()
        if not name.startswith('__')
           and not isinstance(value, types.ModuleType)
           and not isinstance(value, types.FunctionType)
    }
    cfg_dict = cfg_dict.get('config')
    return cfg_dict.get('model_config'), cfg_dict.get('data_config'), cfg_dict.get('optimizer_config')


"""
打印模型信息
"""


def print_info(config):
    model = config.get('model_name')
    loss = config.get('loss_type') if 'loss_type' in config else 'None'
    parameters = config.get('parameters') if 'parameters' in config else 'None'

    TITLE = 'Model info'
    TABLE_DATA = (
        ('Model_Name', 'Loss', 'parameters'),
        (model, loss, parameters))
    table_instance = AsciiTable(TABLE_DATA, TITLE)
    print()
    print(table_instance.table)
    print()


"""
获取学习率
"""


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


"""
机器学习模型获取输入
"""


def torch2vector(train_set, test_set):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for sample in train_set:
        x_sample, y_sample = sample
        X_train.append(x_sample.reshape(1, -1).squeeze().numpy())
        y_train.append(y_sample)
    for sample in test_set:
        x_sample, y_sample = sample
        X_test.append(x_sample.reshape(1, -1).squeeze().numpy())
        y_test.append(y_sample)
    return X_train, y_train, X_test, y_test


"""
深度学习模型的训练
"""


def train_dl(model, runner, save_dir, train_history, device, epoch, epoches):
    train_loss = 0
    train_acc = 0
    model.train()
    with tqdm(total=len(runner.get('train_loader')), desc=f'Train: Epoch {epoch + 1}/{epoches}', postfix=dict,
              mininterval=0.3) as pbar:
        for iter, batch in enumerate(runner.get('train_loader')):
            runner['iter'] += 1
            images, targets = batch
            with torch.no_grad():
                images = images.to(device)
                targets = targets.to(device)

            runner.get('optimizer').zero_grad()
            outputs = model.extract_feats(images)
            criterion = runner.get('criterion')
            loss = criterion(outputs, targets)
            loss.backward()
            # losses = model(images, targets=targets, return_loss=True)
            # losses.get('loss').backward()
            runner.get('optimizer').step()

            train_loss += loss.item()
            _, pred_label = torch.max(outputs, 1)
            pred_label = pred_label.t()
            correct = pred_label.eq(targets.expand_as(pred_label))
            correct = correct.reshape(-1).float().sum(0, keepdims=True).item()
            acc = correct / outputs.size(0)
            train_acc += acc
            pbar.set_postfix(**{'Train Acc': train_acc / (iter + 1),
                                'Loss': train_loss / (iter + 1),
                                'Lr': get_lr(runner.get('optimizer'))
                                })
            pbar.update(1)

    train_history.update_dl([train_loss / (iter + 1), train_acc / (iter + 1)], 'train')

    if train_loss / len(runner.get('train_loader')) < runner.get('best_train_loss'):
        runner['best_train_loss'] = train_loss / len(runner.get('train_loader'))
        if epoch > 0:
            os.remove(runner['best_train_weight'])
            os.remove(runner['best_train_model'])
        runner['best_train_weight'] = os.path.join(save_dir, 'Train_Epoch{:03}-Loss{:.3f}.pth'.format(epoch + 1,
                                                                                                      train_loss / len(
                                                                                                          runner.get(
                                                                                                              'train_loader'))))
        runner['best_train_model'] = os.path.join(save_dir, 'Train_Epoch{:03}-Acc{:.3f}.pth'.format(epoch + 1,
                                                                                                    train_acc / len(
                                                                                                        runner.get(
                                                                                                            'train_loader'))))
        torch.save(model.state_dict(), runner.get('best_train_weight'))
        torch.save(model, runner.get('best_train_model'))

    if epoch > 0:
        os.remove(runner['last_weight'])
        os.remove(runner['last_model'])
    runner['last_weight'] = os.path.join(save_dir, 'Last_Epoch{:03}.pth'.format(epoch + 1))
    runner['last_model'] = os.path.join(save_dir, 'Last_Epoch{:03}_model.pth'.format(epoch + 1))
    torch.save(model.state_dict(), runner.get('last_weight'))
    torch.save(model, runner.get('last_model'))


def train_ml(model, runner, save_dir, train_history):
    auto_search = runner.get('auto_search')
    X_train, y_train = runner.get('X_train'), runner.get('y_train')
    X_train, y_train = np.array(X_train), np.array(y_train)
    start_time = time.time()
    model.fit(X_train, y_train)
    if auto_search:
        print(
            "The best parameters are %s with a score of %0.4f" %
            (model.best_params_, model.best_score_)
        )
        model = model.best_estimator_
    train_scores = cross_val_score(model, X_train, y_train, cv=10, n_jobs=-1, scoring='accuracy')
    train_acc = train_scores.mean()
    runner['train_acc'] = train_acc
    train_history.update_ml(train_acc, 'train')
    end_time = time.time()
    runner['training_time'] = round((end_time - start_time) / 60, 2)


def train(model, runner, save_dir, train_history, device='cpu', epoch=0, epoches=100):
    if runner.get('model_type') == 1:
        train_dl(model, runner, save_dir, train_history, device, epoch, epoches)
    else:
        train_ml(model, runner, save_dir, train_history)


"""
深度学习模型的测试
"""


def test_dl(model, runner, config, save_dir, train_history, device, epoch, epoches):
    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(runner.get('test_loader')), desc=f'Test : Epoch {epoch + 1}/{epoches}',
                  mininterval=0.3) as pbar:
            for iter, batch in enumerate(runner.get('test_loader')):
                images, target = batch
                # runner.get('optimizer').zero_grad()
                outputs = model.extract_feats(images.to(device))
                preds.append(outputs)
                targets.append(target.to(device))
                pbar.update(1)

    eval_results = evaluate(torch.cat(preds), torch.cat(targets), config.get('metrics'), config.get('metric_options'))

    train_history.update_dl(eval_results, 'test')

    TITLE = 'Test Results'
    TABLE_DATA = (
        ('Top-1 Acc', 'Mean Precision', 'Mean Recall', 'Mean F1 Score'),
        ('{:.2f}'.format(eval_results.get('accuracy', 0.0)),
         '{:.2f}'.format(mean(eval_results.get('precision', 0.0))),
         '{:.2f}'.format(mean(eval_results.get('recall', 0.0))),
         '{:.2f}'.format(mean(eval_results.get('f1_score', 0.0)))),

    )
    table_instance = AsciiTable(TABLE_DATA, TITLE)
    print()
    print(table_instance.table)
    print()

    if save_dir and eval_results.get('accuracy') > runner.get('best_test_acc'):
        runner['best_test_acc'] = eval_results.get('accuracy')
        if epoch > 0:
            os.remove(runner['best_test_weight'])
            os.remove(runner['best_test_model'])
        runner['best_test_weight'] = os.path.join(save_dir, 'Test_Epoch{:03}-Acc{:.3f}.pth'.format(epoch + 1,
                                                                                                   eval_results.get(
                                                                                                       'accuracy')))
        runner['best_test_model'] = os.path.join(save_dir, 'Test_Epoch{:03}-Acc{:.3f}_model.pth'.format(epoch + 1,
                                                                                                        eval_results.get(
                                                                                                            'accuracy')))
        torch.save(model.state_dict(), runner.get('best_test_weight'))
        torch.save(model, runner.get('best_test_model'))


def test_ml(model, runner, config, save_dir, train_history):
    eval_results = {}
    average_mode = config.get('metric_options').get('average_mode', 'macro')
    X_test, y_test = runner.get('X_test'), runner.get('y_test')
    X_test, y_test = np.array(X_test), np.array(y_test)
    test_pred = cross_val_predict(model, X_test, y_test, cv=10, n_jobs=-1)
    test_scores = cross_val_score(model, X_test, y_test, cv=10, n_jobs=-1, scoring='accuracy')
    test_acc = test_scores.mean()
    # precisions, recalls, thresholds = precision_recall_curve(y_test, test_pred)
    # fpr, tpr, _ = roc_curve(y_test, test_pred)
    conf_matrix = confusion_matrix(y_test, test_pred)
    eval_results['test_acc'] = test_acc
    eval_results['precision'] = precision_score(y_test, test_pred, average=average_mode)
    eval_results['recall'] = recall_score(y_test, test_pred, average=average_mode)
    # eval_results['threshold'] = thresholds
    # eval_results['fpr'] = fpr
    # eval_results['tpr'] = tpr
    eval_results['confusion_matrix'] = conf_matrix
    eval_results['f1_score'] = f1_score(y_test, test_pred, average=average_mode)

    train_history.update_ml(eval_results, 'test')
    TITLE = 'Test Results'
    TABLE_DATA = (
        ('Test Acc', 'Precision', 'Recall', 'F1 Score'),
        ('{:.2f}'.format(eval_results.get('test_acc', 0.0)),
         '{:.2f}'.format(eval_results.get('precision', 0.0)),
         '{:.2f}'.format(eval_results.get('recall', 0.0)),
         '{:.2f}'.format(eval_results.get('f1_score', 0.0))),
    )
    table_instance = AsciiTable(TABLE_DATA, TITLE)
    print()
    print(table_instance.table)
    print()

    runner['best_model'] = os.path.join(save_dir, 'Trained_Model-TestAcc{:.3}.model'.format(test_acc))
    joblib.dump(model, runner.get('best_model'))


def test(model, runner, config, save_dir, train_history, device='cpu', epoch=0, epoches=100):
    if runner.get('model_type') == 1:
        test_dl(model, runner, config, save_dir, train_history, device, epoch, epoches)
    else:
        test_ml(model, runner, config, save_dir, train_history)
