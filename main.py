import argparse
import copy
import os
import sys
import time

from models.build import BuildModel
from utils.history import History
from utils.train_utils import train, test, print_info, torch2vector

sys.path.insert(0, os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas as pd
from tqdm import tqdm
from utils.train_utils import file2dict


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument('--model_type', help='choose ml model or dl model, 1--dl model, 0--ml model', type=int, default=1)
    parser.add_argument('--config', help='train config file path', default='config/lenet_config.py')
    parser.add_argument('--auto_search', help='use GridSearchCV to look for best estimator', type=bool, default=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)
    model_config, data_config, optimizer_config = file2dict(args.config)
    dirname = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    save_dir = os.path.join('logs', model_config.get('model_name'), dirname)
    timecost_dir = os.path.join(save_dir, 'time_cost.txt')
    train_history = History(save_dir)
    os.makedirs(save_dir)
    torch.manual_seed(int(data_config.get('random_seed')))
    torch.cuda.manual_seed(int(data_config.get('random_seed')))

    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,), (0.3081,)
        )
    ])

    transform_test = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1325,), (0.3104,)
        )
    ])
    # 读取数据
    train_set = torchvision.datasets.MNIST('./datasets/mnist', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.MNIST('./datasets/mnist', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=data_config.get('batch_size_train'),
                              num_workers=data_config.get('num_workers'), shuffle=True)
    test_loader = DataLoader(test_set, batch_size=data_config.get('batch_size_test'),
                             num_workers=data_config.get('num_workers'), shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 是否使用默认参数
    # 如果不使用，则需要在config文件中设置参数范围，使用网格搜索得到最佳参数
    # 默认使用默认参数
    print(args.auto_search)
    model = BuildModel(copy.deepcopy(model_config), args.model_type, args.auto_search)

    if args.model_type == 1:
        model.to(device)

    print_info(model_config)

    if args.model_type == 1:
        optimizer = eval('optim.' + optimizer_config.pop('type'))(params=model.parameters(), **optimizer_config)
        criterion = eval('nn.' + model_config.get('loss_type'))(
            weight=torch.FloatTensor(model_config.get('loss_weight')).to(device))
        runner = dict(
            model_type=args.model_type,
            optimizer=optimizer,
            loss_type=model_config.get('loss_type'),
            loss_weight=torch.FloatTensor(model_config.get('loss_weight')),
            criterion=criterion,
            train_loader=train_loader,
            test_loader=test_loader,
            iter=0,
            epoch=0,
            max_epochs=data_config.get('epochs'),
            max_iters=data_config.get('epochs') * len(train_loader),
            best_train_loss=float('INF'),
            best_test_acc=float(0),
            best_train_weight='',
            best_train_model='',
            best_test_weight='',
            best_test_model='',
            last_weight='',
            last_model='',
            elapsed=float(0),
        )
        start_time = time.time()
        for epoch in range(data_config.get('epochs')):
            runner['epoch'] = epoch
            epoch_st = time.time()
            train(model, runner, save_dir, train_history, device, epoch,
                  data_config.get('epochs'))
            test(model, runner, data_config.get('test'), save_dir, train_history,
                 device, epoch, data_config.get('epochs'))
            epoch_et = time.time()
            train_history.after_epoch(epoch + 1, epoch_et-epoch_st)
        runner['elapsed'] = (round((time.time() - start_time) / 60, 2))
        print("Training spent {:.4f}m.".format(runner.get('elapsed')))
        with open(timecost_dir, 'w') as f:
            text = 'Training spent: ' + str(runner.get('elapsed')) + 'minutes.'
            f.write(text)

    else:
        X_train, y_train, X_test, y_test = torch2vector(train_set, test_set)
        model = model.model
        runner = dict(
            model_type=args.model_type,
            auto_search=args.auto_search,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            test_acc=float(0),
            train_acc=float(0),
            best_model='',
            training_time=float(0),
        )
        start_time = time.time()
        train(model, runner, save_dir, train_history)
        print("Training spent {:.2f}minutes.".format(runner.get('training_time')))
        with open(timecost_dir, 'w') as f:
            text = 'model fit training spent: ' + str(runner.get('training_time')) + 'minutes.\n'
            f.write(text)
        test(model, runner, data_config.get('test'), save_dir, train_history)
        end_time = time.time()
        print("Task spent {:.4f}s.".format(end_time - start_time))
        task_spent = round((end_time - start_time) / 60, 2)
        with open(timecost_dir, 'a') as f:
            text = 'Task spent: ' + str(task_spent) + 'minutes.'
            f.write(text)
        train_history.after_run()


if __name__ == '__main__':
    main()
