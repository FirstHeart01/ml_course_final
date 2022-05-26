import numpy as np
from matplotlib import pyplot as plt
from numpy import mean
import os
import csv


class History:
    def __init__(self, dir):
        self.csv_dir = os.path.join(dir, 'metrics_outputs.csv')
        self.pic_dir = os.path.join(dir, 'loss-acc.png')
        self.conf_mat_dir = os.path.join(dir, 'confusion_matrix.png')
        self.roc_curve_dir = os.path.join(dir, 'roc_curve.png')
        self.pr_curve_dir = os.path.join(dir, 'pr_curve.png')
        self.losses_iter = []
        self.losses_epoch = []

        self.acc_iter = []
        self.test_acc_epoch = []
        self.train_acc_epoch = []

        # self.train_acc = float(0)
        # self.test_acc = float(0)
        # self.f1_score = float(0)
        # self.precisions = []
        # self.recalls = []
        # self.thresholds = []
        # self.fpr = []
        # self.tpr = []
        self.confusion_matrix = [[]]

        # self.f1_epoch = []
        # self.recall_epoch = []
        # self.precision_epoch = []
        self.epoch_outputs = [['Epoch', 'Train Loss', 'Train Acc', 'Test Acc', 'Precision', 'Recall', 'F1 Score', 'Time Spent']]
        self.metrics_outputs = [['Train Acc', 'Test Acc', 'Precision', 'Recall', 'F1 Score']]
        self.temp_data = []

    def update_ml(self, data, mode):
        if mode == 'train':
            # self.train_acc = data
            self.temp_data.append(data)
        elif mode == 'test':
            # self.test_acc = data.get('test_acc')
            # self.precisions = data.get('precisions')
            # self.f1_score = data.get('f1_score')
            # self.recalls = data.get('recalls')
            # self.thresholds = data.get('thresholds')
            # self.fpr = data.get('fpr')
            # self.tpr = data.get('tpr')
            self.confusion_matrix = data.get('confusion_matrix')
            self.temp_data.extend([
                data.get('test_acc'),
                data.get('precision'),
                data.get('recall'),
                data.get('f1_score'),
            ])

    def update_dl(self, data, mode):
        if mode == 'train':
            self.temp_data.append(data[0])
            self.temp_data.extend([data[1]])
            self.losses_epoch.append(data[0])
            self.train_acc_epoch.append(data[1] * 100.0)
        elif mode == 'test':
            self.temp_data.extend(
                [data.get('accuracy'), mean(data.get('precision', 0.0)), mean(data.get('recall', 0.0)),
                 mean(data.get('f1_score', 0.0))], )
            self.test_acc_epoch.append(data.get('accuracy'))

    def after_iter(self, loss, acc):
        pass

    def after_epoch(self, epoch, time_spent):
        """
        保存每周期的 'Train Loss', 'Test Acc', 'Precision', 'Recall', 'F1 Score', 'Time Spent'
        """
        with open(self.csv_dir, 'w', newline='') as f:
            writer = csv.writer(f)
            self.temp_data.insert(0, epoch)
            self.temp_data.extend([time_spent])
            self.epoch_outputs.append(self.temp_data)
            self.temp_data = []
            writer.writerows(self.epoch_outputs)

        '''
        绘制每周期Train Loss以及Test Accuracy
        '''
        total_epoch = np.arange(len(self.losses_epoch)) + 1

        fig, ax1 = plt.subplots(dpi=150)
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.plot(total_epoch, self.losses_epoch, 'red', linewidth=2, marker='o', label='Train loss')
        ax1.grid(True)

        ax2 = ax1.twinx()
        # ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Acc')
        ax2.plot(total_epoch, self.test_acc_epoch, 'blue', linewidth=2, marker='D', label='Test acc')

        ax3 = ax2.twiny()
        ax3.plot(total_epoch, self.train_acc_epoch, 'orange', linewidth=2, marker='*', label='Train acc')
        fig.legend(loc='center', bbox_to_anchor=(0.8, 0.55))
        # fig.legend()
        fig.tight_layout()
        plt.savefig(self.pic_dir)
        plt.close("all")

    def after_run(self):
        """
        保存机器学习模型训练后的各类指标 'Train Acc', 'Test Acc', 'Precision', 'Recall', 'F1 Score'
        """
        with open(self.csv_dir, 'w', newline='') as f:
            writer = csv.writer(f)
            self.metrics_outputs.append(self.temp_data)
            writer.writerows(self.metrics_outputs)
        """
        绘制混淆矩阵
        可以绘制多分类的PR曲线、ROC曲线(未实现)
        """

        def plot_confusion_matrix(matrix, conf_mat_dir):
            fig = plt.figure(figsize=(8, 8), dpi=150)
            ax = fig.add_subplot(111)
            cax = ax.matshow(matrix)
            fig.colorbar(cax)
            # plt.show()
            plt.savefig(conf_mat_dir)
            plt.close("all")

        plot_confusion_matrix(self.confusion_matrix, self.conf_mat_dir)

        # def plot_precision_recall(precisions, recalls, pr_curve_dir):
        #     plt.figure((8, 8), dpi=150)
        #     plt.plot(recalls, precisions, 'b-', linewidth=2)
        #     plt.xlabel('Recall', fontsize=16)
        #     plt.title('Precision VS Recall', fontsize=16)
        #     plt.ylabel('Precision', fontsize=16)
        #     plt.axis([0, 1, 0, 1])
        #     plt.savefig(pr_curve_dir)
        #     plt.close("all")
        #
        # plot_precision_recall(self.precisions, self.recalls, self.pr_curve_dir)

        # def plot_roc_curve(fpr, tpr, roc_curve_dir, label=None):
        #     plt.figure((8, 8), dpi=150)
        #     plt.plot(fpr, tpr, linewidth=2, label=label)
        #     plt.plot([0, 1], [0, 1], 'k--')
        #     plt.axis([0, 1, 0, 1])
        #     plt.xlabel('False Positive Rate', fontsize=16)
        #     plt.ylabel('True Positive Rate', fontsize=16)
        #     plt.savefig(roc_curve_dir)
        #
        # plot_roc_curve(self.fpr, self.tpr, self.roc_curve_dir)
