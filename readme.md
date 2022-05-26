# ml_course_final

#### 介绍

四川大学机器学习引论期末作业

#### 软件架构

root:ml_course_final

+--config[模型配置文件夹]

|&emsp;&emsp;&emsp;                     +--alexnet_config.py

|&emsp;&emsp;&emsp;                     +--knn_config.py

|&emsp;&emsp;&emsp;			+--lenet_config.py

|&emsp;&emsp;&emsp;			+--svm_config.py

+--datasets[数据集位置]

+--logs[模型各指标输出记录文件]

+--models[模型建立文件夹]

|&emsp;&emsp;&emsp;			+--\__init__.py

|&emsp;&emsp;&emsp;			+--alexnet.py

|&emsp;&emsp;&emsp;			+--build.py

|&emsp;&emsp;&emsp;			+--lenet.py

+--tools[指标计算可视化文件夹]

|&emsp;&emsp;&emsp;			+--accuracy.py

|&emsp;&emsp;&emsp;			+--eval_metrics.py

|&emsp;&emsp;&emsp;			+--netviz.py

+--utils[训练工具文件夹]

|&emsp;&emsp;&emsp;			+--dataloader.py

|&emsp;&emsp;&emsp;			+--history.py

|&emsp;&emsp;&emsp;			+--train_utils.py

+--main.py[项目入口]

+--requirements.txt[依赖库]

+--test_prog.py[测试程序]



#### 使用说明

1.  python main.py [-h] [--model_type MODEL_TYPE] [--config CONFIG] [--auto_search AUTO_SEARCH]
2.  python main.py 默认使用深度学习模型LeNet进行训练
3.  使用机器学习模型时，最好加上--auto_search进行自动寻参

### 文件说明

1. config文件夹：

   配置模型参数。其中有model_config，data_config，optimizer_config。

   model_config中的参数有：

   - 自己建立的模型名称
   - 损失函数类型和参数
   - 机器学习模型超参数列表

   data_config中的参数有：

   - 随机种子
   - batch_size
   - epochs
   - num_workers
   - 测试时的指标参数

   optimizer_config中的参数有：

   - 优化器类型名称
   - 学习率
   - 冲量
   - l2正则化参数

2. models文件夹

   建立自己的深度学习模型类，并加入到\__init__.py中。之后需要在config文件夹中添加相应的配置文件，包括模型配置参数、数据配置参数、优化器配置参数等。

3. tools文件夹和utils文件夹

   这些是训练过程或者测试的时候用到的一些工具类，包括计算准确率、F1分数、混淆矩阵、召回率等等。

## 参考
```
@repo{2020mmclassification,
    title={OpenMMLab's Image Classification Toolbox and Benchmark},
    author={MMClassification Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmclassification}},
    year={2020}
}
```