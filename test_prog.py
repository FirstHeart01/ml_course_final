import torch
import torchvision
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from models import *
from utils.train_utils import torch2vector

model = eval('KNeighborsClassifier')(n_neighbors=4, weights='uniform')

# 数据预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    # transforms.Normalize(
    #     (0.1307,), (0.3081,)
    # )
])

# 读取数据
train_set = torchvision.datasets.MNIST('./datasets/mnist', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST('./datasets/mnist', train=False, download=True, transform=transform)

test_loader = DataLoader(test_set, shuffle=True, batch_size=10000)
for batch_idx, data in enumerate(test_loader):
    inputs, targets = data
    x = inputs.view(-1, 28*28)
    x_std = x.std().mean()
    x_mean = x.mean().item()

print(x_mean)
print(x_std)
# X_train, y_train, X_test, y_test = torch2vector(train_set, test_set)
# model.fit(X_train, y_train)

print()
