import torch
import netron
from torchviz import make_dot
from models import *

x = torch.randn(1, 28, 28).requires_grad_(True)
net = eval('MyAlexNet')()
model_path = r'E:\CodeFiles\Pycharm_Projects\dl_learn_pros\ml_course_final\logs\MyAlexNet\2022-05-26-00-02-53\Last_Epoch020.pth'
y = net(x)
# netVis = make_dot(y, params=dict(list(net.named_parameters()) + [('input_shape', x)]))
# netVis.format = "png"
# netVis.directory = "network_vis"
# netVis.view()
