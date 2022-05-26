from models import *
import torch
import torch.nn as nn


def build_model(config, model_type, auto_search):
    if model_type == 1:
        return eval(config.get('model_name'))()
    else:
        if auto_search:
            clf = eval(config.pop('model_name'))()
            return eval('GridSearchCV')(estimator=clf, param_grid=config.get('parameters'), n_jobs=config.get('n_jobs'))
        else:
            return eval(config.pop('model_name'))(**config.get('parameters'))



class BuildModel(nn.Module):
    def __init__(self, config, model_type, auto_search=False):
        super(BuildModel, self).__init__()
        self.model = build_model(config, model_type, auto_search)
        if model_type == 1:
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def extract_feats(self, image):
        x = self.model(image)
        return x

    # def forward(self, x, return_loss=True, **kwargs):
    #     if return_loss:
    #         return self.forward_train(x, **kwargs)
    #     else:
    #         return self.forward_test(x, **kwargs)
    #
    # def forward_train(self, x, targets, **kwargs):
    #     x = self.extract_feat(x)
    #     losses = dict()
    #     loss = self.head.forward_train(x, targets, **kwargs)
    #     losses.update(loss)
    #     return losses
    #
    # def forward_test(self, x, **kwargs):
    #     x = self.extract_feat(x)
