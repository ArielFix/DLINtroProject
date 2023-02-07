import torch
import torch.nn as nn


class ModifyRoberta:

    def modify_binary_output(model):
        model.classifier.out_proj = nn.Linear(in_features=768, out_features=2, bias=True)

    def modify_only_train_calssifier(model):
        for parameter in model.parameters():
            parameter.requires_grad = False

        for classifier_parameter in model.classifier.parameters():
            classifier_parameter.requires_grad = True

