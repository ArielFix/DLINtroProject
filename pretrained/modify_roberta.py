import torch
import torch.nn as nn


class ModifyRoberta:

    def modify_binary_output(model):
        model.classifier.dense = nn.Linear(in_features=768, out_features=768, bias=True)
        model.classifier.dropout = nn.Dropout(p=0.1, inplace=False)
        model.classifier.out_proj = nn.Sequential(nn.Linear(in_features=768, out_features=2, bias=True), nn.LogSoftmax(dim=1))

    def modify_binary_output_activation(model):
        model.classifier.dense = nn.Sequential(nn.Linear(in_features=768, out_features=768, bias=True),
                                               nn.LeakyReLU())
        model.classifier.dropout = nn.Dropout(p=0.1, inplace=False)
        model.classifier.out_proj = nn.Sequential(nn.Linear(in_features=768, out_features=2, bias=True), nn.LogSoftmax(dim=1))

    def modify_binary_output_activation_extended(model):
        model.classifier.dense = nn.Sequential(nn.Linear(in_features=768, out_features=768, bias=True),
                                               nn.LeakyReLU(),
                                               nn.Dropout(p=0.1, inplace=False),
                                               nn.Linear(in_features=768, out_features=768, bias=True),
                                               nn.LeakyReLU())
        model.classifier.dropout = nn.Dropout(p=0.1, inplace=False)
        model.classifier.out_proj = nn.Sequential(nn.Linear(in_features=768, out_features=2, bias=True), nn.LogSoftmax(dim=1))

    def modify_only_train_calssifier(model):
        for parameter in model.parameters():
            parameter.requires_grad = False

        for classifier_parameter in model.classifier.parameters():
            classifier_parameter.requires_grad = True


        model.classifier.out_proj[0].reset_parameters

