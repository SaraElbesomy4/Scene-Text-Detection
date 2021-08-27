#-*- coding:utf-8 -*-
#'''
# Created on 18-12-11 上午10:01
#
# @Author: Greg Gao(laygin)
#'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class RPN_REGR_Loss(nn.Module):
    def __init__(self, device, sigma=9.0):
        super(RPN_REGR_Loss, self).__init__()
        self.sigma = sigma
        self.device = device

    def forward(self, input, target):
        '''
        smooth L1 loss
        :param input:y_preds
        :param target: y_true
        :return:
        '''
        try:
            cls = target[0, :, 0]
            regr = target[0, :, 1:3]
            regr_keep = (cls == 1).nonzero()[:, 0]
            regr_true = regr[regr_keep]
            regr_pred = input[0][regr_keep]
            diff = torch.abs(regr_true - regr_pred)
            less_one = (diff<1.0/self.sigma).float()
            loss = less_one * 0.5 * diff ** 2 * self.sigma + torch.abs(1- less_one) * (diff - 0.5/self.sigma)
            loss = torch.sum(loss, 1)
            loss = torch.mean(loss) if loss.numel() > 0 else torch.tensor(0.0)
        except Exception as e:
            print('RPN_REGR_Loss Exception:', e)
            # print(input, target)
            loss = torch.tensor(0.0)

        return loss.to(self.device)


class RPN_CLS_Loss(nn.Module):
    def __init__(self,device):
        super(RPN_CLS_Loss, self).__init__()
        self.device = device

    def forward(self, input, target):
        y_true = target[0][0]
        cls_keep = (y_true != -1).nonzero()[:, 0]
        cls_true = y_true[cls_keep].long()
        cls_pred = input[0][cls_keep]
        loss = F.nll_loss(F.log_softmax(cls_pred, dim=-1), cls_true)  # original is sparse_softmax_cross_entropy_with_logits
        # loss = nn.BCEWithLogitsLoss()(cls_pred[:,0], cls_true.float())  # 18-12-8
        loss = torch.clamp(torch.mean(loss), 0, 10) if loss.numel() > 0 else torch.tensor(0.0)
        return loss.to(self.device)







class CTPN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.vgg16(pretrained=True)
        layers = list(base_model.features)[:-1]
        self.base_layers = nn.Sequential(*layers)  # block5_conv3 output
        self.rpn = nn.Conv2d(512, 512, 3, 1, 1, 1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.lstm_fc = nn.Conv2d(512, 512, 1, 1, 0, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.rpn_regress = nn.Conv2d(512, 6, 1, 1, 0, bias=True)

    def forward(self, x):
        x = self.base_layers(x)
        x = self.rpn(x)  # [b, c, h, w]
        x = self.relu1(x)
        x = self.lstm_fc(x)
        x = self.relu2(x)
        regr = self.rpn_regress(x)

        return  regr
