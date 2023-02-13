import torch
import torch.nn as nn
import torch.nn.functional as F

import monai

class DiceBCE(nn.Module):
    def __init__(self):
        super(DiceBCE, self).__init__()
        self.bce = nn.BCELoss()
        self.dice = monai.losses.DiceLoss()
        
    def forawrd(self,yhat,y):
        bce = self.bce(yhat,y)
        dice = self.dice(yhat,y)
        return bce + dice

class PropotionalLoss(nn.Module):
    def __init__(self, log=False, per_image=False, smooth=1e-7, beta=0.7, bce=False):
        super(PropotionalLoss, self).__init__()
        self.beta = beta
        self.smooth = smooth 
        self.log = log
        self.per_image = per_image
        self.bce = bce
        
    def forward(self, inputs, targets):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        AXIS = [-1]
        self.alpha = 1 - self.beta
        y_true = targets
        y_pred = inputs
        
        prevalence = torch.mean(y_true, axis=AXIS)
        tp = torch.sum(y_true * y_pred, axis=AXIS)
        tn = torch.sum((1 - y_true) * (1 - y_pred), axis=AXIS)
        fp = torch.sum(y_pred, axis=AXIS) - tp
        fn = torch.sum(y_true, axis=AXIS) - tp
        negative_score = (tn + self.smooth) / (tn + self.beta * fn + self.alpha * fp + self.smooth) * (self.smooth + 1 - prevalence)
        positive_score = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth) * (self.smooth + prevalence)
        score_per_image = negative_score + positive_score
        
        if self.log:
            score_per_image = -1 * torch.log(score_per_image)
        else:
            score_per_image = 1 - score_per_image
            
        if self.per_image == False:
            score_per_image = torch.mean(score_per_image)
        
        if self.bce:
            return score_per_image + F.binary_cross_entropy(y_pred, y_true)

class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha 
        self.gamma = gamma

    def forward(self, inputs, targets):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        # first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE
                       
        return focal_loss + BCE


class Consistency_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.L2_loss  = torch.nn.MSELoss()
        self.maxpool  = torch.nn.MaxPool1d(kernel_size=16, stride=16, padding=0)
        self.avgpool  = torch.nn.AvgPool1d(kernel_size=16, stride=16, padding=0)

    def forward(self, y_cls, y_seg):
        y_cls = torch.sigmoid(y_cls)  # (B, C)
        y_seg = torch.sigmoid(y_seg)  # (B, C, H, W)

        # We have to adjust the segmentation pred depending on classification pred
        # ResNet50 uses four 2x2 maxpools and 1 global avgpool to extract classification pred. that is the same as 16x16 maxpool and 16x16 avgpool
        y_seg = self.avgpool(self.maxpool(y_seg)).flatten(start_dim=1, end_dim=-1)  # (B, C)
        loss  = self.L2_loss(y_seg, y_cls)

        return loss