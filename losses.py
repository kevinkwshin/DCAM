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

class FocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha 
        self.gamma = gamma 

    def forward(self, inputs, targets):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE
                       
        return focal_loss