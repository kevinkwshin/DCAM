import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCE(nn.Module):
    def __init__(self):
        super(DiceBCE, self).__init__()
        self.bce = nn.BCELoss()
        self.dice = monai.losses.DiceLoss()
        
    def forawrd(self,yhat,y):
        bce = self.bce(yhat,y)
        dice = self.dice(yhat,y)
        return bce + dice