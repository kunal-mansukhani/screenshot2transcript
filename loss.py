import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, num_classes=3):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        # One-hot encode y_true
        y_true_one_hot = F.one_hot(y_true, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Apply softmax to y_pred to get probability distributions
        y_pred = F.softmax(y_pred, dim=1)
        
        # Calculate Dice Loss for each class
        intersection = (y_pred * y_true_one_hot).sum(dim=(2, 3))
        union = y_pred.sum(dim=(2, 3)) + y_true_one_hot.sum(dim=(2, 3))
        
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score.mean()  # Average across classes and batch

        return dice_loss