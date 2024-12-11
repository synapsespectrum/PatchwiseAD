import torch
import torch.nn as nn


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        # apply sigmoid to pred
        pred = nn.Sigmoid()(pred)
        return self.bce(pred, target)

class PretrainingLoss(nn.Module):
    """
    Combine MSE loss in time and frequency domain
    """
    def __init__(self, alpha=0.5):
        super(PretrainingLoss, self).__init__()
        self.alpha = alpha
        self.freq_loss = nn.MSELoss()

    def forward(self, pred, target, mask):
        mse_loss = (pred - target) ** 2
        mse_loss = mse_loss.mean(dim=-2)
        time_loss = (mse_loss * mask).sum() / mask.sum()

        # apply fft to pred and target
        pred_fft = torch.fft.fftn(pred, dim=(-3, -2, -1))
        target_fft = torch.fft.fftn(target, dim=(-3, -2, -1))

        mae = torch.abs(pred_fft - target_fft)

        # mse = self.freq_loss(pred_fft, target_fft)
        avg_loss = mae.mean(dim=-2)
        freq_loss = (avg_loss * mask).sum() / mask.sum()

        return self.alpha * time_loss + (1 - self.alpha) * freq_loss


