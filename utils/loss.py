import torch
import torch.nn.functional as F
from torch import nn


class ReLoss(nn.Module):
    def __init__(self, device):
        super(ReLoss, self).__init__()
        self.sobelconv = Sobelxy(device)

    def forward(self, img1, img2):
        img1 = img1.unsqueeze(1)
        img2 = img2.unsqueeze(1)
        grad_x1, grad_y1 = self.sobelconv(img1)
        grad_x2, grad_y2 = self.sobelconv(img2)
        loss_grad = F.l1_loss(grad_x1, grad_x2) + F.l1_loss(grad_y1, grad_y2)
        loss_intensity = F.l1_loss(img1, img2)
        loss_total = loss_intensity + loss_grad
        return loss_total


class Sobelxy(nn.Module):
    def __init__(self, device):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        # 这里不行就采用expend_dims
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(device=device)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(device=device)

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx), torch.abs(sobely)
