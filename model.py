import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(2, 2))
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2))
        self.bn3 = nn.BatchNorm2d(32)

        self.lin1 = nn.Sequential(
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(F.relu(self.bn2(self.conv2(out))))
        out = F.relu(self.bn3(self.conv3(out)))
        out = torch.flatten(out, start_dim=1)
        out = self.lin1(out)

        return out