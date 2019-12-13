import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
# network for localizing the foreground classes


class VggForegroundPretrained(nn.Module):

    def __init__(self, n_class, width, height):
        super().__init__()
        self.n_class = n_class
        self.width = width
        self.height = height
        self.net = models.vgg16(pretrained=True)
        self.fore_features = self.net.features[:23]
        self.middle_features = self.net.features[24:30]
        self.fc6_cam = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.fc6_dropout = nn.Dropout2d()

        self.fc7_cam = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.fc7_dropout = nn.Dropout2d()

        self.fc8 = nn.AvgPool2d(kernel_size=(int(self.height // 8), int(self.width // 8)), stride=1)

        self.scores = nn.Linear(1024, self.n_class, bias=False)

    def forward(self, x):
        h = x
        h = self.fore_features(h)
        h = self.middle_features(h)
        h = self.fc6_dropout(F.relu(self.fc6_cam(h), inplace=True))

        h = self.fc7_dropout(self.fc7_cam(h))
        h = self.fc8(h)
        h = h.view(-1, 1024)
        h = self.scores(h)
        return h