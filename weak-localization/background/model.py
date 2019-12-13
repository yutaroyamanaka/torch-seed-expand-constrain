import torch.nn as nn
from torchvision import models

# network for localizing background


class VggBackgroundPretrained(nn.Module):

    def __init__(self, n_class, width, height):
        super().__init__()
        self.n_class = n_class
        self.width = width
        self.height = height
        self.net = models.vgg16(pretrained=True)
        self.features = self.net.features
        self.fc6 = nn.Linear(512 * (width // 32) * (height // 32), 1024, bias=False)
        # torch.nn.init.normal_(self.fc6.weight, mean=0, std=0.005)

        self.fc7 = nn.Linear(1024, 1024, bias=False)
        # torch.nn.init.normal_(self.fc7.weight, mean=0, std=0.005)

        self.fc8 = nn.Linear(1024, self.n_class, bias=False)
        # torch.nn.init.normal_(self.fc8.weight, mean=0, std=0.005)

        self.grad_in = None
        # register backward hook in conv4_1
        self.features[17].register_backward_hook(self.fun)

    def forward(self, x):
        h = x
        h = self.features(h)
        h = h.view(-1, 512 * (self.width // 32) * (self.height // 32))
        h = self.fc6(h)
        h = self.fc7(h)
        h = self.fc8(h)

        return h

    def fun(self, module, grad_in, grad_out):
        self.grad_in = grad_in[0]
