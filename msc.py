import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torchvision import models
import argparse


def args_solve():
    parser = argparse.ArgumentParser()

    parser.add_argument('--height', type=int)
    parser.add_argument('--width', type=int)
    parser.add_argument('--num_class', type=int, help="number of classes including background")

    args = parser.parse_args()
    return args


class MSC(nn.Module):

    def __init__(self, n_class, height, width, pretrained=True):
        super().__init__()
        self.width = width
        self.height = height
        self.n_class = n_class

        # adjust model parameters according to image input size.
        # For your information, my dataset image has 240(height) * 320(width) pixels.
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (1, 1), ceil_mode=True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (1, 1), ceil_mode=True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (1, 1), ceil_mode=True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (1, 1), (1, 1), ceil_mode=True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (2, 2), dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (2, 2), dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (2, 2), dilation=2),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (1, 1), (1, 1), ceil_mode=True),
            nn.AvgPool2d((4, 4), (1, 1), (1, 1), ceil_mode=True),  # AvgPool2d,
            nn.Conv2d(512, 1024, (3, 3), (1, 1), (12, 12), dilation=12),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(1024, 1024, (1, 1)),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(1024, n_class, (1, 1)),
        )

        if pretrained:
            self.load_weights()

    def forward(self, x):
        input_size = x.size()[2:]
        self.interp1 = nn.UpsamplingBilinear2d(size=(int(input_size[0] * 0.75) + 1, int(input_size[1] * 0.75) + 1))
        self.interp2 = nn.UpsamplingBilinear2d(size=(int(input_size[0] * 0.5) + 1, int(input_size[1] * 0.5) + 1))
        self.interp3 = nn.UpsamplingBilinear2d(size=(int(input_size[0]//8), int(input_size[1]//8)))
        out = []
        x2 = self.interp1(x)
        x3 = self.interp2(x)

        out.append(self.net(x))  # for original scale
        out.append(self.interp3(self.net(x2)))  # for 0.75x scale
        out.append(self.net(x3))  # for 0.5x scale

        x2Out_interp = out[1]
        x3Out_interp = self.interp3(out[2])
        temp1 = torch.max(out[0], x2Out_interp)
        out.append(torch.max(temp1, x3Out_interp))
        return out

    def load_weights(self):
        net_params = self.net.parameters()
        base_net = models.vgg16(pretrained=True)
        org_params = list(base_net.parameters())

        count = 0
        for p, src_p in zip(net_params, org_params):
            if p.shape == src_p.shape:
                p.data[:] = src_p.data[:]
                count += 1

        print('[%d] pre-trained weights are loaded' % count)


def check_net(n_class, height, width):
    net = MSC(n_class, height, width)

    # net.cuda()
    # inputs = Variable(torch.zeros([1, 3, 321, 321]).cuda())

    inputs = Variable(torch.zeros([1, 3, height, width]))
    out = net(inputs)
    print(out[3].shape) # (batch_size, num_class, height, width)


if __name__ == '__main__':
    # python msc.py --height 240 --width 320 --num_class 4
    args = args_solve()
    check_net(args.num_class, args.height, args.width)
