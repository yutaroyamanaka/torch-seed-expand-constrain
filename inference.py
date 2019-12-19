from msc import MSC
import torch
from PIL import Image
from torchvision import transforms
import argparse
from collections import OrderedDict
import scipy.ndimage as nd
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--height', type=int)
parser.add_argument('--width', type=int)
parser.add_argument('--num_class', type=int, help="number of classes including background")
parser.add_argument('--image', type=str, help="path to input image")
parser.add_argument('--weight', type=str, help="path to model weight file")
args = parser.parse_args()

height = args.height
width = args.width
n_class = args.num_class

net = MSC(n_class, height, width)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict


def preprocess(img_path):
    image = Image.open(img_path).convert("RGB")
    image = image.resize((width, height))

    inputs = transform(image)
    inputs = inputs.view(1, 3, height, width)

    return image, inputs


def predict_mask(load_path, inputs, image):

    device = torch.device('cpu')
    net.load_state_dict(fix_model_state_dict(torch.load(load_path, map_location=device)))
    net.eval()

    outputs = net(inputs)

    scores = np.transpose(outputs[3][0].detach().numpy(), [1, 2, 0])
    scores_exp = np.exp(scores - np.max(scores, axis=2, keepdims=True))
    probs = scores_exp / np.sum(scores_exp, axis=2, keepdims=True)

    probs = nd.zoom(probs, (height / probs.shape[0], width / probs.shape[1], 1.0), order=1)
    eps = 0.00001
    probs[probs < eps] = eps
    result = np.argmax(probs, axis=2)

    modified = result.copy()

    uniques = np.unique(result).tolist()
    for i, unique in enumerate(uniques):
        if unique != 0:
            modified[modified == unique] = i

    d = dcrf.DenseCRF2D(width, height, n_class)
    U = unary_from_labels(modified.ravel(), n_class, gt_prob=0.7, zero_unsure=False)

    #  get unary potentials (neg log probability)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=image,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(10)
    MAP = np.argmax(Q, axis=0)
    result_smoothed = MAP.reshape((height, width))

    return result_smoothed


if __name__ == '__main__':

    # python inference.py --height 240 --width 320 --num_class 4 --image ./dataset/train/0/car_215.jpg --weight ./segmentation-weights.pth

    image, inputs = preprocess(args.image)

    image = np.asarray(image)
    image = image.copy()
    mask = predict_mask(args.weight, inputs, image)

    fig = plt.figure()

    ax = fig.add_subplot('121')
    ax.imshow(image)

    ax = fig.add_subplot('122')
    ax.matshow(mask)

    plt.show()
