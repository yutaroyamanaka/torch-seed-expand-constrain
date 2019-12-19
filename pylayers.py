import torch
import numpy as np
from torch.autograd import Variable
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian

min_prob = 0.0001

def seed_loss_layer(fc8_sec_softmax, cues):
    """
    :param fc8_sec_softmax: (batch_size, num_class(including background), height // 8, width // 8)
    :param cues: (batch_size, num_class + 1 (including background), height // 8, width // 8)
    :return: seeding loss
    """
    probs = fc8_sec_softmax
    labels = cues
    count = labels.sum(3).sum(2).sum(1)
    loss_balanced = - ((labels * torch.log((probs + 1e-4) / (1 + 1e-4))).sum(3).sum(2).sum(1) / (count)).mean(0)

    return loss_balanced


def expand_loss_layer(fc8_sec_softmax, labels, height, width, num_class):
    """
    :param fc8_sec_softmax: (batch_size, num_class + 1 (including background), height // 8, width // 8)
    :param labels: (batch_size, 1, 1, num_class(including background)) labels[0, 0, 0] shows one-hot vector of classification inference.
    :param height: one eighth of input image height
    :param width: one eighth of input image width
    :param num_class: number of classes including background
    :return: expansion loss
    """
    probs_tmp = fc8_sec_softmax
    stat_inp = labels

    # only foreground classes
    stat = stat_inp[:, :, :, 1:]

    # background class index is 0
    probs_bg = probs_tmp[:, 0, :, :]

    # foreground class indexes start from 1
    probs = probs_tmp[:, 1:, :, :]

    probs_max, _ = torch.max(torch.max(probs, 3)[0], 2)

    q_fg = 0.996
    probs_sort, _ = torch.sort(probs.contiguous().view(-1, num_class - 1, height * width), dim=2)
    weights = np.array([q_fg ** i for i in range(height * width - 1, -1, -1)])[None, None, :]
    Z_fg = np.sum(weights)
    weights_var = Variable(torch.from_numpy(weights).cuda()).squeeze().float()
    probs_mean = ((probs_sort * weights_var) / Z_fg).sum(2)

    q_bg = 0.999
    probs_bg_sort, _ = torch.sort(probs_bg.contiguous().view(-1, height * width), dim=1)
    weights_bg = np.array([q_bg ** i for i in range(height * width - 1, -1, -1)])[None, :]
    Z_bg = np.sum(weights_bg)
    weights_bg_var = Variable(torch.from_numpy(weights_bg).cuda()).squeeze().float()
    probs_bg_mean = ((probs_bg_sort * weights_bg_var) / Z_bg).sum(1)

    # boolean vector that only training label is true and others are false.
    # (1 - stat2d ) shows one-hot vector that only train label is 0 and others are 1.
    stat_2d = (stat[:, 0, 0, :] > 0.5).float()

    # loss for the class equivalent to training label
    loss_1 = -torch.mean(torch.sum((stat_2d * torch.log(probs_mean) / torch.sum(stat_2d, dim=1, keepdim=True)), dim=1))

    # loss for classes that are not training labels
    loss_2 = -torch.mean(
        torch.sum(((1 - stat_2d) * torch.log(1 - probs_max) / torch.sum(1 - stat_2d, dim=1, keepdim=True)), dim=1))

    # loss for background
    loss_3 = -torch.mean(torch.log(probs_bg_mean))

    loss = loss_1 + loss_2 + loss_3
    return loss


def constrain_loss_layer(fc8_sec_softmax, crf_result):
    """
    :param fc8_sec_softmax: (batch_size, num_class(including background), height // 8, width // 8)
    :param crf_result: (batch_size, num_class(including background), height // 8, width // 8)
    :return: constrain to boundary loss
    """

    probs = fc8_sec_softmax
    probs_smooth_log = Variable(torch.from_numpy(crf_result).cuda())

    probs_smooth = torch.exp(probs_smooth_log).float()
    loss = torch.mean((probs_smooth * torch.log(probs_smooth / probs)).sum(1))

    return loss


def crf_layer(fc8_sec_softmax, downscaled, iternum):
    """
    :param fc8_sec_softmax: (batch_size, num_class(including background), height // 8, width // 8)
    :param downscaled: (batch_size, height, width, 3 (RGB))
    :param iternum: times that calculation CRF inference repeatedly
    :return: crf inference results
    """

    unary = np.asarray(fc8_sec_softmax.cpu().data)
    imgs = downscaled
    N = unary.shape[0]  # batch_size
    result = np.zeros(unary.shape)  # (batch_size, num_class, height, width)

    for i in range(N):
        d = dcrf.DenseCRF2D(imgs[i].shape[1], imgs[i].shape[0], unary[i].shape[0])  # DenseCRF(width, height, num_class)

        # get unary
        U = unary_from_softmax(unary[i])
        # set unary potentials
        d.setUnaryEnergy(U)

        # This creates the color-independent features and then add them to the CRF
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        img = imgs[i].cpu().numpy()
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

        Q = d.inference(iternum)
        proba = np.asarray(Q)
        result[i] = np.reshape(proba, (unary[i].shape[0], img.shape[0], img.shape[1]))

    result[result < min_prob] = min_prob
    result = result / np.sum(result, axis=1, keepdims=True)

    crf_result = np.log(result)

    return crf_result
