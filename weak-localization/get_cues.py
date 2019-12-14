from background.model import VggBackgroundPretrained
from foreground.model import VggForegroundPretrained
import torch
from torchvision import transforms
import numpy as np
from scipy.ndimage import median_filter
import os
from PIL import Image
import pickle
import sys

width = 320
height = 240
n_class = 3

# network for getting foreground cues
bg_net = VggBackgroundPretrained(width=width, height=height, n_class=n_class)
# network for getting background cues
fg_net = VggForegroundPretrained(width=width, height=height, n_class=n_class)

# weight path for bg_net
background_cam_model_path = "./background/vgg-background-weights_fine_tuning.pth"
# weight path for fg_net
foreground_cam_model_path = "./foreground/vgg-foreground-weights_fine_tuning.pth"

bg_net.load_state_dict(torch.load(background_cam_model_path, map_location={'cuda:0': 'cpu'}))
bg_net.eval()
fg_net.load_state_dict(torch.load(foreground_cam_model_path, map_location={'cuda:0': 'cpu'}))
fg_net.eval()

# middle layer of foreground net
fore_feature = fg_net.fore_features
middle_features = fg_net.middle_features
fc6_cam = fg_net.fc6_cam
fc6_dropout = fg_net.fc6_dropout
fc7_cam = fg_net.fc7_cam
fg_CAM_scores = list(fg_net.fc7_cam.parameters())
fg_params = list(fg_net.scores.parameters())

# transform setting
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def get_fg_cues(inputs):
    """
    :param inputs: normalized tensor
    :return: boolean mask of foreground cues
    """
    heat_maps = np.zeros((n_class, height // 8, width // 8))
    localization = np.zeros((n_class, height // 8, width // 8))

    # input forward to fc7_cam
    h = inputs
    h = fore_feature(h)
    h = middle_features(h)
    h = fc6_cam(h)
    h = fc6_dropout(h)
    h = fc7_cam(h)
    h = h.detach().cpu().numpy()

    for i in range(n_class):
        w = fg_params[0][i].detach().cpu().numpy()
        heat_maps[i, :, :] = np.sum(h[0] * w[:, None, None], axis=0)
        localization[i, :, :] = heat_maps[i, :, :] > 0.7 * np.max(heat_maps[i])

    return localization


def get_bg_cues(inputs):
    """
    :param inputs: normalized tensor
    :return: boolean mask of background cues
    """

    output = bg_net(inputs)
    target = torch.FloatTensor(1, output.shape[-1])
    target[0, :] = 1
    bg_net.zero_grad()
    output.backward(gradient=target)
    grad = bg_net.grad_in.cpu().data.numpy()
    saliency_maps = median_filter(np.max(np.abs(grad[0]), axis=0), 3)

    # threshold value
    thr = np.sort(saliency_maps.ravel())[int(0.1 * (width // 8) * (height // 8))]
    localization = saliency_maps < thr

    return localization


def solve_conflicts(fg_cues, bg_cues, full_width, full_height):
    """
    :param fg_cues: mask of foreground cues
    :param bg_cues: mask of background cues
    :return: cues array to save in pickle
    """

    classes = []
    cols = []
    rows = []
    # number of cues equivalent to each foreground class
    stack_nums = np.count_nonzero(fg_cues, axis=(1, 2))
    for h in range(full_height):
        for w in range(full_width):

            fg_temp_indexes = []
            for c in range(fg_cues.shape[0]):
                if fg_cues[c][h][w]:
                    fg_temp_indexes.append(c)

            # if there is foreground cues in the pixel
            if len(fg_temp_indexes):
                # solve class which has minimum stack_nums
                min_nums = full_height * full_width
                min_index = -1
                for fg_temp_index in fg_temp_indexes:
                    if stack_nums[fg_temp_index] < min_nums:
                        min_index = fg_temp_index
                        min_nums = stack_nums[fg_temp_index]

                classes.append(min_index)
                cols.append(h)
                rows.append(w)
            # if there is background cue and no foreground cues
            elif bg_cues[h][w]:
                classes.append(-1)  # bg_cue ravel is defined -1
                cols.append(h)
                rows.append(w)

    cues = [classes, cols, rows]
    return np.asarray(cues)


def generate_pickle(path_to_dataset):

    path_cues_dict = dict()
    for label in os.listdir(path_to_dataset):
        img_dir = os.path.join(path_to_dataset, label)
        for file_name in os.listdir(img_dir):
            img_file_path = os.path.join(img_dir, file_name)

            # read image
            image = Image.open(img_file_path).convert("RGB")
            image = image.resize((width, height))

            image = transform(image)
            inputs = image
            inputs = inputs.view(1, 3, height, width)

            # get fg_cues
            fg_cues = get_fg_cues(inputs)

            # get bg_cues
            bg_cues = get_bg_cues(inputs)

            # cues which has no conflict
            cues = solve_conflicts(fg_cues, bg_cues, width // 8, height // 8)
            path_cues_dict[img_file_path] = cues

    with open("localization_cues.pickle", "wb") as f:
        pickle.dumps(path_cues_dict, f)

    print("generate cues pickle")

if __name__ == '__main__':
    # python get_cues.py path_to_root_dataset
    generate_pickle(sys.argv[1])

