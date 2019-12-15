import torch
import os
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from data_loader import ImageDataset
from msc import MSC
import pickle
import numpy as np
from torch.autograd import Variable
from pylayers import seed_loss_layer, expand_loss_layer, constrain_loss_layer, crf_layer

root = "./dataset/"
width = 320
height = 240

# foreground classes and background
# background label is 0.
n_class = 4
batch_size = 32
lr = 1e-4
momentum = 0.9
epochs = 50

net = MSC(n_class=n_class, height=height, width=width)
net.train()

train_data = ImageDataset(root, width=width, height=height, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

path_to_localization_cues = "./weak-localization/localization_cues.pickle"

with open(path_to_localization_cues, "rb") as f:
    dict_src = pickle.load(f)


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-------------')

        for iter, batch in enumerate(train_loader):
            inputs = batch["image"].to(device)
            path = batch["path"].to(device)
            downscaled = batch["downscaled"].to(device)

            labels, dense_gt = get_data_from_batch(len(batch), path)

            # optimizer init
            optimizer.zero_grad()

            outputs = net(inputs)
            outputs = outputs[3]

            preds_max, _ = torch.max(outputs, dim=1, keepdim=True)
            preds_max_cpu = Variable(preds_max.data)
            preds_exp = torch.exp(outputs - preds_max_cpu)
            probs = preds_exp / torch.sum(preds_exp, dim=1, keepdim=True) + 1e-4

            fc8_sec_softmax = probs / torch.sum(probs, dim=1, keepdim=True)

            loss_s = seed_loss_layer(fc8_sec_softmax, dense_gt)
            loss_e = expand_loss_layer(fc8_sec_softmax, labels)
            crf_result = crf_layer(fc8_sec_softmax, downscaled, 10)
            loss_c = constrain_loss_layer(fc8_sec_softmax, crf_result)

            print(loss_s.cpu().data[0], loss_e.cpu().data[0], loss_c.cpu().data[0])
            loss = loss_s + loss_e + loss_c

            loss.backward()
            optimizer.step()


def get_data_from_batch(batch_len, img_path):

    dense_gt = np.zeros((height // 8, width // 8, n_class, len(batch_len)))
    labels = np.zeros((batch_len, 1, 1, n_class))

    for i in range(batch_len):
        img_name = os.path.basename(img_path[i])  # ./dataset/train/label_index/*.png
        cues_key = "./" + img_name  # cues object keys start from "./"
        cues_i = dict_src[cues_key]

        labels_i = np.unique(cues_i[0]).tolist()
        for lab in labels_i:
            if lab == -1:  # background
                labels[i, 0, 0, 0] = 1
            else:
                labels[i, 0, 0, lab+1] = 1

        gt_temp = np.zeros((height // 8, width // 8))

        for idx, class_index in enumerate(cues_i[0]): # cues_i is (label_array, height_array, width_array)
            if class_index == -1: # background
                gt_temp[cues_i[1, idx], cues_i[2, idx]] = 0  # defines background as 0 label.
            else: # foreground classes
                gt_temp[cues_i[1, idx], cues_i[2, idx]] = (cues_i[0, idx] + 1)

        gt_temp = gt_temp.astype('float')

        gt_temp_trues = np.zeros((height // 8, width // 8, n_class))

        for lab in labels_i:
            if lab == -1: # background
                gt_temp_trues[:, :, 0] = (gt_temp == 0).astype('float')

        dense_gt[:, :, :, i] = gt_temp_trues

    labels = Variable(labels.cuda())
    dense_gt = dense_gt.transpose((3, 2, 0, 1))
    dense_gt = Variable(torch.from_numpy(dense_gt).float()).cuda()

    return labels, dense_gt


if __name__ == '__main__':
    train()
    save_path = "./segmentation-weights.pth"
    torch.save(net.state_dict(), save_path)
