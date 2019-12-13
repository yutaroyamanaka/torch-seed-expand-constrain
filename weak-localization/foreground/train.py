import torch
import torch.nn as nn
from model import VggForegroundPretrained
import os
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import sys
# search for data_loader
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from data_loader import ImageDataset

root = "../../dataset/"
width = 320
height = 240
n_class = 3
batch_size = 32
lr = 1e-4
momentum = 0.9
epochs = 15

net = VggForegroundPretrained(n_class=n_class, width=width, height=height)
net.train()

train_data = ImageDataset(root, width=width, height=height, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

val_data = ImageDataset(root, 'val', width=width, height=height, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))
val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=0)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-------------')

        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()  # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数

            # 未学習時の検証性能を確かめるため、epoch=0の訓練は省略
            if (epoch == 0) and (phase == 'train'):
                continue

            if phase == 'train':

                for iter, batch in enumerate(train_loader):
                    inputs = batch["image"].to(device)
                    labels = batch["target"].to(device)

                    # optimizerを初期化
                    optimizer.zero_grad()

                    # 順伝搬（forward）計算
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = net(inputs)
                        loss = criterion(outputs, labels)  # 損失を計算
                        _, preds = torch.max(outputs, 1)  # ラベルを予測
                        print(preds)
                        loss.backward()
                        optimizer.step()

                        # 結果の計算
                        epoch_loss += loss.item() * inputs.size(0)  # lossの合計を更新
                        # 正解数の合計を更新
                        epoch_corrects += torch.sum(preds == labels.data)

                        # epochごとのlossと正解率を表示
                        epoch_loss = epoch_loss / len(train_loader.dataset)
                        epoch_acc = epoch_corrects.double(
                        ) / len(train_loader.dataset)

                        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                            phase, epoch_loss, epoch_acc))
            elif phase == 'val':

                for iter, batch in enumerate(val_loader):
                    # GPUが使えるならGPUにデータを送る
                    inputs = batch["image"].to(device)
                    labels = batch["target"].to(device)

                    outputs = net(inputs)
                    loss = criterion(outputs, labels)  # 損失を計算
                    _, preds = torch.max(outputs, 1)  # ラベルを予測

                    # 結果の計算
                    epoch_loss += loss.item() * inputs.size(0)  # lossの合計を更新
                    # 正解数の合計を更新
                    epoch_corrects += torch.sum(preds == labels.data)

                    # epochごとのlossと正解率を表示
                    epoch_loss = epoch_loss / len(val_loader.dataset)
                    epoch_acc = epoch_corrects.double(
                    ) / len(val_loader.dataset)

                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase, epoch_loss, epoch_acc))


if __name__ == '__main__':
    train()
    save_path = './vgg-foreground-weights_fine_tuning.pth'
    torch.save(net.state_dict(), save_path)

