import torch
from model import VggBackgroundPretrained
from torchvision import transforms
import sys
from PIL import Image

width = 320
height = 240
n_class = 3

net = VggBackgroundPretrained(n_class=n_class, width=width, height=height)

load_path = "./vgg-background-weights_fine_tuning.pth"
load_weights = torch.load(load_path, map_location={'cuda:0': 'cpu'})

net.load_state_dict(load_weights)
net.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def inference(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((width, height))

    image = transform(image)
    inputs = image
    inputs = inputs.view(1, 3, height, width)
    output = net(inputs)
    _, preds = torch.max(output, 1)  # ラベルを予測

    print(path, preds)

if __name__ == '__main__':
    # python inference.py path_to_image_file
    inference(sys.argv[1])