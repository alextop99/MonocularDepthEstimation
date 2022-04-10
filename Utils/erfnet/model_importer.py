import torch
from erfnet import ERFNet
from PIL import Image
from torchvision.transforms import Compose, Resize
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from matplotlib import pyplot as plt
import numpy as np

NUM_CLASSES = 20
#IMAGE_PATH = "./0000000005.png"
IMAGE_PATH = "./0000000038.png"

cityscapes_palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
                      220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0,
                      70, 0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

zero_pad = 256 * 3 - len(cityscapes_palette)
for i in range(zero_pad):
    cityscapes_palette.append(0)

def cityscapes_colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(cityscapes_palette)

    return new_mask

input_transform = Compose([
    Resize((256, 1024), Image.BILINEAR),
    ToTensor(),
])

def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            if name.startswith("module."):
                own_state[name.split("module.")[-1]].copy_(param)
            else:
                print(name, " not loaded")
                continue
        else:
            own_state[name].copy_(param)
    return model

model = ERFNet(NUM_CLASSES)
model = load_my_state_dict(model, torch.load("./erfnet_pretrained.pth", map_location=lambda storage, loc: storage))
model.eval()

print("Everything loaded successfully!!")

with open(IMAGE_PATH, 'rb') as f:
    image = Image.open(f).convert('RGB')

image = input_transform(image)
image = torch.unsqueeze(image, dim=0)
input = Variable(image)

output = model(input)

output = output.cpu().data[0].numpy()
output = output.transpose(1, 2, 0)
output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
output_color = cityscapes_colorize_mask(output)

fig, ax = plt.subplots(3, 1, figsize=(50, 50))
ax[0].imshow(image[0].transpose(0, 2).transpose(0, 1))
ax[1].imshow(output)
ax[2].imshow(output_color)

plt.show()