import os
import torch
from erfnet import ERFNet
from PIL import Image
from torchvision.transforms import Compose, Resize
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import numpy as np
import threading

NUM_CLASSES = 20
THREADS = 12

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

# This is my path
path="raw_data/"

os.chdir("../../dataset/kitti/")

list = []

#Create directoryes 
for (root, dirs, file) in os.walk(path):
    for f in file:
        if '.png' in f:
            newPath = root.replace('raw_data', 'semantic_segmentation').replace("\\","/") + "/"
            if not os.path.exists(newPath):
                os.makedirs(newPath)
            
            list.append((root + "/" + f, newPath + f))


sizePerThread = int(len(list) / THREADS)

def ThreadFunc(start, end, data):
    for (filePath, newPath) in data[start:end]:
        if not os.path.exists(newPath):
            with open(filePath, 'rb') as imFile:
                image = Image.open(imFile).convert('RGB')

            image = input_transform(image)
            image = torch.unsqueeze(image, dim=0)
            input = Variable(image)

            output = model(input)

            output = output.cpu().data[0].numpy()
            output = output.transpose(1, 2, 0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            output_color = cityscapes_colorize_mask(output)

            output_color.save(newPath)
            print(newPath)

t = []
for i in range(THREADS):
    if i != THREADS - 1:
        t += [threading.Thread(target=ThreadFunc, args=(sizePerThread * i, sizePerThread * (i+1), list))]
    else:
        t += [threading.Thread(target=ThreadFunc, args=(sizePerThread * i, len(list), list))]
  
    t[i].start()

for thread in t:
    thread.join()
    


        