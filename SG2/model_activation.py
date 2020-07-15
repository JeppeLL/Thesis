import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json
import Inception

import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F


def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

img = get_image('/zhome/ca/6/92701/Desktop/Master_Thesis/stylegan2/rl_images/256/Training/rejected/reject_ffa38c95-8ca8-4a53-9bfd-06696a09058b_5_rl.jpg')
plt.imshow(img)


def get_input_transform():
  
    transf = transforms.Compose([
        transforms.Pad((21,22,22,21)),
        transforms.ToTensor()
        ])    

    return transf

def get_input_tensors(img):
    transf = get_input_transform()

    return transf(img).unsqueeze(0)
    
model = Inception.inception_v3(img_size=256)
cp = torch.load('/zhome/ca/6/92701/Desktop/Master_Thesis/Results_Inception/1_output/inception_data-augment.pth.tar',map_location=torch.device('cpu'))
model.load_state_dict(cp['state_dict'])


img_t = get_input_tensors(img)
model.eval()
logits = model(img_t)

probs = F.sigmoid(logits)


