import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
from io import BytesIO
import IPython.display
import numpy as np
from math import ceil
from PIL import Image, ImageDraw
import imageio
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

import pretrained_networks
network_pkl = "/zhome/ca/6/92701/Desktop/Master_Thesis/stylegan2/results/00011-stylegan2-Accepted_records-4gpu-config-f/network-snapshot-007579.pkl"


print('Loading networks from "%s"...' % network_pkl)
_G, _D, Gs = pretrained_networks.load_networks(network_pkl)

img = np.random.randint(255+1, size=(1,3,256,256))
labels = np.random.randint(2, size=(1,0))
TP = 0
TN = 0
FP = 0
FN = 0
Total = 0


batch_size = 1





def test_loader():
	data_path = "/zhome/ca/6/92701/Desktop/Master_Thesis/stylegan2/rl_images/Final/Test/"
	test_dataset = torchvision.datasets.ImageFolder(
		root=data_path,
		transform=torchvision.transforms.Compose([
			torchvision.transforms.Pad((21, 22, 22, 21)),
			torchvision.transforms.ToTensor()])
	)
	test_loader = torch.utils.data.DataLoader(
		test_dataset,
		batch_size=batch_size,
		num_workers=0,
		shuffle=False
	)
	return test_loader

for i, (test_images, test_labels) in enumerate(test_loader(),0):
        loader = test_loader()
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output_tensor = tf.nn.softplus(-_D.get_output_for(img, labels, is_training=False))
            pred = sess.run(output_tensor)
            
        truth = test_labels.detach().cpu().numpy()[0]
        sample_fname, _ = loader.dataset.samples[i]
        if pred == 1 and truth == 1:
            TP += 1
        elif pred ==0 and truth == 0:
            TN += 1
        elif pred == 1 and truth == 0:
            FP += 1
        elif pred == 0 and truth == 1:
            FN += 1
            error_list.append(sample_fname.rsplit('/', 1)[-1])
        Total += 1


        
        if Total % 20 == 0 or Total == Num_test_pics:
            clear_output(wait=True)
            print(f"Positive images so far: {TP+FN}")
            print(f"Negative images so far: {TN+FP}")
            print(f"Tested {Total} images out of {Num_test_pics}!")
            print(f"Accuracy on positive images is {TP/(TP+FN+0.001)*100}%")
            print(f"Accuracy on negative images is {TN/(TN+FP+0.001)*100}%")
            print(f"Total Accuracy is {(TP+TN)/(Total+0.001)*100}%")   



#print(output_numpy)
#noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
