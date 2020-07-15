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
import matplotlib as plt

import pretrained_networks
network_pkl = "/zhome/ca/6/92701/Desktop/Master_Thesis/stylegan2/results/00011-stylegan2-Accepted_records-4gpu-config-f/network-snapshot-007579.pkl"


print('Loading networks from "%s"...' % network_pkl)
_G, _D, Gs = pretrained_networks.load_networks(network_pkl)

batch_size=1
IMG_HEIGHT = 256
IMG_WIDTH = 256
validation_dir = '/zhome/ca/6/92701/Desktop/Master_Thesis/stylegan2/rl_images/Final/Validation/'

validation_Accept_dir = os.path.join(validation_dir, 'accepted')
validation_Reject_dir = os.path.join(validation_dir, 'rejected')

validation_image_generator = ImageDataGenerator(rescale=1./255)

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')
img_list = list()
for img, label in val_data_gen:
	
	img = img.reshape(1,3,256,256)
	label = np.expand_dims(label,1)
	dummy_label = np.random.randint(1, size=(1,0))
	dummy_img = np.random.rand(1,3,256,256)
	img_list.append(img) 
	#img_list.append(dummy_img)
	
	for img in img_list:
		output_tensor = _D.get_output_for(img, dummy_label, is_training=False)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			pred = sess.run(output_tensor)
		print(f"out_score= {pred},    label = {label}")

"""
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
"""
