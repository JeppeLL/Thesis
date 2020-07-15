# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:30:08 2020

@author: lauri
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob
import os

##################################### Flipping and rotating ###################
img_00=Image.open("C:/Users/lauri/Desktop/DataPreprocessing/Data/Images_256/Training_rare_examples/stor bakterie/reject_58b8fa28-97e2-4446-aa45-0508a0b1ba4c_4_rl.jpg")
img_01=img_00.rotate(45)
img_10=img_00.transpose(Image.FLIP_TOP_BOTTOM)
img_11=img_01.transpose(Image.FLIP_TOP_BOTTOM)


fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(6,6))
axes[0,0].imshow(img_00)
axes[0,0].set_yticks([])
axes[0,0].set_xticks([])
axes[0,0].set_title("Original")

axes[0,1].imshow(img_01)
axes[0,1].set_yticks([])
axes[0,1].set_xticks([])
axes[0,1].set_title("Rotated")

axes[1,0].imshow(img_10)
axes[1,0].set_yticks([])
axes[1,0].set_xticks([])
axes[1,0].set_title("Flipped")

axes[1,1].imshow(img_11)
axes[1,1].set_yticks([])
axes[1,1].set_xticks([])
axes[1,1].set_title("Rotated and Flipped")

plt.show()


#### Lets try with our own implementation



################################# Flipping and rotating done ##################



################################### Fancy PCA ################################

def load_image_as_array(path):
    image = Image.open(path)
    original_image = np.array(image)
    return original_image
def load_images_as_arrays(paths):
    #Output dims: (batch_size,height,width,RGB)
    images=np.array([load_image_as_array(paths[0])])
    if len(paths)>1:
        for path in paths[1:]:
            images=np.append(images,np.array([load_image_as_array(path)]),axis=0)
    return images

image_dir="C:\\Users\\lauri\\Desktop\\DataPreprocessing\\Data\\Training"
rejected=sorted(glob.glob(os.path.join(image_dir,'Rejected','*')))
accepted=sorted(glob.glob(os.path.join(image_dir,'Accepted','*_rl.jpg')))
all = rejected+accepted
images = load_images_as_arrays(rejected)
del accepted, rejected, image_dir, all



def PCA(images):
    images_reshape=np.reshape(images,(-1,3))
    images_reshape = images_reshape.astype('float64')
    mean = np.mean(images_reshape, axis=0)
    std = np.std(images_reshape, axis=0)
    images_reshape -= mean
    images_reshape /= std
    cov = np.cov(np.transpose(images_reshape))
    lambdas, p = np.linalg.eig(cov)
    pca=np.zeros((3,3))
    for i in range(3):
        for c in range(3):
            pca[i,c]=p[c,i]*lambdas[i]
    pca = pca.astype('float32')
    mean = mean.astype('float32')
    std = std.astype('float32')
    return pca, mean, std
#pca, mean, std = PCA(images) #Takes a long time to run,
# the results are hard-coded below:
pca = np.array([[-1.5800763e+00, -1.5902529e+00, -1.4200467e+00],
       [-6.5569100e-03,  6.7732222e-03, -2.8922837e-04],
       [-1.3565156e-01, -1.1917505e-01,  2.8439787e-01]], dtype='float32')
mean = np.array([41.676228, 43.959198, 21.325665], dtype='float32')
std  = np.array([40.373974, 38.43527 , 25.48677 ], dtype='float32')


#Now we apply a Fancy PCA-transformation to an image

def fancy_pca_get_deltas(N_images,N_augmentations,pca,image_pixels=256**2):    
    alpha = np.random.normal(0,0.1,(N_images*N_augmentations,3))
    delta = np.matmul(alpha,pca)    
    delta = np.tile(delta,[1,image_pixels])
    delta = np.reshape(delta,[N_images,N_augmentations,-1,3])
    return delta


def augment_images_fancy_pca(images,N_augmentations):
  pca=np.array([[-1.5800763e+00, -1.5902529e+00, -1.4200467e+00],
       [-6.5569100e-03,  6.7732222e-03, -2.8922837e-04],
       [-1.3565156e-01, -1.1917505e-01,  2.8439787e-01]], dtype='float32')
  pca_mean = np.array([41.676228, 43.959198, 21.325665], dtype='float32')
  pca_std  = np.array([40.373974, 38.43527 , 25.48677 ], dtype='float32')


  images_augmented = np.zeros_like(images)
  images_augmented = np.tile(images_augmented[:,np.newaxis], [1,N_augmentations,1,1,1])
  
  deltas = fancy_pca_get_deltas(len(images),N_augmentations,pca)
  images_augmented = np.reshape(images_augmented,(len(images),N_augmentations,-1,3))
  images_augmented = (images_augmented-pca_mean)/pca_std
  images_augmented = (images_augmented + deltas)*pca_std + pca_mean
  images_augmented = np.maximum(np.minimum(images_augmented, 255), 0)
  images_augmented = np.reshape(images_augmented,(len(images)*N_augmentations,256,256,3))
  images_augmented = np.round(images_augmented,0).astype('int16')
  return images_augmented


imgs = images[0:2,120:140,120:140]

imgs_aug = augment_images_fancy_pca(imgs,4)


for i in range(len(imgs_aug)):
    plt.imshow(imgs_aug[i])
    plt.show()

    
