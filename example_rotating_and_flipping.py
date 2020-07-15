# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:30:08 2020

@author: lauri
"""

import PIL

img_00=PIL.Image.open("C:/Users/lauri/Desktop/DataPreprocessing/Data/Images_256/Training_rare_examples/stor bakterie/reject_58b8fa28-97e2-4446-aa45-0508a0b1ba4c_4_rl.jpg")
img_01=img_00.rotate(45)
img_10=img_00.transpose(PIL.Image.FLIP_TOP_BOTTOM)
img_11=img_01.transpose(PIL.Image.FLIP_TOP_BOTTOM)

import matplotlib.pyplot as plt

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
