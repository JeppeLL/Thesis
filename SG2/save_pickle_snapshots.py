# -*- coding: utf-8 -*-
"""
Created on Thu May  7 12:22:41 2020

@author: lauri
"""
import glob
from matplotlib import pyplot as plt
import pickle
import argparse
import numpy as np
import pandas as pd
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import os
import sys
from io import BytesIO
import IPython.display
import numpy as np
from math import ceil
from PIL import Image, ImageDraw
import imageio
import pretrained_networks


def generate_zs_from_seeds(seeds):
    zs = []
    for seed_idx, seed in enumerate(seeds):
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
        zs.append(z)
    return zs
def convertZtoW(latent, truncation_psi=0.7, truncation_cutoff=9):
    dlatent = Gs.components.mapping.run(latent, None) # [seed, layer, component]
    dlatent_avg = Gs.get_var('dlatent_avg') # [component]
    for i in range(truncation_cutoff):
        dlatent[0][i] = (dlatent[0][i]-dlatent_avg)*truncation_psi + dlatent_avg
    return dlatent
def generate_ws_from_seeds(seeds):
    zs = generate_zs_from_seeds(seeds)
    ws=list()
    for i,z in enumerate(zs):
        w=convertZtoW(z,truncation_psi=1.0)

        if i % 500 == 0:
            print(f"Progress: {i} of {len(zs)}")
            print(w.shape)
        ws.append(w)
    return ws

def generate_images_in_w_space(dlatents, truncation_psi=1.0):
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    Gs_kwargs.truncation_psi = truncation_psi
    #dlatent_avg = Gs.get_var('dlatent_avg') # [component]

    imgs = []
    
    for w_i in dlatents:
        imgs_w_i = []
        for quantile_j in w_i:
            #print(quantile_j)
            #print()
            #print()
            #print(len(quantile_j))
            #print()
            #print()
            #print()
            correct_shape = np.array([np.repeat([np.float32(quantile_j)],14,axis=0)])
            correct_shape = np.squeeze(correct_shape)
            
            row_images = Gs.components.synthesis.run(correct_shape,  **Gs_kwargs)
            imgs_w_i.append(Image.fromarray(row_images[0], 'RGB'))
        imgs.append(imgs_w_i)
    return imgs

def save_images_list(imgs_list,path="/home/novogan/master_thesis/stylegan2/generated_images/snapshots/"):    
    for imgs in imgs_list:
        pkl=-1
        for img in imgs:
            pkl+=1
            img.save(f"{path}img_{i}.jpg")

def save_images(imgs,iteration,path="/home/novogan/master_thesis/stylegan2/generated_images/snapshots/"):    
        seed_i=-1
        for img in imgs:
            seed_i+=1
            img[0].save(f"{path}{seed_i}_img_{iteration}.jpg")

if __name__=='__main__':
    
    #Fill in:
    save_path="/zhome/ca/6/92701/Desktop/Master_Thesis/stylegan2/Old_SG2_pkl/Images/"
    network_pickles = sorted(glob.glob(os.path.join("/zhome/ca/6/92701/Desktop/Master_Thesis/stylegan2/Old_SG2_pkl", "n*")))
    
    
    
    
    latest_pkl = network_pickles[-1]
    _G, _D, Gs = pretrained_networks.load_networks(latest_pkl)
    seeds=[1,2,3,4]
    ws=generate_ws_from_seeds(seeds)
    
    
    j=0
    for network_pkl in network_pickles:
        print('Loading networks from "%s"...' % network_pkl)
        _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
        imgs=generate_images_in_w_space(ws,truncation_psi=1.0)
    
        save_images(imgs,j,save_path)
        j+=1
