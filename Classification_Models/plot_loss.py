import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

"""
paths = ['/zhome/ca/6/92701/Desktop/Master_Thesis/Results_Inception/inception.pth.tar', '/zhome/ca/6/92701/Desktop/Master_Thesis/Results_Inception/inception_data-augment.pth.tar', '/zhome/ca/6/92701/Desktop/Master_Thesis/Results_BasicCNN/basiccnn.pth.tar', '/zhome/ca/6/92701/Desktop/Master_Thesis/Results_BasicCNN/basiccnn_data-augment.pth.tar']


for i in paths:
	cp = torch.load(i,map_location='cpu')

	train_loss = cp['train_loss']
	test_loss = cp['test_loss']
	best_loss = cp['best_loss']

	print(i)
	print(test_loss[-1])
	print(best_loss)

"""

#paths = ['/zhome/ca/6/92701/Desktop/Master_Thesis/Results/Inception/First/inception_downsample.pth.tar','/zhome/ca/6/92701/Desktop/Master_Thesis/Results/Inception/First/inception.pth.tar','/zhome/ca/6/92701/Desktop/Master_Thesis/Results/Inception/First/inception_data-augment.pth.tar']

import os
for i, file in enumerate(os.listdir("/zhome/ca/6/92701/Desktop/Master_Thesis/Results/")):
	if file.endswith(".pth.tar"):
		fname = file
		file = "/zhome/ca/6/92701/Desktop/Master_Thesis/Results/" + file
		cp = torch.load(file,map_location='cpu')

		train_loss = cp['train_loss']
		test_loss = cp['test_loss']
		best_loss = cp['best_loss']




		# -- Plotting --
		f, ax1 = plt.subplots(figsize=(8,8))

		#Loss
		ax1.set_title("Loss")
		ax1.set_xlabel('Epoch')
		ax1.set_ylabel('Loss')


		ax1.plot(train_loss[50:], color="blue", linestyle="-.")
		ax1.plot(test_loss[50:], color="red", linestyle="-.")
		ax1.legend(['Training','Validation'])
		plt.tight_layout()

		f.savefig(f'/zhome/ca/6/92701/Desktop/Master_Thesis/Results/{fname}.png')
		plt.close(f)

#"""
