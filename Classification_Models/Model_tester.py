#!/usr/bin/env python
# coding: utf-8


from __future__ import print_function, division, absolute_import
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, display, clear_output
import os
from sklearn.metrics import average_precision_score
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch.optim as optim
import pickle
import time
from collections import OrderedDict
import VGG
import Inception
import BasicCNN
from shutil import copy
import argparse
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_pretrained_model(types, file_name):
	e = 0.00001
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	Total = 0
	
	if list(file_name.split("/")[-1])[0] == 'b':
		model_type = 'b'
		model = BasicCNN.Net_256()
	elif list(file_name.split("/")[-1])[0] == 'i':
		model_type = 'i'
		model = Inception.inception_v3()
	
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	batch_size = 1

	def test_loader():
		data_path = "/zhome/ca/6/92701/Desktop/Master_Thesis/stylegan2/rl_images/256_Train_upsample_only/"+types
		if model_type == 'i':
			test_dataset = torchvision.datasets.ImageFolder(
				root=data_path,
				transform=torchvision.transforms.Compose([
					#torchvision.transforms.RandomRotation(180),
					#torchvision.transforms.RandomHorizontalFlip(p=0.5),
					torchvision.transforms.Pad((21, 22, 22, 21)),
					torchvision.transforms.ToTensor()])
			)
			test_loader = torch.utils.data.DataLoader(
				test_dataset,
				batch_size=batch_size,
				num_workers=0,
				shuffle=False
			)
		else:
			test_dataset = torchvision.datasets.ImageFolder(
				root=data_path,
				transform=torchvision.transforms.Compose([
					#torchvision.transforms.RandomRotation(180),
					#torchvision.transforms.RandomHorizontalFlip(p=0.5),
					#torchvision.transforms.Pad((21, 22, 22, 21)),
					torchvision.transforms.ToTensor()])
			)
			test_loader = torch.utils.data.DataLoader(
				test_dataset,
				batch_size=batch_size,
				num_workers=0,
				shuffle=False
			)
		return test_loader
	#model = nn.DataParallel(model)
	net = model.to(device)
	
	#net = model
	
	cp = torch.load(file_name,map_location='cuda:0')
	#net.load_state_dict(cp['state_dict'])
	
	cp_new_dict = dict()
	for key in cp['state_dict']:
		newkey=key[7:]
		cp_new_dict[newkey] = cp['state_dict'][key]
	net.load_state_dict(cp_new_dict)
	"""
	cp = torch.load(file_name,map_location='cpu')
	best_loss = cp['best_loss']
	curr_loss = cp['test_loss']
	#print(curr_loss)
	#print(best_loss)
	#"""
	#net.load_state_dict(cp['best_state_dict'])



	Num_test_pics = len(test_loader())
	net.eval()
	error_list = list()
	correct_list = list()
	for i, (test_images, test_labels) in enumerate(test_loader(),0):
		loader = test_loader()
		test_images = test_images.to('cuda:0')
		#test_images = test_images.to(device)
		#test_labels = test_labels.to(device)
		out = net(test_images).detach().cpu()
		pred = torch.sigmoid(out)
		pred = (pred>0.00222).float().numpy()
		truth = test_labels.numpy()[0]
		#truth = test_labels.detach().cpu().numpy()[0]
		sample_fname, _ = loader.dataset.samples[i]
		if pred == 1 and truth == 1:
			TP += 1
			correct_list.append(" - ".join(("rejected",sample_fname.rsplit('/', 1)[-1])))
		elif pred ==0 and truth == 0:
			TN += 1
		elif pred == 1 and truth == 0:
			FP += 1
			#error_list.append(" - ".join(("accepted",sample_fname.rsplit('/', 1)[-1])))
		elif pred == 0 and truth == 1:
			FN += 1
			#error_list.append(" - ".join(("rejected",sample_fname.rsplit('/', 1)[-1])))
		Total += 1


		
		if Total == Num_test_pics:
			clear_output(wait=True)
			print(f"Positive images so far: {TP+FN}")
			print(f"Negative images so far: {TN+FP}")
			print(f"Tested {Total} images out of {Num_test_pics}!\n")
			print(f"Accuracy on positive images is {round(TP/(TP+FN+e)*100,2)}%")
			print(f"Accuracy on negative images is {round(TN/(TN+FP+e)*100,2)}%\n")
			print(f"F2-Score: {round((5 * TP/(TP+FP+e) * TP/(TP+FN+e)) / (4 * TP/(TP+FP+e) + TP/(TP+FN+e)+e),2)}")
			print(f"Recall: {round(TP/(TP+FN+e)*100,2)}%")
			print(f"Precision: {round(TP/(TP+FP+e)*100,2)}%")
			print(f"False-Negative fraction: {round(FN/(TP+FN+e)*100,2)}%")
			print(f"Total Accuracy is {round((TP+TN)/(Total+e)*100,2)}%\n")
			print(f"True Positives: {TP}")
			print(f"False Positives: {FP}")
			print(f"True Negatives: {TN}")
			print(f"False Negatives: {FN}")
	#for i in error_list:
	#	print(i)
	"""
	with open('rejected_errors.pkl', 'wb') as f:
		pickle.dump(error_list, f)
	with open('rejected_correct.pkl', 'wb') as f:
		pickle.dump(correct_list, f)
	"""
	"""
	files = glob.glob('/zhome/ca/6/92701/Desktop/Master_Thesis/results/Enhanced FN/*')
	for f in files:
		os.remove(f)

	files = glob.glob('/zhome/ca/6/92701/Desktop/Master_Thesis/results/TP/*')
	for f in files:
		os.remove(f)
	
	for img in error_list:
		copy("/".join(('/zhome/ca/6/92701/Desktop/Master_Thesis/stylegan2/rl_images/Enhanced 256/Validation/rejected',img[11:])), '/zhome/ca/6/92701/Desktop/Master_Thesis/results_lime/Enhanced FN')
	#for img in correct_list:
	#	copy("/".join(('/zhome/ca/6/92701/Desktop/Master_Thesis/stylegan2/rl_images/256/Validation/rejected',img[11:])), '/zhome/ca/6/92701/Desktop/Master_Thesis/results/TP')
"""
#-----------------------------------------------------------------------------------------		
def main():
	parser = argparse.ArgumentParser(
		description='Train model.',
		formatter_class=argparse.RawDescriptionHelpFormatter
	)
	
	parser.add_argument('--types', '-t', help='Test or Validation set', default='Validation', required=True)
	parser.add_argument('--file-name', '-f', help='Path to file containing model', required=True)

	
	
	args = parser.parse_args()
		



	test_pretrained_model(**vars(args))
#----------------------------------------------------------------------------

if __name__ == "__main__":
	main()

#----------------------------------------------------------------------------



