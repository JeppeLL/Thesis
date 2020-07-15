#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 12:37:08 2020

@author: s140452
"""

from __future__ import print_function, division, absolute_import
import faulthandler; faulthandler.enable()
import torchvision
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import BasicCNN
import VGG
import Inception
import argparse
import sys
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


_valid_models = [
		'vgg',
		'inception',
		'basiccnn'
		]

_valid_optimizers = [
		'adamw',
		'rmsprop',
		'SGDR'
		]
		
_valid_imgsize = [
		256,
		1024
		]




#Fancy PCA data augmentation
class fancy_pca(object):
	def __init__(self):
		self.pca=np.array([[-1.5800763e+00, -1.5902529e+00, -1.4200467e+00],
		[-6.5569100e-03,  6.7732222e-03, -2.8922837e-04],
		[-1.3565156e-01, -1.1917505e-01,  2.8439787e-01]], dtype='float32')

		self.mean = np.array([41.676228, 43.959198, 21.325665], dtype='float32')
		self.std  = np.array([40.373974, 38.43527 , 25.48677 ], dtype='float32')
		
	def __call__(self,img):
		
		img = np.array(img)
		img_rs = img.reshape(-1, 3)
		# center mean
		img_centered = (img_rs - self.mean)/self.std

		###get add_vect / deltas:
		alpha=np.random.normal(0,0.1,3) #batch_size=32
		delta=np.matmul(alpha,self.pca)    
		delta=np.tile(delta,[1,256*256]) #image_pixels=256*256
		add_vect=np.reshape(delta,[-1,3])

		orig_img = (img_centered+add_vect)*self.std + self.mean
		# was easier than integers between 0-255
		orig_img = np.clip(orig_img, 0.0, 255.0)

		orig_img = orig_img.astype(np.uint8)
		orig_img = orig_img.reshape(256,256,3)
		return Image.fromarray(orig_img)


"""
#Define cyclical learning rate scheduler			
def cyclical_lr(step_sz, min_lr=0.001, max_lr=1, mode='triangular', scale_func=None, scale_md='cycles', gamma=1.):
    if scale_func == None:
        if mode == 'triangular':
            scale_fn = lambda x: 1.
            scale_mode = 'cycles'
        elif mode == 'triangular2':
            scale_fn = lambda x: 1 / (2.**(x - 1))
            scale_mode = 'cycles'
        elif mode == 'exp_range':
            scale_fn = lambda x: gamma**(x)
            scale_mode = 'iterations'
        else:
            raise ValueError(f'The {mode} is not valid value!')
    else:
        scale_fn = scale_func
        scale_mode = scale_md
    lr_lambda = lambda iters: min_lr + (max_lr - min_lr) * rel_val(iters, step_sz, scale_mode)
    def rel_val(iteration, stepsize, mode):
        cycle = math.floor(1 + iteration / (2 * stepsize))
        x = abs(iteration / stepsize - 2 * cycle + 1)
        if mode == 'cycles':
            return max(0, (1 - x)) * scale_fn(cycle)
        elif mode == 'iterations':
            return max(0, (1 - x)) * scale_fn(iteration)
        else:
            raise ValueError(f'The {scale_mode} is not valid value!')
    return lr_lambda
"""




#Loader object for loading training images
def train_loader(data_dir, model_name, data_augment, batch_size, img_size, enhanced):
	if enhanced:
		data_path = "/".join((data_dir,"Enhanced 256/Training/"))
	else:			
		data_path = "/".join((data_dir,str(img_size),"Training/"))
		
	trans = [torchvision.transforms.ToTensor()]
	if img_size == 256:
		if model_name == "inception":
			trans.insert(0, torchvision.transforms.Pad((21,22,22,21)))
		elif model_name == 'vgg':
			trans.insert(0, torchvision.transforms.Resize((244,244)))
	if data_augment:
		trans.insert(0, fancy_pca())
		trans.insert(0, torchvision.transforms.RandomRotation(180))
		trans.insert(0, torchvision.transforms.RandomHorizontalFlip(p=0.5))

	train_dataset = torchvision.datasets.ImageFolder(
			root=data_path,
			transform=torchvision.transforms.Compose(trans)
		)

	train_loader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=batch_size,
		num_workers=0,
		shuffle=True
	)
	return train_loader


#Loader object for loading validation images
def test_loader(data_dir, model_name, batch_size, img_size):
	data_path = "/zhome/ca/6/92701/Desktop/Master_Thesis/stylegan2/rl_images/256_Train_upsample_only/Validation"
	
	trans = [torchvision.transforms.ToTensor()]
	if img_size == 256:
		if model_name == "inception":
			trans.insert(0, torchvision.transforms.Pad((21,22,22,21)))
		elif model_name == 'vgg':
			trans.insert(0, torchvision.transforms.Resize((244,244)))
	test_dataset = torchvision.datasets.ImageFolder(
		root=data_path,
		transform=torchvision.transforms.Compose(trans)
	)
	
	test_loader = torch.utils.data.DataLoader(
		test_dataset,
		batch_size=batch_size,
		num_workers=0,
		shuffle=True
	)
	return test_loader


#Check if model has lower loss than previous best
def save_checkpoint(state, old_loss, loss, is_best, filename=None):
	"""Save checkpoint if a new best is achieved"""
	
	if is_best:
		print (f"=> Saving a new best loss improved from {round(old_loss,5)} to {round(loss,5)}")
	else:
		print ("Validation Accuracy did not improve")
	torch.save(state, filename)  # save checkpoint


#Initialize wheights for linear and conv2d layers
def init_weights(m):
	if type(m) == nn.Linear or type(m) == nn.Conv2d:
		torch.nn.init.xavier_uniform_(m.weight)
		m.bias.data.fill_(0.01)
		

#Build model based on argument
def create_net(model_name, img_size):

	if model_name == 'vgg':
		net = VGG.vgg16_bn(img_size=img_size)
	elif model_name == 'inception':
		net = Inception.inception_v3()
	else:
		if img_size==256:
			net = BasicCNN.Net_256()
			#net = BasicCNN.Net_256pp(split_size=200)
		else:
			net = BasicCNN.Net_1024()
		net.apply(init_weights)
		
	net = nn.DataParallel(net)
	
	return net.to(device)
	

#Run training based on given parameters		
#def run_training(data_dir, model_name, num_epochs, data_augment, batch_size, optimizer, learning_rate, load_name, img_size, new_optim, smoothing, cwd):
def run_training(**args):
	
	data_dir = args.get("data_dir")
	model_name = args.get("model_name")
	num_epochs = args.get("num_epochs")
	data_augment = args.get("data_augment")
	batch_size = args.get("batch_size")
	optimizer = args.get("optimizer")
	learning_rate = args.get("learning_rate")
	load_name = args.get("load_name")
	img_size = args.get("img_size")
	new_optim = args.get("new_optim")
	smoothing = args.get("smoothing")
	cwd = args.get("cwd")
	enhanced = args.get("enhanced")
	save_add = args.get("save_add")
	
	if enhanced:
		data_path = "/".join((data_dir,"Enhanced 256/Training/"))
	else:
		data_path = "/".join((data_dir,str(img_size),"Training/"))
	print("Using %d images for training" % (len(os.listdir(data_path+"/rejected"))*2))

	try:
		if enhanced == True:
			os.mkdir('Enhanced results')
		else:
			os.mkdir('Results')
			
	except Exception:
		pass
	
	if enhanced:
		cwd = cwd + 'Enhanced results/'
	else:
		cwd = cwd + 'Results/'

	if img_size == 1024:
		cwd = "".join((cwd, "1024/"))
		
	
	print("Building model %s \n" % model_name)
	net = create_net(model_name, img_size)
	
	print("{} \n".format(net))
	criterion = nn.BCEWithLogitsLoss()
	
	if optimizer == "adamw":
		if model_name == 'inception':
			optim = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-3)
		else:
			optim = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-3)
	elif optimizer == "rmsprop":
		optim = torch.optim.RMSprop(net.parameters(), lr=learning_rate, momentum = 0.9, weight_decay = 0.1, eps = 1.0)
	elif optimizer == "SGDR":
		optim = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum = 0.9, weight_decay = 5e-3, nesterov=True)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2)

	#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=20,factor=0.5)
	
	#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 10)
	#scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.95)
	#scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.16, -1)
	if model_name == 'basiccnn':
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [4000], gamma=0.1)
	else:
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,[4000], gamma=0.1)
		
	#Cyclical learning rate
	#clr = cyclical_lr(step_size, min_lr=0.001, max_lr=1, mode='triangular2')
	#scheduler = lr_scheduler.LambdaLR(optim, [clr])
	#scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2)

	
	print("Data augmentation = %s \nBatch_size %s \nUsing %s \nLearning rate = %.5f \nScheduler: %s" % (data_augment, batch_size, optimizer, learning_rate, scheduler.state_dict()['milestones'].keys()))
	
	start_epoch = 0
	train_loss = []
	test_loss = []
	tmp_img = "tmp_ae_out.png"
	timings = []

	
	#Load model if any is given
	if load_name != None:
		print("Loading model from %s" % load_name)
		cp = torch.load(load_name)
		start_epoch = cp['epoch']
		scheduler.load_state_dict(cp['scheduler'])
		net.load_state_dict(cp['state_dict'])
		train_loss = cp['train_loss']
		test_loss = cp['test_loss']
		best_loss = cp['best_loss']
		num_images = cp['num_images']
		best_state_dict = cp['best_state_dict']
		
		if new_optim == False:				
			optim.load_state_dict(cp['optim'])
		elif new_optim == True:
			if optimizer == "adamw":
				optim = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-3)
				
			elif optimizer == "rmsprop":
				optim = torch.optim.RMSprop(net.parameters(), lr=learning_rate, momentum = 0.9, weight_decay = 0.9, eps = 1.0)
				
			elif optimizer == "SGDR":
				optim = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum = 0.9, weight_decay = 5e-3, nesterov=True)
				
			#scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2)


	if start_epoch >= num_epochs:
		print("Model has already trained for %d epochs. Increase num_epochs to continue training" % start_epoch)
		sys.exit(1)

	print("Training for %d epochs \n" % num_epochs)
	epoch = 0 + start_epoch
	while epoch < num_epochs:
		print(optim.param_groups[0]['lr'])
		start_time = time.time()
		net.train()
		batch_loss = []
		running_loss = 0.0
		for i, data in enumerate(train_loader(data_dir, model_name, data_augment=data_augment, batch_size=batch_size, img_size=img_size, enhanced=enhanced)):
			inputs, labels = data
			inputs = inputs.to(device)
			labels = labels.to(device)

			#labels = (1-labels)*smoothing+labels*(1-smoothing)

			# zero the parameter gradients
			optim.zero_grad()

			# forward + backward + optimize
			if model_name == "inception":
							
				outputs, aux_outputs = net(inputs).values()
				outputs = torch.squeeze(outputs)
				aux_outputs = torch.squeeze(aux_outputs)
				labels = labels.type_as(outputs)
				loss1 = criterion(outputs, labels)
				loss2 = criterion(aux_outputs, labels)
				loss = loss1 + 0.4*loss2
				loss.backward()
				
				#if optimizer == 'rmsprop':
					#torch.nn.utils.clip_grad_norm(net.parameters(), 2.0)
			else:
				outputs = net(inputs)
				outputs = torch.squeeze(outputs)
				labels = labels.to(outputs.device)
				labels = labels.type_as(outputs)
				loss = criterion(outputs, labels)
				loss.backward()
			optim.step()

			# print statistics
			running_loss += loss.item()
			batch_loss.append(loss.item())
		#else:	
		train_loss.append(np.mean(batch_loss))

		with torch.no_grad():
			batch_loss = []
			pred = []
			agg_labels = []
			net.eval()
			for i, data in enumerate(test_loader(data_dir, model_name=model_name, batch_size=batch_size, img_size=img_size)):
				images, labels = data
				images = images.to(device)
				#labels = (1-labels)*smoothing+labels*(1-smoothing)
				labels = labels.to(device)
				labels = labels.type_as(outputs)
				outputs = net(images)
				outputs = torch.squeeze(outputs)
				loss = criterion(outputs, labels)
				batch_loss.append(loss.item())
			test_loss.append(np.mean(batch_loss))
			
		scheduler.step()
		#scheduler.step(np.mean(batch_loss))


		if epoch + 1 == 1:
			best_loss = test_loss[epoch]
			best_state_dict = net.state_dict()
			epoch += 1
			continue

		# Get bool not ByteTensors
		is_best = bool(test_loss[epoch] < best_loss)
		if is_best:
			old_loss = best_loss
			best_state_dict = net.state_dict()
		else:
			old_loss = 0
		 
		# Get greater Tensor to keep track best loss
		best_loss = min(test_loss[epoch], best_loss)

		# Time the epoch
		timer = time.time() - start_time
		timings.append(timer)
		print(f"Epoch {epoch} took {timer} seconds to run")
		
		
		
		# Save checkpoint if is a new best
		if save_add != None:
			fname="".join((cwd,model_name, save_add, '.pth.tar'))
		elif data_augment:
			fname="".join((cwd,model_name,'_data-augment.pth.tar'))
		else:
			fname="".join((cwd,model_name,'.pth.tar'))
		save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': net.state_dict(),
			'best_state_dict': best_state_dict,
			'best_loss': best_loss,
			'time': timer,
			'test_loss': test_loss,
			'train_loss': train_loss,
			'optim': optim.state_dict(),
			'scheduler': scheduler.state_dict(),
			'num_images': epoch*len(train_loader(data_dir, model_name, data_augment=data_augment, batch_size=batch_size, img_size=img_size, enhanced=enhanced))*batch_size
		}, old_loss, best_loss, is_best, filename=fname)
		
		num_images = epoch*len(train_loader(data_dir, model_name, data_augment=data_augment, batch_size=batch_size, img_size=img_size, enhanced=enhanced))*batch_size
		print("Trained on %d images" % num_images)
		print(f"Training loss: {train_loss[-1]}")
		print(f"Validation loss: {test_loss[-1]}\n")
		if epoch == 0:
			continue

		# -- Plotting --
		f, ax1 = plt.subplots(figsize=(8,8))

		#Loss
		ax1.set_title("Loss")
		ax1.set_xlabel('Epoch')
		ax1.set_ylabel('Loss')


		ax1.plot(np.arange(epoch+1), train_loss, color="blue", linestyle="-.")
		ax1.plot(np.arange(epoch+1), test_loss, color="red", linestyle="-.")
		ax1.legend(['Training','Validation'])
		plt.tight_layout()
		
		
		
		if data_augment:
			f.savefig("".join((cwd,model_name,'_', save_add,'_data-augment.png')))
		else:
			f.savefig("".join((cwd,model_name,'.png')))
		plt.close(f)
		
		epoch += 1

#-----------------------------------------------------------------------------------------

def _str_to_bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')
#-----------------------------------------------------------------------------------------		
def main():
	parser = argparse.ArgumentParser(
		description='Train model.',
		formatter_class=argparse.RawDescriptionHelpFormatter
	)
	parser.add_argument('--data-dir', help='Path to folder that contains training subfolder and validation subfolder', required=True) 
	parser.add_argument('--model-name', '-M', help='Name of model to train (default: %(default)s)', default='basiccnn')
	parser.add_argument('--num-epochs', '-N', help='Number of epochs to train (default: %(default)s)', default=1000, type=int)
	parser.add_argument('--data-augment', '-D', help='Use data augmentation during training (default: %(default)s)', default=True, type=_str_to_bool)
	parser.add_argument('--batch-size', '-B', help='Number of images per batch (default: %(default)s)', default=256, type=int)
	parser.add_argument('--optimizer', '-O', help='Which optimizer to use (default: %(default)s)', default='adamw')
	parser.add_argument('--learning-rate', '-lr', help='Set learning rate (default: %(default)s)', default=1e-4, type=float)
	parser.add_argument('--load-name', '-L', help='Path to file used to resume training', default=None)
	parser.add_argument('--img-size', '-I', help='Size of images to use in training (256x256 or 1024x1024)', default=256, type=int)
	parser.add_argument('--new-optim', help='Which new optimizer to use when continuing training', default=False, type=_str_to_bool)
	parser.add_argument('--smoothing', help='How much to smoothing target labels (between 0 and 1)', default=0, type=float)
	parser.add_argument('--enhanced', help='Boolean indication to use synthetic images in training', default=False, type=_str_to_bool)
	parser.add_argument('--save_add', '-s', help='Add to the save name', default="", type=str)
	
	
	args = parser.parse_args()
	
	#check smoothing constant is viable
	if not 0 <= args.smoothing <= 1:
		print('Smoothing must be a value between 0 and 1, choose a different value')
		sys.exit(1)
	
	#Check if data-dir path exists
	if not os.path.exists(args.data_dir):
		print('Data directory does not exist')
		sys.exit(1)
	
	#Check if img_size is valid
	if not args.img_size in _valid_imgsize:
		print('Error: --img-size must be one of: ', ', '.join(_valid_imgsize))
		sys.exit(1)

	#Check model is loaded if new-optim not None
	if args.load_name == None and args.new_optim != False:
		print('Model must be loaded to start new optimizer. Use optimizer-flag instead')
		sys.exit(1)
	
	# Check if model_name valid
	if args.model_name not in _valid_models:
		print ('Error: --model-name value must be one of: ', ', '.join(_valid_models))
		sys.exit(1)  
		  
	# Check if optimizer valid
	if args.optimizer not in _valid_optimizers:
		print ('Error: --optimizer value must be one of: ', ', '.join(_valid_optimizers))
		sys.exit(1)	
		
	
	# Check if it is a file
	if not args.load_name==None:
		assert os.path.isfile(args.load_name)
		
	cwd = "".join((os.getcwd(),'/'))

	run_training(cwd=cwd,**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
	main()

#----------------------------------------------------------------------------
