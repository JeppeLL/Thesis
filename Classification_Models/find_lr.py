from torch_lr_finder import LRFinder
import torchvision
import torch
import torch.nn as nn
import Inception
import numpy as np
from PIL import Image
import BasicCNN


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
		alpha=np.random.normal(0,0.1,3) 

		delta=np.matmul(alpha,self.pca)	
		delta=np.tile(delta,[1,256*256]) 
		add_vect=np.reshape(delta,[-1,3])
	
		orig_img = (img_centered+add_vect)*self.std + self.mean
		orig_img = np.clip(orig_img, 0.0, 255.0)
		
		orig_img = orig_img.astype(np.uint8)
		orig_img = orig_img.reshape(256,256,3)
		return Image.fromarray(orig_img)


#Loader object for loading training images
def train_loader(model_name=None):
	data_path = '/zhome/ca/6/92701/Desktop/Master_Thesis/stylegan2/rl_images/Enhanced 256/Training/'
	
	trans = [torchvision.transforms.ToTensor()]
	if model_name == "inception":
		trans.insert(0, torchvision.transforms.Pad((21,22,22,21)))
	elif model_name == 'vgg':
		trans.insert(0, torchvision.transforms.Resize((244,244)))
	trans.insert(0, fancy_pca())
	trans.insert(0, torchvision.transforms.RandomRotation(180))
	trans.insert(0, torchvision.transforms.RandomHorizontalFlip(p=0.5))

	train_dataset = torchvision.datasets.ImageFolder(
			root=data_path,
			transform=torchvision.transforms.Compose(trans)
		)

	train_loader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=256,
		num_workers=0,
		shuffle=True
	)
	return train_loader


trainloader = train_loader(model_name='inception')

model = Inception.inception_v3(img_size=256)
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum = 0.9, weight_decay = 5e-3, nesterov=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7, weight_decay=1e-4)
lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(trainloader, end_lr=1, num_iter=100)
lr_finder.plot() # to inspect the loss-learning rate graph
lr_finder.reset() # to reset the model and optimizer to their initial state
