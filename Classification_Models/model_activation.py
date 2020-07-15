import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json
import sys
import Inception_old
import Inception
import imageio
import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
import pickle as p
from lime import lime_image
from skimage.segmentation import mark_boundaries
import glob
from lime.wrappers.scikit_image import SegmentationAlgorithm
import argparse


_valid_tp_fn = [
		'tp',
		'fn'
		]

_valid_path_pkl = [
		'path',
		'pkl'
		]




def run_segment_activation(**args):
	
	#Set parameters for script
	sp = args.get("super_pixels")
	num_samples = args.get("num_samples")
	path_or_pkl = args.get("path_pkl").lower()
	tp_or_fn = args.get("tp_fn").upper()
	pth = f'/zhome/ca/6/92701/Desktop/Master_Thesis/Results/Lime/Good Performance/'	
	
	
	
	#Get list of image names to use
	if path_or_pkl == 'path':
		path = f"{pth}/*.jpg"
		images = glob.glob(path)
	elif path_or_pkl == 'pkl':
		images = p.load(open('rejected_errors.pkl','rb'))
		for i, img in enumerate(images):
			images[i] = "/".join(("./stylegan2/rl_images/256/Validation/rejected/",img[11:]))
									
	print(f"Loaded {len(images)} images")
	print(f"Using {sp} super pixels and loading from {path_or_pkl.lower()}")
	def get_image(path):
		with open(os.path.abspath(path), 'rb') as f:
			with Image.open(f) as img:
				return img.convert('RGB')



	#Load the model	
	model = Inception.inception_v3()
	cp = torch.load('/zhome/ca/6/92701/Desktop/Master_Thesis/Results/Inception/First/inception_data-augment.pth.tar')

	state_dict = cp['state_dict']

	from collections import OrderedDict
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:] 
		new_state_dict[name] = v

	model.load_state_dict(new_state_dict)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model.eval()
	model.to(device)



	#Transformations in PIL image
	def get_PIL_transform():
		transf = transforms.Compose([
			transforms.Pad((21, 22, 22, 21))
		])
		
		return transf
		

	#Transformations in numpy image
	def get_preprocess_transform():
		normalize = transforms.Normalize(mean=[0.1446, 0.1561, 0.0794], std=[0.1223, 0.1178, 0.0936])
		transf = transforms.Compose([
			transforms.ToTensor(),
			normalize
		])
				
		return transf		

	pil_transf = get_PIL_transform()
	preprocess_transform = get_preprocess_transform()


	#Make batch and apply transforms
	def batch_predict(images):
		model.eval()

		batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)
		batch = batch.to(device)
		std = torch.tensor([0.1223, 0.1178, 0.0936])
		mean = torch.tensor([0.1446, 0.1561, 0.0794])
		logits = model(batch*std[None,:,None, None].to(device)+mean[None,:,None, None].to(device))
		
		probs0 = 1-torch.sigmoid(logits)
		probs1 = torch.sigmoid(logits)
		
		probs = torch.stack((probs0, probs1),dim=1).squeeze()

		return probs.detach().cpu().numpy()


	#Define segmentation alogrithm ("Quickshift")
	#segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
	#													max_dist=200, ratio=0.2,
	#													random_seed=41)
	segmentation_fn = SegmentationAlgorithm('slic', n_segments=99, compactness=2, sigma=3)


	#Where to save output
	save_dir = "/".join((os.getcwd(),"Boundry", "Handpicked", f"{sp}")) #f"{tp_or_fn}",
	
	try:
		os.mkdir(save_dir)
	except Exception:
		pass
	
	#Clear output folder
	files = glob.glob("/".join((save_dir, '*')))
	for f in files:
		os.remove(f)

	#Iterate over every image
	for idx, img_name in enumerate(reversed(images)):
		print("Running boundries on images %d of %d" % (idx+1, len(images)))
		img = get_image(img_name)
		explainer = lime_image.LimeImageExplainer()
		explanation = explainer.explain_instance(np.array(pil_transf(img)), 
												 batch_predict, 
												 top_labels=2, 
												 hide_color=0, 
												 num_samples=num_samples, 
												 segmentation_fn=segmentation_fn)

		img_name_split = img_name.split("/")
		file_name = img_name_split[-1]	
		
		temp, mask = explanation.get_image_and_mask(1, positive_only=False, num_features=sp, hide_rest=False)
		img_boundry = mark_boundaries(temp/255.0, mask)
		
		try:
			os.mkdir(save_dir)
			os.mkdir(save_dir)
		except Exception:
			pass
			
		fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

		ax[0].imshow(img)
		ax[1].imshow(img_boundry[21:277,22:278])


		for a in ax.ravel():
			a.set_axis_off()
			
		plt.tight_layout()
		plt.savefig("/".join((save_dir, file_name)))	

		
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
	
	parser.add_argument('--super-pixels', '-sp', help='Number of super-pixels to include', default=5, type=int)
	parser.add_argument('--num-samples', '-n', help='Number of samples to use for the regression', default=1000, type=int)
	parser.add_argument('--tp-fn', '-t', help='What type of images? True Positives of False Negatives', default='tp')
	parser.add_argument('--path-pkl', '-p', help='Reading from a folder or pkl file?', default='path')
	
	
	args = parser.parse_args()
	
	#Check super pixels
	if not 1 <= args.super_pixels:
		print('Super pixels must be greater or equal 1')
		sys.exit(1)
		
	#Check num samples
	if not 1 <= args.num_samples:
		print('num-samples must be greater or equal 1')
		sys.exit(1)
	
	#Check if tp-fn is valid
	if not args.tp_fn.lower() in _valid_tp_fn:
		print('Error: --tp-fn must be either tp or fn')
		sys.exit(1)

	#Check if path-pkl is valid
	if not args.path_pkl.lower() in _valid_path_pkl:
		print('Error: --path-pkl must be either path or pkl')
		sys.exit(1)		


	run_segment_activation(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
	main()

#----------------------------------------------------------------------------
