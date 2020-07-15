import shap
import Inception
import torch
import BasicCNN


data_dir = '/zhome/ca/6/92701/Desktop/Master_Thesis/stylegan2/rl_images'
model_name = 'basiccnn'
data_augment = False
batch_size = 1
img_size = 256
enhanced = False


net = BasicCNN.Net_256()
cp = torch.load('/zhome/ca/6/92701/Desktop/Master_Thesis/Results/basiccnn_data-augment.pth.tar',map_location='cuda:0')

cp_new_dict = dict()
for key in cp['state_dict']:
	newkey=key[7:]
	cp_new_dict[newkey] = cp['state_dict'][key]
net.load_state_dict(cp_new_dict)


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

def map2layer(x):
    
    return net(x)

e = shap.GradientExplainer(
    (model.layers[7].input, model.layers[-1].output),
    map2layer(X, 7),
    local_smoothing=0 # std dev of smoothing noise
