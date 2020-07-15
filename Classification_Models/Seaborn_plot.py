import seaborn as sns
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
import os



def plot_seaborn(file_name, start_epoch, end_epoch):
	size_factor = 1.3
	sns.set(rc={'figure.figsize':(11.7*size_factor,8.27*size_factor)})
	sns.set_style("white")
	sns.set_style("ticks")

	
	print("Loading model results...", end="")
	file_name_splits = file_name.split("/")[-1]
	model_name = file_name_splits.split(".")[0]
	save_name = "/zhome/ca/6/92701/Desktop/Master_Thesis/Snsplots/sns_plot_"+model_name+".png"
	cp = torch.load(file_name)
	
	last_epoch = cp['epoch']
	train_loss = cp['train_loss']
	test_loss = cp['test_loss']
	best_loss = cp['best_loss']
	num_images = cp['num_images']
	
	
	if end_epoch > last_epoch:
		print("End epoch can't be greater than the number of epochs run, choose a number lower or equal to %s" % last_epoch)
		sys.exit(1)
	
	if end_epoch == -1:
		end_epoch = last_epoch
	
	epochs = [i for i in range(start_epoch, end_epoch)]
	train_loss = train_loss[start_epoch:end_epoch]
	test_loss = test_loss[start_epoch:end_epoch]
	
	
	
	d = {'epochs':epochs, 'Train':train_loss, 'Validation':test_loss}
	df = pd.DataFrame(d)
	df = df.melt('epochs', var_name='cols', value_name='vals')
	
	print("Done\nPlotting results...", end="")
	
	palette = sns.color_palette("BrBG",2)

	sns_plot = sns.lineplot(x="epochs", y="vals",
				 hue="cols",
				 palette=palette,
				 size="cols",
				 sizes=(3., 3.),
				 data=df)
	sns_plot.set_xlabel("Epochs",fontsize=18)
	sns_plot.set_ylabel("Binary CrossEntropy Loss",fontsize=18)


	
	handles, labels = sns_plot.get_legend_handles_labels()
	sns_plot.legend(handles=handles[1:], labels=labels[1:])
	
	plt.setp(sns_plot.get_legend().get_texts(), fontsize='15')


	fig = sns_plot.get_figure()
	fig.savefig(save_name)
	print("Done")
	



#-----------------------------------------------------------------------------------------		
def main():
	
	parser = argparse.ArgumentParser(
		description='Plot loss using Seaborn',
		formatter_class=argparse.RawDescriptionHelpFormatter
	)
	parser.add_argument('--file-name', help='Path to network file', required=True, type=str)
	parser.add_argument('--start-epoch', help='Epoch to from where to start', default=1, type=int)
	parser.add_argument('--end-epoch', help='Epoch to end plot', default=-1, type=int)
	

	args = parser.parse_args()		
	
	# Check if it is a file
	if not args.file_name==None:
		assert os.path.isfile(args.file_name)
		

	plot_seaborn(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
	main()

#----------------------------------------------------------------------------
