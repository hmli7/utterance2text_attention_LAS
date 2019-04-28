import torch
import shutil
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def save_checkpoint(state, is_best, output_path, filename='checkpoint.pth.tar'):
	torch.save(state, os.path.join(output_path, filename))
	if is_best:
		shutil.copyfile(os.path.join(output_path, filename), os.path.join(output_path, 'model_best.pth.tar'))
		

def print_file_and_screen(*inputs, f):
	for input in inputs:
		print(input, end=' ')
		print(input, end=' ', file=f)
	print()
	print(file=f)
	

def wrap_up_experiment(metrics_file_path):
	with open(metrics_file_path, 'a') as f:
		print('### Experiment finished at', time.asctime( time.localtime(time.time()) ), file=f)
		print('-'*40, file=f)

# collate fn lets you control the return value of each batch
# for packed_seqs, you want to return your data sorted by length
def sort_instances(seq_list):
	if len(seq_list) == 1:
		# testing
		inputs = seq_list
		targets = None
	else:
		inputs, targets = seq_list
	
	if type(inputs) == torch.Tensor and len(inputs.size()) == 2: # batch size == 1
		inputs = [inputs]
		if targets is not None:
			targets = [targets]
#         return inputs, targets, torch.tensor([len(inputs)]), [0], [0]
	lens = [len(sequence) for sequence in inputs]
	# N, L, E
	# sort by length
	seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
	ordered_inputs = [inputs[i] for i in seq_order]
	if targets is not None:
		ordered_targets = [targets[i] for i in seq_order]
	ordered_seq_lens = torch.tensor([lens[i] for i in seq_order])
	reverse_order = sorted(range(len(lens)), key=seq_order.__getitem__, reverse=False)

	if targets is not None:
		return ordered_inputs, ordered_targets, ordered_seq_lens, seq_order, reverse_order
	else:
		return ordered_inputs, ordered_seq_lens, seq_order, reverse_order

def plot_grad_flow(named_parameters, epoch, path):
	'''Plots the gradients flowing through different layers in the net during training.
	Can be used for checking for possible gradient vanishing / exploding problems.
	
	Usage: Plug this function in Trainer class after loss.backwards() as 
	"plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
	ave_grads = []
	max_grads= []
	layers = []
	for n, p in named_parameters:
		if(p.requires_grad) and ("bias" not in n):
			layers.append(n)
			ave_grads.append(p.grad.abs().mean())
			max_grads.append(p.grad.abs().max())
	fig = plt.figure()
	plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
	plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
	plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
	plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
	plt.xlim(left=0, right=len(ave_grads))
	plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
	plt.xlabel("Layers")
	plt.ylabel("average gradient")
	plt.title("Gradient flow")
	plt.grid(True)
	plt.legend([Line2D([0], [0], color="c", lw=4),
				Line2D([0], [0], color="b", lw=4),
				Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
	fig.savefig(os.path.join(path,"gradient_epoch%d-%d.png") % (epoch, time.time()), bbox_inches = 'tight')
	plt.close()
	
def plot_grad_flow_simple(named_parameters, epoch, path):
	ave_grads = []
	layers = []
	for n, p in named_parameters:
		if(p.requires_grad) and ("bias" not in n):
			layers.append(n)
			ave_grads.append(p.grad.abs().mean())
	fig = plt.figure()
	plt.plot(ave_grads, alpha=0.3, color="b")
	plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
	plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
	plt.xlim(xmin=0, xmax=len(ave_grads))
	plt.xlabel("Layers")
	plt.ylabel("average gradient")
	plt.title("Gradient flow")
	plt.grid(True)
	fig.savefig(os.path.join(path,"gradient_epoch%d-%d.png") % (epoch, time.time()), bbox_inches = 'tight')
	plt.close()
	
def plot_single_attention(attention_weights, epoch, path):
	fig = plt.figure()
#     plt.imshow(attention_weights)
	ax = fig.add_subplot(111)
	cax = ax.matshow(attention_weights, cmap='bone')
	fig.colorbar(cax)
	fig.savefig(os.path.join(path,"attention_epoch%d-%d.png") % (epoch, time.time()), bbox_inches = 'tight')
	plt.close()

def plot_single_attention_return(attention_weights):
	fig = plt.figure()
#     plt.imshow(attention_weights)
	ax = fig.add_subplot(111)
	cax = ax.matshow(attention_weights, cmap='bone')
	fig.colorbar(cax)
	return fig

def plot_grad_flow_simple_return(named_parameters):
	ave_grads = []
	layers = []
	for n, p in named_parameters:
		if(p.requires_grad) and ("bias" not in n):
			layers.append(n)
			ave_grads.append(p.grad.abs().mean())
	fig = plt.figure(figsize=(8, 14), frameon=False, dpi=100)
	plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
	plt.plot(ave_grads, alpha=0.3, color="b")
	plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
	plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
	plt.xlim(xmin=0, xmax=len(ave_grads))
	plt.xlabel("Layers")
	plt.ylabel("average gradient")
	plt.title("Gradient flow")
	plt.grid(True)
	return fig

def plot_grad_flow_return(named_parameters):
	'''Plots the gradients flowing through different layers in the net during training.
	Can be used for checking for possible gradient vanishing / exploding problems.
	
	Usage: Plug this function in Trainer class after loss.backwards() as 
	"plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
	ave_grads = []
	max_grads= []
	layers = []
	for n, p in named_parameters:
		if(p.requires_grad) and ("bias" not in n):
			layers.append(n)
			ave_grads.append(p.grad.abs().mean())
			max_grads.append(p.grad.abs().max())
	fig = plt.figure(figsize=(8, 14), frameon=False, dpi=100)
	plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
	plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
	plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
	plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
	plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
	plt.xlim(left=0, right=len(ave_grads))
	plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
	plt.xlabel("Layers")
	plt.ylabel("average gradient")
	plt.title("Gradient flow")
	plt.grid(True)
	plt.legend([Line2D([0], [0], color="c", lw=4),
				Line2D([0], [0], color="b", lw=4),
				Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
	return fig