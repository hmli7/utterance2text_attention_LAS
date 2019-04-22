import torch
import shutil
import os
import time


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