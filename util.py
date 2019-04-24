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

# collate fn lets you control the return value of each batch
# for packed_seqs, you want to return your data sorted by length
def collate_lines_for_test(seq_list):
    inputs, targets = zip(*seq_list)
    lens = [len(sequence) for sequence in inputs]
    # N, L, E
    # sort by length
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    ordered_inputs = inputs[seq_order]
    ordered_targets = targets[seq_order]
    ordered_seq_lens = torch.tensor([lens[i] for i in seq_order])
    reverse_order = sorted(range(len(lens)), key=seq_order.__getitem__, reverse=False)
    return ordered_inputs, ordered_targets, ordered_seq_lens, seq_order, reverse_order