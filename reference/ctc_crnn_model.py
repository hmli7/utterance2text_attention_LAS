import argparse
import csv
import os
import sys
import time

import numpy as np
import phoneme_list
import config
from util import *
import paths

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
import Levenshtein as L
from ctcdecode import CTCBeamDecoder
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
from warpctc_pytorch import CTCLoss

from functools import partial





class SpeechModel(nn.Module):

    def __init__(self,n_phonemes,num_rnn, hidden_size, nlayers, embed_size = 128):
        super(SpeechModel, self).__init__()
        self.n_phonemes=n_phonemes
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.nlayers=nlayers
        # cnn layer before lstm layers
        
        self.cnns = nn.Sequential(
            nn.Conv1d(in_channels=40, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU()
        )
        
        # the timestep size shrink to ceil(n/2)
        
        self.rnns = nn.ModuleList()
#         embedding size becomes 128 after the cnn layers
        self.rnns.append(nn.LSTM(input_size=embed_size, hidden_size=hidden_size,num_layers=nlayers, bidirectional=True))
        for i in range(num_rnn-1):
            self.rnns.append(nn.LSTM(input_size=hidden_size*2, hidden_size=hidden_size,num_layers=nlayers, bidirectional=True))
#         self.rnn = nn.LSTM(input_size = embed_size,hidden_size=hidden_size,num_layers=nlayers) # Recurrent network
        self.scoring = nn.Linear(hidden_size*2,n_phonemes+1) # Projection layer

    def forward(self, seq_batch):
       
        
        ## features u, p, 40
        batch_size = len(seq_batch)
        sequence_lens = [len(s) for s in seq_batch]
        # pad sequence
        padded_sequence = pad_sequence(seq_batch) # time batch embedding
        # permute for cnn
        padded_sequence = padded_sequence.permute(1, 2, 0) # batch embedding time
        embeddings = self.cnns(padded_sequence) # batch embedding time/2
        embeddings = embeddings.permute(2, 0, 1)
        sequence_lens_cnn = [np.ceil(length/2) for length in sequence_lens]
        # change pad lengths
        packed_sequence = pack_padded_sequence(embeddings,sequence_lens_cnn)

        h = packed_sequence
        for l in self.rnns:
            h, _ = l(h)
        output_padded, sequence_lens_rnn = pad_packed_sequence(h) # unpacked output (padded)
        scores_flatten = self.scoring(output_padded) 
        return scores_flatten, sequence_lens_rnn

class CTCCriterion(CTCLoss):
    def forward(self, prediction, target):
        acts = prediction[0]
        act_lens = prediction[1].int()
        label_lens = prediction[2].int()
        labels = (target + 1).view(-1).int()
#         print(acts.size(),act_lens.size(),label_lens.size(),labels.size())
        return super(CTCCriterion, self).forward(
            acts=acts,
            labels=labels.cpu(),
            act_lens=act_lens.cpu(),
            label_lens=label_lens.cpu()
        )


class ER:

    def __init__(self, predict_flag=False):
        self.label_map = [' '] + phoneme_list.PHONEME_MAP
        self.decoder = CTCBeamDecoder(
            labels=self.label_map,
            blank_id=0,
            beam_width=200, 
            log_probs_input=False
        )
        self.predict_flag = predict_flag

    def __call__(self, prediction, target):
        return self.forward(prediction, target)

    def forward(self, prediction, target):
        phoneme = prediction[0]
        feature_lengths = prediction[1].int()
        
        
#         phoneme = phoneme.cpu()
        probs = F.softmax(phoneme, dim=2)
#         probs = torch.transpose(probs, 0, 1)
        probs = probs.permute(1,0,2)
        
#         probs = phoneme.softmax(2)
        output, scores, timesteps, out_seq_len = self.decoder.decode(probs=probs, seq_lens=feature_lengths)
        pos = 0
        ls = 0.
        if self.predict_flag:
            decoded_predictions = []
            for i in range(output.size(0)):
                pred = "".join(self.label_map[o] for o in output[i, 0, :out_seq_len[i, 0]])
                decoded_predictions.append(pred)
            return decoded_predictions
        else:
            labels = target[0] + 1
            label_lengths = target[1].int()
            for i in range(output.size(0)):
                pred = "".join(self.label_map[o] for o in output[i, 0, :out_seq_len[i, 0]])
                true = "".join(self.label_map[l] for l in labels[pos:pos + label_lengths[i]])
                #print("Pred: {}, True: {}".format(pred, true))
                pos += label_lengths[i].item()
                ls += L.distance(pred, true)
            assert pos == labels.size(0)
            return ls / output.size(0)

def predict(model, test_dataloader):
    with torch.no_grad():
        model.eval()
        error_rate_op = ER(predict_flag=True)
        print('start prediction')
        encoded_prediction = []
        for i, data_batch in enumerate(test_dataloader):
            if i%100 == 0:
                    print(i)
            predictions_batch, feature_lengths_batch = model(data_batch)
            encoded_prediction.append(error_rate_op((predictions_batch.cpu(), feature_lengths_batch.cpu()), 
                                  []))
    return encoded_prediction

def run_eval(model, test_dataloader):
    with torch.no_grad():
        model.eval()
        error_rate_op = ER()
        print('start eval')
        avg_error = 0
        for data_batch, labels_batch in test_dataloader:

            predictions_batch, feature_lengths_batch = model(data_batch)
            target_lengths = torch.tensor([len(seq_labels) for seq_labels in labels_batch])
#             predictions_batch_log = predictions_batch.softmax(2)
            error = error_rate_op((predictions_batch.cpu(), feature_lengths_batch.cpu()), 
                                  (torch.cat(labels_batch).cpu(),target_lengths.cpu()))
            avg_error+=error
            
    return avg_error/len(test_dataloader)


def run(model, optimizer, train_dataloader, valid_dataloader, best_epoch, best_vali_loss, start_epoch=None):
    best_eval = None
    start_epoch = 0 if start_epoch is None else start_epoch
    max_epoch = config.max_epoch
    batch_size = config.batch_size
    
    model = model.cuda() if torch.cuda.is_available() else model
    
    ctc = CTCCriterion(size_average=True)
#     ctc = nn.CTCLoss()
    for epoch in range(start_epoch, max_epoch+1):
        start_time = time.time()
        model.train()
        # outputs records
        f = open(os.path.join(paths.output_path,'metrics.txt'), 'a')
        print_file_and_screen('### Epoch %5d' % (epoch), f=f)
        
        avg_loss = 0
        num_batches = len(train_dataloader)
        for batch, (data_batch, label_batch) in enumerate(train_dataloader): # lists, presorted, preloaded on GPU
            optimizer.zero_grad()
            phoneme, input_lengths = model(data_batch)
            target_lengths = torch.tensor([len(seq_labels) for seq_labels in label_batch])
            loss = ctc.forward((phoneme, input_lengths, target_lengths), torch.cat(label_batch))

            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if batch%50 == 49:
                print_file_and_screen('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch, batch+1, avg_loss/50), f = f)
                avg_loss = 0.0
            # clear memory
            torch.cuda.empty_cache()
            data_batch = data_batch.detach()
            label_batch = label_batch.detach()
            del data_batch
            del label_batch
            del loss
                
        train_loss = test_validation(model, train_dataloader)
        val_loss = test_validation(model, valid_dataloader)
        print_file_and_screen('Train Loss: {:.4f}\tVal Loss: {:.4f}\t'.format(train_loss, val_loss), f=f)
        
        # check whether the best
        if val_loss < best_vali_loss:
            best_vali_loss = val_loss
            best_epoch = epoch
            is_best = True
        else:
            is_best = False
        
        with torch.no_grad():
            avg_ldistance = run_eval(model, valid_dataloader)
        print_file_and_screen('vali_distance: {:.4f}\t'.format(avg_ldistance), f=f)
        
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'best_vali_loss': best_vali_loss,
            'best_epoch': best_epoch,
            'optimizer_label_state_dict' : optimizer.state_dict()
        }, is_best, paths.output_path, filename='4bilstm_adam_'+str(epoch)+'.pth.tar')
        
        
        end_time = time.time()
        print_file_and_screen('Epoch time used: ', end_time - start_time, 's', f=f)
        
        f.close()
    
    # print summary to the file
    with open(os.path.join(paths.output_path,'metrics.txt'), 'a') as f:
        print_file_and_screen('Summary:', f=f)
        print_file_and_screen('- Best Epoch: %1d | - Best Val Acc: %1d'%(best_epoch, best_vali_loss), f=f)

def run_with_scheduler(model, optimizer, scheduler_patience, scheduler_factor, train_dataloader, valid_dataloader, best_epoch, best_vali_loss, start_epoch=None):
    best_eval = None
    start_epoch = 0 if start_epoch is None else start_epoch
    max_epoch = config.max_epoch
    batch_size = config.batch_size
    
    model = model.cuda() if torch.cuda.is_available() else model
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True)
    print('-- Starting training with scheduler.')
    ctc = CTCCriterion(size_average=True)
#     ctc = nn.CTCLoss()
    for epoch in range(start_epoch, max_epoch+1):
        start_time = time.time()
        model.train()
        # outputs records
        f = open(os.path.join(paths.output_path,'metrics.txt'), 'a')
        print_file_and_screen('### Epoch %5d' % (epoch), f=f)
        
        avg_loss = 0
        num_batches = len(train_dataloader)
        for batch, (data_batch, label_batch) in enumerate(train_dataloader): # lists, presorted, preloaded on GPU
            optimizer.zero_grad()
            phoneme, input_lengths = model(data_batch)
            target_lengths = torch.tensor([len(seq_labels) for seq_labels in label_batch])
            loss = ctc.forward((phoneme, input_lengths, target_lengths), torch.cat(label_batch))

            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if batch%50 == 49:
                print_file_and_screen('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch, batch+1, avg_loss/50), f = f)
                avg_loss = 0.0
                
        train_loss = test_validation(model, train_dataloader)
        val_loss = test_validation(model, valid_dataloader)
        print_file_and_screen('Train Loss: {:.4f}\tVal Loss: {:.4f}\t'.format(train_loss, val_loss), f=f)
        
#         # scheduler upates
#         scheduler.step(val_loss)
        
        # check whether the best
        if val_loss < best_vali_loss:
            best_vali_loss = val_loss
            best_epoch = epoch
            is_best = True
        else:
            is_best = False
        
        with torch.no_grad():
            avg_ldistance = run_eval(model, valid_dataloader)
        print_file_and_screen('vali_distance: {:.4f}\t'.format(avg_ldistance), f=f)
        # scheduler upates on validation distance rather than loss
        scheduler.step(avg_ldistance)
        
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'best_vali_loss': best_vali_loss,
            'best_epoch': best_epoch,
            'optimizer_label_state_dict' : optimizer.state_dict()
        }, is_best, paths.output_path, filename='4bilstm_adam_'+str(epoch)+'.pth.tar')
        
        
        end_time = time.time()
        print_file_and_screen('Epoch time used: ', end_time - start_time, 's', f=f)
        
        f.close()
    
    # print summary to the file
    with open(os.path.join(paths.output_path,'metrics.txt'), 'a') as f:
        print_file_and_screen('Summary:', f=f)
        print_file_and_screen('- Best Epoch: %1d | - Best Val Acc: %1d'%(best_epoch, best_vali_loss), f=f)

def test_validation(model, valid_dataloader):
    model.eval()
    num_batches = len(valid_dataloader)
    avg_loss = 0
    ctc = CTCCriterion(size_average=True)
    for batch, (data_batch, label_batch) in enumerate(valid_dataloader):
        phoneme, input_lengths = model(data_batch)
        target_lengths = torch.tensor([len(seq_labels) for seq_labels in label_batch])
        loss = ctc.forward((phoneme, input_lengths, target_lengths), torch.cat(label_batch))
        avg_loss += loss.item()
    return avg_loss/num_batches
    
def init_weights(net, weight_init,
                 tanh_weight_gain=init.calculate_gain("tanh"),
                 bias_init=partial(init.constant_, val=0.0),
                 forget_bias_init=partial(init.constant_, val=1.0)):
    """
    Args:
        net (nn.Module): nn.RNN like module
        weight_init: its input is torch.FloatTensor
        
        #  init_weights(self.rnn, initializer, tanh_weight_gain=1.0)
        #  source： https://gist.github.com/ShigekiKarita/a2dea387d0d4cd7c3f3388d4986e7547
        weight_hh weight_hl
    """
    for name, p in net.named_parameters():
        if 'rnn' in name:
            # initialize rnn layers
            if "bias" in name:
                bias_init(p.data)
                if isinstance(net, (nn.LSTM, nn.LSTMCell)):
                    n = p.size(0)
                    forget_bias_init(p.data[n // 4:n // 2])
            elif "weight" in name:
                weight_init(p.data)

                # NOTE: according to init.calculate_gain, tanh requires 5/3 gain from sigmoid.
                if isinstance(net, (nn.LSTM, nn.LSTMCell)):
                    n = p.size(0)
                    p.data[n // 2:n // 4 * 3] *= tanh_weight_gain
                if isinstance(net, (nn.GRU, nn.GRUCell)):
                    n = p.size(0)
                    p.data[n // 2:] *= tanh_weight_gain
# def main():
#     run()


# if __name__ == '__main__':
#     main()
