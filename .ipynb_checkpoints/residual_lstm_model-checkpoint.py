import torch.nn as nn
import torch.nn.functional as F

import time
import os
import sys

import numpy as np
import config
from util import *
import paths

class ResidualLstm(nn.Module):
    '''https://arxiv.org/pdf/1602.02373.pdf lstm+global pooling'''
    def __init__(self, vocab_size, embed_size, hidden_size, output_dim, nlayers, bidirectional, lstm_dropout, dropout1,dropout2, dropout3, pad_idx, train_embedding=True):
        super(ResidualLstm, self).__init__()
        # input padding index to embedding to prevent training embedding for paddings
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx = pad_idx)
        if not train_embedding:
            self.embedding.weight.requires_grad = False # make embedding non trainable
        self.lstm = nn.LSTM(embed_size, 
                           hidden_size, 
                           num_layers=nlayers, 
                           bidirectional=bidirectional, 
                           dropout=lstm_dropout)
        self.fc = nn.Linear(hidden_size * 2 + embed_size, output_dim)
        self.globalpooling = nn.AdaptiveAvgPool2d((1,None))
        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)
        self.dropout3 = nn.Dropout(dropout3)
        
    def forward(self, text, text_lengths, testing=False):
        # text already padded by iterator
        if testing:
            # if we are predicting for test set
            text, text_lengths, reverse_order = self.collate_lines_for_test(text, text_lengths)
        # [sent len, batch size]
        embedded = self.embedding(text) #[sent len, batch size, emb dim]
        embedded = self.dropout1(embedded)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
#         #unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output) # [sent len, batch size, hid dim * num directions]
        output_for_pooling = output.permute(1, 0, 2) # [batch size, sent len, hid dim * num directions]
        pooled = self.globalpooling(output_for_pooling).squeeze(1) # [batch size, hid dim * num directions]
        # residual block
        residual_embedded = embedded.permute(1, 0, 2)
        residual_pooled = F.avg_pool2d(residual_embedded, (residual_embedded.shape[1], 1)).squeeze(1) # [batch size, hiddim]
        x = torch.cat([self.dropout2(pooled), self.dropout3(residual_pooled)], dim=1) #[batch size, hiddim * 3]
        y_hat = self.fc(x)
        y_hat_hat = torch.zeros_like(y_hat)
        if testing:
            y_hat_hat = y_hat[reverse_order]
            return y_hat_hat
        return y_hat
    
    # collate fn lets you control the return value of each batch
    # for packed_seqs, you want to return your data sorted by length
    def collate_lines_for_test(self, seq_list, lens):
        inputs = seq_list.permute(1,0).cpu().numpy()
    #     lens = [len(seq) for seq in inputs]
        # sort by length
        seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
        ordered_inputs = torch.tensor([inputs[i] for i in seq_order]).permute(1,0).cuda()
        ordered_seq_lens = torch.tensor([lens[i] for i in seq_order]).cuda()
        reverse_order = sorted(range(len(lens)), key=seq_order.__getitem__, reverse=False)
        return ordered_inputs, ordered_seq_lens, seq_order
    
def run(model, optimizer, criterion, train_dataloader, valid_dataloader, best_epoch, best_vali_loss, DEVICE, start_epoch=None,model_prefix=config.model_prefix):
    best_eval = None
    start_epoch = 0 if start_epoch is None else start_epoch
    max_epoch = config.max_epoch
    batch_size = config.batch_size
    
    model = model.to(DEVICE)
    criterion.to(DEVICE)
    
    for epoch in range(start_epoch, max_epoch+1):
        start_time = time.time()
        model.train()
        # outputs records
        f = open(os.path.join(paths.output_path,'metrics.txt'), 'a')
        print_file_and_screen('### Epoch %5d' % (epoch), f=f)
        
        avg_loss = 0
        avg_acc = 0
        num_batches = len(train_dataloader)
        for idx, batch in enumerate(train_dataloader): # lists, presorted, preloaded on GPU
            optimizer.zero_grad()
            text, text_lengths = batch.text
            predictions = model(text, text_lengths, testing=False)
            predictions = predictions.squeeze(1) if len(predictions.size())>1 else predictions # prepare for batch size == 1
            loss = criterion.forward(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            avg_acc += acc.item()
            
            if idx%200 == 199:
                print_file_and_screen('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}\tAvg-Acc: {:.4f}'.format(epoch, idx+1, avg_loss/200 ,avg_acc/200), f = f)
                avg_loss = 0.0
                avg_acc = 0.0
            # clear memory
            torch.cuda.empty_cache()
#             batch = batch.detach()
            del batch
            del loss
                
        train_loss, train_acc = test_validation(model, criterion, train_dataloader)
        val_loss, val_acc = test_validation(model, criterion, valid_dataloader)
        print_file_and_screen('Train Loss: {:.4f}\tTrain Acc: {:.4f}\tVal Loss: {:.4f}\tVal Acc: {:.4f}'.format(train_loss, train_acc, val_loss, val_acc), f=f)
        
        # check whether the best
        if val_loss < best_vali_loss:
            best_vali_loss = val_loss
            best_epoch = epoch
            is_best = True
        else:
            is_best = False
        
        
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_vali_loss': best_vali_loss,
            'best_epoch': best_epoch,
            'optimizer_label_state_dict' : optimizer.state_dict()
        }, is_best, paths.output_path, filename=model_prefix+str(epoch)+'.pth.tar')
        
        
        end_time = time.time()
        print_file_and_screen('Epoch time used: ', end_time - start_time, 's', f=f)
        
        f.close()
    
    # print summary to the file
    with open(os.path.join(paths.output_path,'metrics.txt'), 'a') as f:
        print_file_and_screen('Summary:', f=f)
        print_file_and_screen('- Best Epoch: %1d | - Best Val Loss: %.4f'%(best_epoch, best_vali_loss), f=f)

def test_validation(model, criterion, valid_dataloader):
    model.eval()
    num_batches = len(valid_dataloader)
    avg_loss = 0.0
    avg_acc = 0.0
    for idx, batch in enumerate(valid_dataloader):
        text, text_lengths = batch.text
        predictions = model(text, text_lengths, testing=False)
        predictions = predictions.squeeze(1) if len(predictions.size())>1 else predictions # prepare for batch size == 1
        loss = criterion.forward(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        avg_loss += loss.item()
        avg_acc += acc.item()
    return avg_loss/num_batches, avg_acc/num_batches


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def predict(model, test_dataloader, DEVICE):
    model.to(DEVICE)
    model.eval()
    prediction = []
    for i, batch in enumerate(test_dataloader):
        if i%400 == 0:
            print(i)
        text, text_lengths = batch.text
        predictions_batch = model(text, text_lengths, testing=True)
        predictions_batch = predictions_batch.squeeze(1) if len(predictions_batch.size())>1 else predictions_batch # prepare for batch size == 1
        rounded_preds = torch.round(torch.sigmoid(predictions_batch))
#         rounded_preds = np.round(1 / (1 + np.exp(-predictions_batch.detach().cpu().numpy())))
        prediction.append(rounded_preds)
#     return np.hstack(prediction)
    return torch.cat(prediction, dim=0).detach().cpu().numpy()