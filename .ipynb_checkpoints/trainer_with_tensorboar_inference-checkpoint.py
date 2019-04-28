import torch
import torch.nn as nn
import torch.nn.functional as F

import Levenshtein as L

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import time
import os
import sys

import numpy as np
import config
from util import *
import paths
import char_language_model

import pdb

from tensorboardX import SummaryWriter


def run(model, optimizer, criterion, validation_criterion, train_dataloader, valid_dataloader, language_model, best_epoch, best_vali_loss, DEVICE, tLog, vLog, TEACHER_FORCING_Ratio=1, scheduler=None, start_epoch=None, model_prefix=config.model_prefix):
    best_eval = None
    start_epoch = 0 if start_epoch is None else start_epoch
    max_epoch = config.max_epoch
    batch_size = config.batch_size

    model = model.to(DEVICE)
    criterion = criterion.to(DEVICE)
    validation_criterion = validation_criterion.to(DEVICE)
    
    global_iteration_index = 0 #

    if scheduler is not None:
        print('-- Starting training with scheduler.')

    for epoch in range(start_epoch, max_epoch+1):
        start_time = time.time()
        model.train()
        # outputs records
        f = open(os.path.join(paths.output_path, 'metrics.txt'), 'a')
        print_file_and_screen('### Epoch %5d' % (epoch), f=f)

        avg_loss = 0
        avg_distance = 0
        avg_perplexity = 0
        num_batches = len(train_dataloader)
        # lists, presorted, preloaded on GPU
        for idx, (data_batch, label_batch) in enumerate(train_dataloader):
            optimizer.zero_grad()
            predictions, prediction_labels, sorted_labels, labels_lens, batch_attention = model(
                (data_batch, label_batch), TEACHER_FORCING_Ratio, TEST=False)  # N, max_label_L, vocab_size; N, max_label_L
            if len(sorted_labels)==2:
                # the model returns true labels and smoothed onehot targets
                true_sorted_labels, sorted_labels = sorted_labels
            else:
                true_sorted_labels = sorted_labels
            # mask for cross entropy loss
            batch_size, max_label_lens, _ = predictions.size()
#             prediction_mask = torch.stack([torch.arange(0, max_label_lens) for i in range(batch_size)]).int()
#             prediction_mask = prediction_mask < torch.stack([torch.full(
#                 (1, max_label_lens), length).squeeze(0) for length in labels_lens]).int()
            prediction_mask = (torch.arange(0, max_label_lens) < torch.tensor(labels_lens).view(len(labels_lens),-1)).float().to(DEVICE) # float for matrix mul
            prediction_mask.requires_grad = False
            # use cross entropy loss, assume set (reduce = False)
#             sorted_labels = sorted_labels.to(DEVICE)
#             sorted_labels.requires_grad = False
            batch_loss = 0.0
            for utterance_index, utterance_pred in enumerate(predictions):
                # N,1 | N,L,Hidden_size
                utterance_loss = criterion(utterance_pred, sorted_labels[utterance_index]) * prediction_mask[utterance_index]
                batch_loss += utterance_loss.sum() # use sum to punish long

    #         batch_loss = criterion.forward(predictions.permute(0,2,1), sorted_labels.to(DEVICE)) #(N,C,d1) & (N,d1)
    #         batch_loss = batch_loss * prediction_mask
            loss = batch_loss.sum()/batch_size # loss per instance used for train
            perplexity_loss = torch.exp(batch_loss.sum()/prediction_mask.float().sum()).detach().cpu() # perplexity
            
            SHOW_RESULT = False # whether print out the prediction result while calculating distance 
            if idx % 50 == 49:
                SHOW_RESULT = True
            # distance
            distance = evaluate_distance(prediction_labels.detach().cpu().numpy(), true_sorted_labels.detach().cpu().numpy(), labels_lens, language_model, SHOW_RESULT)
#             print('distance', distance)
#             print('-'*60)
    

            loss.backward()
            optimizer.step()
            avg_loss += loss.detach().cpu().item()
            avg_distance += distance
            avg_perplexity += perplexity_loss
            
            # add log to tensorboard
            # 1. Log scalar values (scalar summary)
            tr_info = { 'loss': loss.cpu().detach().numpy(), 'perplexity': perplexity_loss.numpy(), 'distance': distance }

            for tag, value in tr_info.items():
                tLog.add_scalar(tag, value, global_iteration_index+1)

            # 2. Log values and gradients of the parameters (histogram summary)
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                tLog.add_histogram(tag, value.data.cpu().numpy(), global_iteration_index+1)
                tLog.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), global_iteration_index+1)
            
            # 3. Log two visualizations
            tLog.add_figure('attention_last_instance_batch', plot_single_attention_return(batch_attention[0].squeeze(0).cpu().numpy()), global_step=global_iteration_index+1)
            tLog.add_figure('gradient_flow_batch', plot_grad_flow_return(model.named_parameters()), global_step=global_iteration_index+1)
            
            global_iteration_index += 1
            
            if idx == 0:
                plot_grad_flow_simple(model.named_parameters(), epoch, paths.gradient_path)
            
            if idx % 50 == 49:
                # plot gradients
                plot_grad_flow_simple(model.named_parameters(), epoch, paths.gradient_path)
                # plot attention
                plot_single_attention(batch_attention[0].squeeze(0).cpu().numpy(), epoch, paths.attention_path)

            if idx % 100 == 99:
                print_file_and_screen('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}\tAvg-Perplexity: {:.4f}\tAvg-Distance: {:.4f}'.format(
                    epoch, idx+1, avg_loss/100, avg_perplexity/100, avg_distance/100), f=f)
                avg_loss = 0.0
                avg_distance = 0.0
                avg_perplexity = 0.0

            # clear memory
            data_batch = [data.detach() for data in data_batch]
            label_batch = [label.detach() for label in label_batch]
            predictions = predictions.detach()
            loss = loss.detach()
            batch_loss = batch_loss.detach()
            del data_batch, label_batch
            del predictions
            del loss, batch_loss
            torch.cuda.empty_cache()

        train_loss, train_distance, train_perplexity_loss = test_validation(
            model, validation_criterion, train_dataloader, language_model, DEVICE)
        val_loss, val_distance, val_perplexity_loss = test_validation(
            model, validation_criterion, valid_dataloader, language_model, DEVICE)
        print_file_and_screen('Train Loss: {:.4f}\tTrain Distance: {:.4f}\tVal Loss: {:.4f}\tVal Distance: {:.4f}'.format(
            train_loss, train_distance, val_loss, val_distance), f=f)
        
        # add log to tensorboard
        # 4. Log scalar values (scalar summary)
        vr_info = { 'train_loss': train_loss, 'train_perplexity': train_perplexity_loss, 'train_distance': train_distance,'val_loss': val_loss, 'val_perplexity': val_perplexity_loss, 'val_distance': val_distance }

        for tag, value in vr_info.items():
            vLog.add_scalar(tag, value, global_iteration_index+1)

        if scheduler is not None:
            # update loss on scheduer
            scheduler.step(val_loss)

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
            'val_distance': val_distance,
            'best_vali_loss': best_vali_loss,
            'best_epoch': best_epoch,
            'optimizer_label_state_dict': optimizer.state_dict()
        }, is_best, paths.output_path, filename=model_prefix+str(epoch)+'.pth.tar')

        end_time = time.time()
        print_file_and_screen('Epoch time used: ',
                              end_time - start_time, 's', f=f)

        f.close()

    # print summary to the file
    with open(os.path.join(paths.output_path, 'metrics.txt'), 'a') as f:
        print_file_and_screen('Summary:', f=f)
        print_file_and_screen(
            '- Best Epoch: %1d | - Best Val Loss: %.4f' % (best_epoch, best_vali_loss), f=f)


def test_validation(model, validation_criterion, valid_dataloader, language_model, DEVICE):
    print('## Start testing....')
    model.eval()
    num_batches = len(valid_dataloader)
    avg_loss = 0.0
    avg_distance = 0.0
    avg_perplexity = 0.0
    for idx,  (data_batch, label_batch) in enumerate(valid_dataloader):
        predictions, prediction_labels, sorted_labels, labels_lens, _ = model(
            (data_batch, label_batch), TEACHER_FORCING_Ratio=0, TEST=False, VALIDATE=True, MAX_SEQ_LEN=500, SEARCH_MODE='greedy')  # N, max_label_L, vocab_size; N, max_label_L
        
        # mask for cross entropy loss
        batch_size, max_label_lens, _ = predictions.size()
#             prediction_mask = torch.stack([torch.arange(0, max_label_lens) for i in range(batch_size)]).int()
#             prediction_mask = prediction_mask < torch.stack([torch.full(
#                 (1, max_label_lens), length).squeeze(0) for length in labels_lens]).int()
        prediction_mask = (torch.arange(0, max_label_lens) < torch.tensor(labels_lens).view(len(labels_lens),-1)).float().to(DEVICE) # float for matrix mul
        prediction_mask.requires_grad = False
        # use cross entropy loss, assume set (reduce = False)
#             sorted_labels = sorted_labels.to(DEVICE)
#             sorted_labels.requires_grad = False
        batch_loss = 0.0

        for utterance_index, utterance_pred in enumerate(predictions):
            # pad to reach the same time length
            utterance_padded_label = sorted_labels[utterance_index] # max_label_len
            utterance_mask = prediction_mask[utterance_index] # max_prediction_len
            if utterance_pred.size(0) < utterance_padded_label.size(0):
                # pad utterance pred
                pad = ((0,utterance_padded_label.size(0)-utterance_pred.size(0)))
                utterance_pred = F.pad(utterance_pred, pad, mode='constant', value=char_language_model.EOS_token)
                utterance_mask = F.pad(utterance_mask, pad, mode='constant', value=0)
                utterance_mask.requires_grad = False
            elif utterance_padded_label.size(0) < utterance_pred.size(0):
                # pad utterance label and mask
                pad = ((0,utterance_pred.size(0)-utterance_padded_label.size(0)))
                utterance_padded_label = F.pad(utterance_padded_label, pad, mode='constant', value=char_language_model.EOS_token)
                utterance_padded_label.requires_grad = False
            # N,1 | N,L,Hidden_size
            utterance_loss = validation_criterion(utterance_pred, utterance_padded_label) * utterance_mask
            batch_loss += utterance_loss.sum() # use sum to punish long

#         batch_loss = validation_criterion.forward(predictions.permute(0,2,1), sorted_labels.to(DEVICE)) #(N,C,d1) & (N,d1)
#         batch_loss = batch_loss * prediction_mask
        loss = batch_loss.sum()/batch_size # loss per instance used for train
        perplexity_loss = torch.exp(batch_loss.sum()/prediction_mask.float().sum()).detach().cpu() # perplexity

        SHOW_RESULT = False # whether print out the prediction result while calculating distance 
        if idx % 50 == 49:
            SHOW_RESULT = True
        # distance
        distance = evaluate_distance(prediction_labels.detach().cpu().numpy(), true_sorted_labels.detach().cpu().numpy(), labels_lens, language_model, SHOW_RESULT)
        
        avg_loss += loss.detach().cpu().item()
        avg_distance += distance
        avg_perplexity += perplexity_loss
        
        # clear memory
        data_batch = [data.detach() for data in data_batch]
        label_batch = [label.detach() for label in label_batch]
        predictions = predictions.detach()
        loss = loss.detach()
        batch_loss = batch_loss.detach()
        del data_batch, label_batch
        del predictions
        del loss, batch_loss, perplexity_loss
        torch.cuda.empty_cache()
        
    return avg_loss/num_batches, avg_distance/num_batches, avg_perplexity/num_batches


def inference(model, test_dataloader):
    print('## Start inferencing....')
    model.eval()
    num_batches = len(test_dataloader)
    inferences = []
    for idx,  data_batch in enumerate(test_dataloader):
        predictions, prediction_labels, reverse_sequence_order = model(
            data_batch, TEACHER_FORCING_Ratio=0, TEST=True, VALIDATE=False, MAX_SEQ_LEN=500, SEARCH_MODE='greedy')  # N, max_label_L, vocab_size; N, max_label_L; with grad; predict till EOS
        
        # all predictions are done till EOS token; unsort the list and append
        inferences.extend(predictions.detach().cpu().numpy()[reverse_sequence_order])
        # clear memory
        data_batch = [data.detach() for data in data_batch]
        predictions = predictions.detach()
        loss = loss.detach()
        batch_loss = batch_loss.detach()
        del data_batch, label_batch
        del predictions
        del loss, batch_loss, perplexity_loss
        torch.cuda.empty_cache()
        
    return inferences

def evaluate_distance(predictions, padded_label, label_lens, lang, SHOW_RESULT=False):
    """ predictions: N, Max_len
        padded_labels: N, Max_len
        label_lens: N, len"""
    prediction_checker = []
    ls = 0.0
    for i in range(predictions.shape[0]): # for each instance
        if char_language_model.EOS_token in predictions[i]: # there exists EOS, use the first 1 as the end of sentence
            end_position = predictions[i].tolist().index(char_language_model.EOS_token)
            if end_position < len(predictions[i]): # not the last one
                pred = predictions[i][:end_position+1]
            else:
                pred = predictions[i]
        else:
            pred = predictions[i] # if no EOS, use the entire prediction
        
        true = padded_label[i][:label_lens[i]]
        pred = lang.indexes2string(pred)
        true = lang.indexes2string(true)
        ls += L.distance(pred, true)
        if i == 0:
            prediction_checker.extend([pred, true])
    if SHOW_RESULT:
        print("Pred: {}, True: {}".format(prediction_checker[0], prediction_checker[1]))
    return ls / predictions.shape[0]

def validate_manually(encoder, decoder, lang, utterance, transcript, TEACHER_FORCING_Ratio=0):
    # prepare data for encoder
#     if type(transcript) == torch.Tensor:
#         label = transcript
#     else:
#         label = torch.tensor(np.array(lang.string2indexes(transcript))).to(DEVICE)
    encoder_argument_list = [(utterance, label), TEST]
    keys, values, labels, final_seq_lens, sequence_order, reverse_sequence_order = self.encoder(
        encoder_argument_list)
    argument_list = [keys, values, final_seq_lens, sequence_order,
                        reverse_sequence_order, char_language_model.SOS_token, char_language_model.EOS_token, MAX_SEQ_LEN]
    y_hat, y_hat_labels, attentions = self.decoder.inference(argument_list)
    labels_lens = [len(label) for label in labels]
    # y_hat_label N, MaxLen
    return y_hat_labels, labels, attentions

def showAttention(output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    fig.set_size_inches(20, 20)
    plt.show()


def evaluateAndShowAttention(input_sentence):
    # TODO: not in-use for now
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)
