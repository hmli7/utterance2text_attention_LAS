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
import char_language_model

import pdb

# from tensorboardX import SummaryWriter

import OneStepBeam
import multiprocessing as m


def run(model, optimizer, criterion, validation_criterion, train_dataloader, valid_dataloader, language_model, best_epoch, best_vali_loss, DEVICE, tLog, vLog, teacher_forcing_scheduler, scheduler=None, start_epoch=None, model_prefix=config.model_prefix, NUM_CONFIG=0, TRAIN_SEARCH_MODE='greedy', output_path=None):
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
        f = open(os.path.join(output_path, 'metrics.txt'), 'a')
        print_file_and_screen('### Epoch %5d' % (epoch), f=f)

        avg_loss = 0
        avg_distance = 0
        avg_perplexity = 0
        num_batches = len(train_dataloader)
        
        # update teacher forcing ratio
        # update teacher forcing ratio
        TEACHER_FORCING_Ratio = teacher_forcing_scheduler.get_rate(epoch)
        
        # lists, presorted, preloaded on GPU
        for idx, (data_batch, label_batch) in enumerate(train_dataloader):
            optimizer.zero_grad()
            predictions, prediction_labels, sorted_labels, labels_lens, batch_attention = model(
                (data_batch, label_batch), TEACHER_FORCING_Ratio, TEST=False, NUM_CONFIG=NUM_CONFIG, SEARCH_MODE=TRAIN_SEARCH_MODE)  # N, max_label_L, vocab_size; N, max_label_L
            if len(sorted_labels)==2:
                # the model returns true labels and smoothed onehot targets
                true_sorted_labels, gumbel_true_labels = sorted_labels
            else:
                true_sorted_labels = sorted_labels
            # mask for cross entropy loss
            batch_size, max_label_lens = sorted_labels.size()
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
                if utterance_pred.size(0) < utterance_padded_label.size(0):
                    # pad utterance pred
                    pred_pad = (0,0,0,utterance_padded_label.size(0)-utterance_pred.size(0)) # pad the second last dim with 0
                    pred_label_pad = ((0,utterance_padded_label.size(0)-utterance_pred.size(0)))
                    utterance_pred = F.pad(utterance_pred, pred_pad, mode='constant', value=0)
#                     utterance_padded_y_hat_label = F.pad(prediction_labels[utterance_index], pred_label_pad, mode='constant', value=char_language_model.EOS_token)
                elif utterance_padded_label.size(0) < utterance_pred.size(0):
                    # pad utterance label
                    pad = ((0,utterance_pred.size(0)-utterance_padded_label.size(0)))
                    utterance_padded_label = F.pad(utterance_padded_label, pad, mode='constant', value=char_language_model.EOS_token)
                    utterance_padded_label.requires_grad = False
                    
#                 if utterance_padded_label is not None:
#                     padded_pred_labels.append(utterance_padded_label)
#                 else:
#                     padded_pred_labels.append(prediction_labels[utterance_index])
                    
                # N,1 | N,L,Hidden_size
                utterance_loss = criterion(utterance_pred, true_sorted_labels[utterance_index]) * prediction_mask[utterance_index]
                batch_loss += utterance_loss.sum() # use sum to punish long

    #         batch_loss = criterion.forward(predictions.permute(0,2,1), sorted_labels.to(DEVICE)) #(N,C,d1) & (N,d1)
    #         batch_loss = batch_loss * prediction_mask
            loss = batch_loss.sum()/batch_size # loss per instance used for train
            perplexity_loss = torch.exp(batch_loss.sum()/prediction_mask.float().sum()).detach().cpu() # perplexity
            
            SHOW_RESULT = False # whether print out the prediction result while calculating distance 
            if idx % 50 == 49:
                SHOW_RESULT = True
                
            # distance
            if type(prediction_labels) == torch.Tensor:
                padded_pred_labels = prediction_labels.detach().cpu().numpy()
            else:
                # for beam search y hat labels are already numpy
                padded_pred_labels = prediction_labels
            distance = evaluate_distance(padded_pred_labels, true_sorted_labels.detach().cpu().numpy(), labels_lens, language_model, SHOW_RESULT)
#             print('distance', distance)
#             print('-'*60)
    

            loss.backward()
            optimizer.step()
            avg_loss += loss.detach().cpu().item()
            avg_distance += distance
            avg_perplexity += perplexity_loss
            
            
            if idx == 0:
                plot_grad_flow_simple(model.named_parameters(), epoch, os.path.join(output_path, 'gradient_plots'))
            
            if idx % 50 == 49:
                # plot gradients
                plot_grad_flow_simple(model.named_parameters(), epoch, os.path.join(output_path, 'gradient_plots'))
                # plot attention
                plot_single_attention(batch_attention[0].squeeze(0).cpu().numpy(), epoch, os.path.join(output_path, 'attention_plots'))
                
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

            if idx % 100 == 99:
                print_file_and_screen('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}\tAvg-Perplexity: {:.4f}\tAvg-Distance: {:.4f}'.format(
                    epoch, idx+1, avg_loss/100, avg_perplexity/100, avg_distance/100), f=f)
                avg_loss = 0.0
                avg_distance = 0.0
                avg_perplexity = 0.0

            # clear memory
            data_batch = [data.detach() for data in data_batch]
            label_batch = [label.detach() for label in label_batch]
            if type(predictions) == torch.Tensor:
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
        print_file_and_screen('Train Loss: {:.4f}\tTrain Distance: {:.4f}\tTrain Perplexity: {:.4f}\tVal Loss: {:.4f}\tVal Distance: {:.4f}\tVal Perplexity: {:.4f}'.format(
            train_loss, train_distance, train_perplexity_loss, val_loss, val_distance, val_perplexity_loss), f=f)
        
        # add log to tensorboard
        # 4. Log scalar values (scalar summary)
        vr_info = { 'train_loss': train_loss, 'train_perplexity': train_perplexity_loss, 'train_distance': train_distance,'val_loss': val_loss, 'val_perplexity': val_perplexity_loss, 'val_distance': val_distance }

        for tag, value in vr_info.items():
            vLog.add_scalar(tag, value, global_iteration_index+1)

        if scheduler is not None:
            # update loss on scheduer
            scheduler.step(val_distance)

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
        }, is_best, output_path, filename=model_prefix+str(epoch)+'.pth.tar')

        end_time = time.time()
        print_file_and_screen('Epoch time used: ',
                              end_time - start_time, 's', f=f)

        f.close()

    # print summary to the file
    with open(os.path.join(output_path, 'metrics.txt'), 'a') as f:
        print_file_and_screen('Summary:', f=f)
        print_file_and_screen(
            '- Best Epoch: %1d | - Best Val Loss: %.4f' % (best_epoch, best_vali_loss), f=f)


def test_validation(model, validation_criterion, valid_dataloader, language_model, DEVICE, PROMPT=True, RETURN_UTTERANCE_LEVEL=False):
    if PROMPT:
        print('## Start validating....')
    model.eval()
    num_batches = len(valid_dataloader)
    avg_loss = 0.0
    avg_distance = 0.0
    avg_perplexity = 0.0
    losses = []
    distances = []
    perplexities = []
    for idx,  (data_batch, label_batch) in enumerate(valid_dataloader):
        
        predictions, prediction_labels, sorted_labels, labels_lens, _ = model(
            (data_batch, label_batch), TEACHER_FORCING_Ratio=0, TEST=False, VALIDATE=True, NUM_CONFIG=500, SEARCH_MODE='greedy')  # N, max_label_L, vocab_size; N, max_label_L; NUM_CONFIG == MAX_SEQ_LEN
        
        if len(sorted_labels)==2:
            # the model returns true labels and smoothed onehot targets
            true_sorted_labels, gumbel_true_labels = sorted_labels
        else:
            true_sorted_labels = sorted_labels
        
        # mask for cross entropy loss
        batch_size, max_label_lens, _ = predictions.size()
## first version mask 
#             prediction_mask = torch.stack([torch.arange(0, max_label_lens) for i in range(batch_size)]).int()
#             prediction_mask = prediction_mask < torch.stack([torch.full(
#                 (1, max_label_lens), length).squeeze(0) for length in labels_lens]).int()
## second version mask
#         prediction_mask = (torch.arange(0, max_label_lens) < torch.tensor(labels_lens).view(len(labels_lens),-1)).float().to(DEVICE) # float for matrix mul
#         prediction_mask.requires_grad = False
        # use cross entropy loss, assume set (reduce = False)
#             sorted_labels = sorted_labels.to(DEVICE)
#             sorted_labels.requires_grad = False

        batch_loss = 0.0
        batch_perplexity = 0.0
        # for returning utterance level metrics
        Batch_losses = []
        Batch_distances = []
        Batch_perplexities = []
        
        for utterance_index, utterance_pred_residual in enumerate(predictions):
            utterance_pred_label = prediction_labels[utterance_index]
            # slice the label to the length where the model was trying to end for one utterance
            if char_language_model.EOS_token in utterance_pred_label: # there exists EOS, use the first 1 as the end of sentence; the model will return a matrix, with the width of the last ended prediction
                end_position = utterance_pred_label.detach().cpu().numpy().tolist().index(char_language_model.EOS_token)
                if end_position < len(utterance_pred_label): # not the last one
                    utterance_pred = utterance_pred_residual[:end_position+1]
                else:
                    utterance_pred = utterance_pred_residual
            else:
                utterance_pred = utterance_pred_residual # if no EOS, use the entire prediction
            
            # pad to reach the same time length
            utterance_padded_label = true_sorted_labels[utterance_index] # max_label_len
            if utterance_pred.size(0) < utterance_padded_label.size(0):
                # pad utterance pred
                pred_pad = (0,0,0,utterance_padded_label.size(0)-utterance_pred.size(0)) # pad the second last dim with 0
                utterance_pred = F.pad(utterance_pred, pred_pad, mode='constant', value=0)
            elif utterance_padded_label.size(0) < utterance_pred.size(0):
                # pad utterance label
                pad = ((0,utterance_pred.size(0)-utterance_padded_label.size(0)))
                utterance_padded_label = F.pad(utterance_padded_label, pad, mode='constant', value=char_language_model.EOS_token)
                utterance_padded_label.requires_grad = False
            assert utterance_padded_label.size(0) == utterance_pred.size(0)
            utterance_mask = (torch.arange(0, utterance_padded_label.size(0)) < torch.tensor(labels_lens[utterance_index])).float().to(DEVICE)
            utterance_mask.requires_grad = False

#             pdb.set_trace()
            # N,1 | N,L,Hidden_size
            utterance_loss = validation_criterion(utterance_pred, utterance_padded_label.to(DEVICE)) * utterance_mask
            batch_loss += utterance_loss.sum() # use sum to punish long
            utterance_perplexity = utterance_loss.sum()/utterance_mask.sum()
            batch_perplexity += utterance_perplexity
            
            # for returning utterance level metrics
            Batch_losses.append(utterance_loss.sum())
            Batch_perplexities.append(utterance_perplexity)

#         batch_loss = validation_criterion.forward(predictions.permute(0,2,1), sorted_labels.to(DEVICE)) #(N,C,d1) & (N,d1)
#         batch_loss = batch_loss * prediction_mask
        loss = batch_loss.sum()/batch_size # loss per instance used for train
        perplexity_loss = torch.exp(batch_perplexity.sum()/batch_size).detach().cpu() # perplexity

        SHOW_RESULT = False # whether print out the prediction result while calculating distance 
        if idx % 50 == 49:
            SHOW_RESULT = True
        # distance
        if RETURN_UTTERANCE_LEVEL:
            distance, Batch_distances = evaluate_distance(prediction_labels.detach().cpu().numpy(), true_sorted_labels.detach().cpu().numpy(), labels_lens, language_model, SHOW_RESULT, RETURN_UTTERANCE_LEVEL)
        else:
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
    
        losses.append(Batch_losses)
        distances.append(Batch_distances)
        perplexities.append(Batch_perplexities)
    
    if RETURN_UTTERANCE_LEVEL:
        return losses, distances, perplexities
    return avg_loss/num_batches, avg_distance/num_batches, avg_perplexity/num_batches

def test_validation_random_search(model, validation_criterion, valid_dataloader, language_model, DEVICE, MAX_SEQ_LEN=500, N_CANDIDATES=100):
    print('## Start validating....')
    model.eval()
    num_batches = len(valid_dataloader)
    avg_loss = 0.0
    avg_distance = 0.0
    avg_perplexity = 0.0
    for idx,  (data_batch, label_batch) in enumerate(valid_dataloader):
        print(idx)
        candidate_y_hats, candidate_labels, sorted_labels, labels_lens, sequence_order = model(
            (data_batch, label_batch), TEACHER_FORCING_Ratio=0, TEST=False, VALIDATE=True, NUM_CONFIG=(MAX_SEQ_LEN, N_CANDIDATES), SEARCH_MODE='random')
        # N_candidates, batch_size, max_label_lens (vary in different try same within each try), vocab_size; N_candidates, batch_size, max_label_lens; batch_size, max_label_len; batch_size; NUM_CONFIG = (MAX_SEQ_LEN, N_CANDIDATES)
        
        # mask for cross entropy loss
        batch_size, max_label_lens = sorted_labels.size()

#         prediction_mask = (torch.arange(0, max_label_lens) < torch.tensor(labels_lens).view(len(labels_lens),-1)).float().to(DEVICE) # float for matrix mul
#         prediction_mask.requires_grad = False # same mask for all candidates
        
        # retrieve sorted utterance data
        utterance_data = np.array(data_batch)[sequence_order]

        batch_loss = 0.0
        batch_perplexity = 0.0
        prediction_labels = []
        # for each utterance, pick a candidate and update loss; choose using pilot dummy loss using candidate as dummy true label; reporting dummy_loss
        for utterance_index in range(batch_size):
            print(idx, end=' ')
            # retrieve candidates
#             utterance_labels_candidates = [pilot[utterance_index] for pilot in candidate_labels]
            utterance_labels_candidates = candidate_labels[utterance_index*N_CANDIDATES:(utterance_index+1)*N_CANDIDATES]
            
            # search winner
#             candidates_dummy_losses = []
#             for candidate in utterance_labels_candidates:
#                 dummy_dataloader = [(utterance_data[utterance_index], candidate)] # a dummy data loader to get the dummy loss, distance, and perplexity
#                 dummy_loss, dummy_distance, dummy_perplexity_loss = test_validation(model, validation_criterion, dummy_dataloader, language_model, DEVICE, PROMPT=False) # testing with greedy search
#                 candidates_dummy_losses.append(dummy_loss)
        
            # make sure the candidate pred label are where it mean to stop
            # slice the label to the length where the model was trying to end for one utterance
            utterance_dummy_labels = []
            for candidate in utterance_labels_candidates:
                if char_language_model.EOS_token in candidate: # there exists EOS, use the first 1 as the end of sentence; the model will return a matrix, with the width of the last ended prediction
                    end_position = candidate.detach().numpy().tolist().index(char_language_model.EOS_token)
                    if end_position < len(candidate): # not the last one
                        utterance_dummy_label = candidate[:end_position+1]
                    else:
                        utterance_dummy_label = candidate
                else:
                    utterance_dummy_label = candidate # if no EOS, use the entire prediction
                utterance_dummy_labels.append(utterance_dummy_label)
            
            dummy_dataloader = [([utterance_data[utterance_index]]*len(utterance_dummy_labels), utterance_dummy_labels)]
            candidates_dummy_losses, candidates_dummy_distances, candidates_dummy_perplexity_losses = test_validation(model, validation_criterion, dummy_dataloader, language_model, DEVICE, PROMPT=False, RETURN_UTTERANCE_LEVEL=True) # testing with greedy search
            winner_index = np.argmin(candidates_dummy_distances[0]) # use 0th one because we have batch size 1
            
            utterance_pred = candidate_y_hats[utterance_index*N_CANDIDATES+winner_index, :, :] # label_len, vocab_size
            prediction_labels.append(utterance_dummy_labels[winner_index].detach().cpu().numpy())
            
            # calculate gumbel_dummy_loss
            # pad to reach the same time length
            utterance_padded_label = sorted_labels[utterance_index] # max_label_len

            if utterance_pred.size(0) < utterance_padded_label.size(0):
                # pad utterance pred
                pred_pad = (0,0,0,utterance_padded_label.size(0)-utterance_pred.size(0)) # pad the second last dim with 0
                utterance_pred = F.pad(utterance_pred, pred_pad, mode='constant', value=0)
            elif utterance_padded_label.size(0) < utterance_pred.size(0):
                # pad utterance label
                pad = ((0,utterance_pred.size(0)-utterance_padded_label.size(0)))
                utterance_padded_label = F.pad(utterance_padded_label, pad, mode='constant', value=char_language_model.EOS_token)
                utterance_padded_label.requires_grad = False
                
            utterance_mask = (torch.arange(0, utterance_padded_label.size(0)) < torch.tensor(labels_lens[utterance_index])).float().to(DEVICE)
            utterance_mask.requires_grad = False
            
#             pdb.set_trace()
            # N,1 | N,L,Hidden_size
            utterance_loss = validation_criterion(utterance_pred, utterance_padded_label.to(DEVICE)) * utterance_mask
            batch_loss += utterance_loss.sum() # use sum to punish long
            batch_perplexity += utterance_loss.sum()/utterance_mask.sum()

        loss = batch_loss.sum()/batch_size # loss per instance used for train
        perplexity_loss = torch.exp(batch_perplexity.sum()/batch_size).detach().cpu() # perplexity

        SHOW_RESULT = False # whether print out the prediction result while calculating distance 
        if idx % 50 == 49:
            SHOW_RESULT = True
        # distance
        distance = evaluate_distance(np.array(prediction_labels), sorted_labels.detach().cpu().numpy(), labels_lens, language_model, SHOW_RESULT)
        
        avg_loss += loss.detach().cpu().item()
        avg_distance += distance
        avg_perplexity += perplexity_loss
        
        # clear memory
        data_batch = [data.detach() for data in data_batch]
        label_batch = [label.detach() for label in label_batch]
        loss = loss.detach()
        batch_loss = batch_loss.detach()
        del data_batch, label_batch
        del loss, batch_loss, perplexity_loss
        torch.cuda.empty_cache()
        
    return avg_loss/num_batches, avg_distance/num_batches, avg_perplexity/num_batches

def test_validation_beam_search(model, validation_criterion, valid_dataloader, language_model, DEVICE, MAX_SEQ_LEN=500, beam_size=5, num_candidates=1):
    if PROMPT:
        print('## Start validating....')
    model.eval()
    num_batches = len(valid_dataloader)
    avg_loss = 0.0
    avg_distance = 0.0
    avg_perplexity = 0.0

    for idx,  (data_batch, label_batch) in enumerate(valid_dataloader):
        
        predictions, prediction_labels, sorted_labels, labels_lens, _ = model(
            data_batch, TEACHER_FORCING_Ratio=None, TEST=False, VALIDATE=True, NUM_CONFIG=(MAX_SEQ_LEN, beam_size, num_candidates), SEARCH_MODE='beam_search')  # N, max_label_L, vocab_size; N, max_label_L; with grad; predict till EOS; NUM_CONFIG == MAX_SEQ_LEN
        
        
        if len(sorted_labels)==2:
            # the model returns true labels and smoothed onehot targets
            true_sorted_labels, gumbel_true_labels = sorted_labels
        else:
            true_sorted_labels = sorted_labels
        
        # mask for cross entropy loss
        batch_size = len(data_batch)

        batch_loss = 0.0
        batch_perplexity = 0.0

        
        for utterance_index, utterance_pred_residual in enumerate(predictions):
            utterance_pred_label = prediction_labels[utterance_index]
            # slice the label to the length where the model was trying to end for one utterance
            if char_language_model.EOS_token in utterance_pred_label: # there exists EOS, use the first 1 as the end of sentence; the model will return a matrix, with the width of the last ended prediction
                end_position = utterance_pred_label.detach().cpu().numpy().tolist().index(char_language_model.EOS_token)
                if end_position < len(utterance_pred_label): # not the last one
                    utterance_pred = utterance_pred_residual[:end_position+1]
                else:
                    utterance_pred = utterance_pred_residual
            else:
                utterance_pred = utterance_pred_residual # if no EOS, use the entire prediction
            
            # pad to reach the same time length
            utterance_padded_label = true_sorted_labels[utterance_index] # max_label_len
            if utterance_pred.size(0) < utterance_padded_label.size(0):
                # pad utterance pred
                pred_pad = (0,0,0,utterance_padded_label.size(0)-utterance_pred.size(0)) # pad the second last dim with 0
                utterance_pred = F.pad(utterance_pred, pred_pad, mode='constant', value=0)
            elif utterance_padded_label.size(0) < utterance_pred.size(0):
                # pad utterance label
                pad = ((0,utterance_pred.size(0)-utterance_padded_label.size(0)))
                utterance_padded_label = F.pad(utterance_padded_label, pad, mode='constant', value=char_language_model.EOS_token)
                utterance_padded_label.requires_grad = False
            assert utterance_padded_label.size(0) == utterance_pred.size(0)
            utterance_mask = (torch.arange(0, utterance_padded_label.size(0)) < torch.tensor(labels_lens[utterance_index])).float().to(DEVICE)
            utterance_mask.requires_grad = False

#             pdb.set_trace()
            # N,1 | N,L,Hidden_size
            utterance_loss = validation_criterion(utterance_pred, utterance_padded_label.to(DEVICE)) * utterance_mask
            batch_loss += utterance_loss.sum() # use sum to punish long
            utterance_perplexity = utterance_loss.sum()/utterance_mask.sum()
            batch_perplexity += utterance_perplexity
            

#         batch_loss = validation_criterion.forward(predictions.permute(0,2,1), sorted_labels.to(DEVICE)) #(N,C,d1) & (N,d1)
#         batch_loss = batch_loss * prediction_mask
        loss = batch_loss.sum()/batch_size # loss per instance used for train
        perplexity_loss = torch.exp(batch_perplexity.sum()/batch_size).detach().cpu() # perplexity

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

def inference(model, test_dataloader,language_model, MAX_SEQ_LEN=500):
    print('## Start inferencing....')
    model.eval()
    num_batches = len(test_dataloader)
    inferences = []
    for idx,  data_batch in enumerate(test_dataloader):
        predictions, prediction_labels, _, reverse_sequence_order = model(
            data_batch, TEACHER_FORCING_Ratio=None, TEST=True, VALIDATE=False, NUM_CONFIG=MAX_SEQ_LEN, SEARCH_MODE='greedy')  # N, max_label_L, vocab_size; N, max_label_L; with grad; predict till EOS; NUM_CONFIG == MAX_SEQ_LEN
        
        # change label indexes to sentences
        # use the first EOS token as the end of sentence because when infering the whole batch, it stop as the last one meet EOS token
        prediction_sentences = []
        for prediction in prediction_labels.numpy():
            if char_language_model.EOS_token in prediction: # there exists EOS, use the first 1 as the end of sentence
                end_position = prediction.tolist().index(char_language_model.EOS_token)
                if end_position < len(prediction): # not the last one
                    pred = prediction[:end_position+1]
                else:
                    pred = prediction
            else:
                pred = prediction # if no EOS, use the entire prediction
            prediction_sentences.append(language_model.indexes2string(pred))
        # all predictions are done till EOS token; unsort the list and append
        inferences.extend(np.array(prediction_sentences)[reverse_sequence_order])
        
        # clear memory
        data_batch = [data.detach() for data in data_batch]
        del data_batch
        torch.cuda.empty_cache()
        
    return inferences

def inference_random_search(model, test_dataloader, language_model, validation_criterion, DEVICE, MAX_SEQ_LEN=500, N_CANDIDATES=100, GUMBEL_T=1.0):
    print('## Start inferencing....')
    model.eval()
    num_batches = len(test_dataloader)
    inferences = []
    for idx,  data_batch in enumerate(test_dataloader):
        candidate_y_hats, candidate_labels, sequence_order, reverse_sequence_order = model(
            data_batch, TEACHER_FORCING_Ratio=0, TEST=True, VALIDATE=False, NUM_CONFIG=(MAX_SEQ_LEN, N_CANDIDATES, GUMBEL_T), SEARCH_MODE='random')
        # N_candidates, batch_size, max_label_lens (vary in different try same within each try), vocab_size; N_candidates, batch_size, max_label_lens; NUM_CONFIG = (MAX_SEQ_LEN, N_CANDIDATES)
        
        # mask for cross entropy loss
        batch_size = len(sequence_order)
        
        # retrieve sorted utterance data
        utterance_data = np.array(data_batch)[sequence_order]

        prediction_labels = []
        # for each utterance, pick a candidate and update loss; choose using pilot dummy loss using candidate as dummy true label; reporting dummy_loss
        for utterance_index in range(batch_size):
            utterance_labels_candidates = candidate_labels[utterance_index*N_CANDIDATES:(utterance_index+1)*N_CANDIDATES]
            
            utterance_dummy_labels = []
            for candidate in utterance_labels_candidates:
                if char_language_model.EOS_token in candidate: # there exists EOS, use the first 1 as the end of sentence; the model will return a matrix, with the width of the last ended prediction
                    end_position = candidate.detach().numpy().tolist().index(char_language_model.EOS_token)
                    if end_position < len(candidate): # not the last one
                        utterance_dummy_label = candidate[:end_position+1]
                    else:
                        utterance_dummy_label = candidate
                else:
                    utterance_dummy_label = candidate # if no EOS, use the entire prediction
                utterance_dummy_labels.append(utterance_dummy_label)
            
            dummy_dataloader = [([utterance_data[utterance_index]]*len(utterance_dummy_labels), utterance_dummy_labels)]
            candidates_dummy_losses, candidates_dummy_distances, candidates_dummy_perplexity_losses = test_validation(model, validation_criterion, dummy_dataloader, language_model, DEVICE, PROMPT=False, RETURN_UTTERANCE_LEVEL=True) # testing with greedy search
            winner_index = np.argmin(candidates_dummy_distances[0]) # use 0th one because we have batch size 1
            
            utterance_pred = candidate_y_hats[utterance_index*N_CANDIDATES+winner_index, :, :] # label_len, vocab_size
            prediction_labels.append(utterance_dummy_labels[winner_index].detach().cpu().numpy())
            
        # change label indexes to sentences
        prediction_sentences = []
        for prediction in prediction_labels:
#             if char_language_model.EOS_token in prediction: # there exists EOS, use the first 1 as the end of sentence
#                 end_position = prediction.tolist().index(char_language_model.EOS_token)
#                 if end_position < len(prediction): # not the last one
#                     pred = prediction[:end_position+1]
#                 else:
#                     pred = prediction
#             else:
#                 pred = prediction # if no EOS, use the entire prediction
            prediction_sentences.append(language_model.indexes2string(prediction))
            
        inferences.extend(np.array(prediction_sentences)[reverse_sequence_order])
                    
        # clear memory
        data_batch = [data.detach() for data in data_batch]
        del data_batch
        torch.cuda.empty_cache()
        
    return inferences

def inference_beam_search(model, test_dataloader,language_model, MAX_SEQ_LEN=500, beam_size=5, num_candidates=1):
    print('## Start inferencing....')
    model.eval()
    num_batches = len(test_dataloader)
    inferences = []
    for idx,  data_batch in enumerate(test_dataloader):
        predictions, prediction_labels, _, reverse_sequence_order = model(
            data_batch, TEACHER_FORCING_Ratio=None, TEST=True, VALIDATE=False, NUM_CONFIG=(MAX_SEQ_LEN, beam_size, num_candidates), SEARCH_MODE='beam_search')  # N, max_label_L, vocab_size; N, max_label_L; with grad; predict till EOS; NUM_CONFIG == MAX_SEQ_LEN
        
        # change label indexes to sentences
        # use the first EOS token as the end of sentence because when infering the whole batch, it stop as the last one meet EOS token
        prediction_sentences = []
        for prediction in prediction_labels:
            if char_language_model.EOS_token in prediction: # there exists EOS, use the first 1 as the end of sentence
                end_position = prediction.tolist().index(char_language_model.EOS_token)
                if end_position < len(prediction): # not the last one
                    pred = prediction[:end_position+1]
                else:
                    pred = prediction
            else:
                pred = prediction # if no EOS, use the entire prediction
            prediction_sentences.append(language_model.indexes2string(pred))
        # all predictions are done till EOS token; unsort the list and append
        inferences.extend(np.array(prediction_sentences)[reverse_sequence_order])
        
        # clear memory
        data_batch = [data.detach() for data in data_batch]
        del data_batch
        torch.cuda.empty_cache()
        
    return inferences


def inference_fast_beam_search(model, model_reference, test_dataloader, language_model, MAX_SEQ_LEN=500, beam_size=5, num_candidates=1):
    '''use encoder to generate key and value pairs for all test data and use cpu multiprocessing to do beam search
    model_reference is on cpu and model is on gpu'''
    print('## Start inferencing....')
    model.eval()
    model_reference.eval()
    num_batches = len(test_dataloader)
    all_values, all_keys, all_sequence_lens = [], [], []
    

    # generate key and value pairs for all instances, unsort the sequence
    for idx,  data_batch in enumerate(test_dataloader):
        TEST=True
        encoder_argument_list = [data_batch, TEST]
        keys, values, final_seq_lens, sequence_order, reverse_sequence_order = model.encoder(encoder_argument_list)
        all_values.extend([value for value in values[reverse_sequence_order].detach().cpu()])
        all_keys.extend([key for key in keys[reverse_sequence_order].detach().cpu()])
        all_sequence_lens.extend([seq_len for seq_len in final_seq_lens[reverse_sequence_order].detach().cpu()])

        # clear memory
        data_batch = [data.detach() for data in data_batch]
        del data_batch
        torch.cuda.empty_cache()
        
    # initialize one step beam class
    # do this one cpu
    one_step_beam = OneStepBeam.OneStepBeam(
        model_reference, char_language_model.SOS_token, char_language_model.EOS_token, MAX_SEQ_LEN, beam_size, num_candidates)
    
#     # testing
#     return one_step_beam.do_one_step((
#         all_keys[0], all_values[0], all_sequence_lens[0]))

    # three list contains total number of key, value, and sequence length for each utterance
    # do beam search with multiprocessing
    pool = m.Pool(m.cpu_count())
    results = pool.map(one_step_beam.do_one_step, zip(
        all_keys, all_values, all_sequence_lens))  # [[y_hat, decoded_prediction, attention]]

    return results


def evaluate_distance(predictions, padded_label, label_lens, lang, SHOW_RESULT=False, RETURN_UTTERANCE_DISTANCE=False):
    """ predictions: N, Max_len
        padded_labels: N, Max_len
        label_lens: N, len"""
    prediction_checker = []
    ls = 0.0
    batch_distance = []
    batch_size = len(predictions)
    for i in range(batch_size): # for each instance
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
        utterance_distance = L.distance(pred, true)
        ls += utterance_distance
        batch_distance.append(utterance_distance)
        if i == 0:
            prediction_checker.extend([pred, true])
    if SHOW_RESULT:
        print("Pred: {}, True: {}".format(prediction_checker[0], prediction_checker[1]))
    if RETURN_UTTERANCE_DISTANCE:
        return ls / batch_size, batch_distance
    return ls / batch_size

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
