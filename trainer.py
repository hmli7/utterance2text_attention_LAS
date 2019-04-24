import torch.nn as nn
import torch.nn.functional as F

import time
import os
import sys

import numpy as np
import config
from util import *
import paths


def run(model, optimizer, criterion, train_dataloader, valid_dataloader, best_epoch, best_vali_loss, DEVICE, scheduler=None, start_epoch=None, model_prefix=config.model_prefix):
    best_eval = None
    start_epoch = 0 if start_epoch is None else start_epoch
    max_epoch = config.max_epoch
    batch_size = config.batch_size

    model = model.to(DEVICE)
    criterion.to(DEVICE)

    if scheduler is not None:
        print('-- Starting training with scheduler.')

    for epoch in range(start_epoch, max_epoch+1):
        start_time = time.time()
        model.train()
        # outputs records
        f = open(os.path.join(paths.output_path, 'metrics.txt'), 'a')
        print_file_and_screen('### Epoch %5d' % (epoch), f=f)

        avg_loss = 0
        num_batches = len(train_dataloader)
        # lists, presorted, preloaded on GPU
        for idx, (data_batch, label_batch) in enumerate(train_dataloader):
            optimizer.zero_grad()
            predictions, sorted_labels, labels_lens = model(
                data_batch, TEACHER_FORCING_Ratio=0, TEST=False)  # N, max_label_L, vocab_size; N, max_label_L

            # mask for cross entropy loss
            batch_size, max_label_lens, _ = predictions.size()
            prediction_mask = torch.stack(
                [torch.arange(0, max_label_lens) for i in range(batch_size)]).int()
            prediction_mask = prediction_mask < torch.stack([torch.full(
                (1, max_label_lens), length).squeeze(0) for length in labels_lens]).int()

            # use cross entropy loss, assume set (reduce = False)
            batch_loss = criterion.forward(predictions, sorted_labels)
            batch_loss = batch_loss * prediction_mask
            loss = batch_loss.sum()/prediction_mask.sum()

            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

            if idx % 200 == 199:
                print_file_and_screen('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(
                    epoch, idx+1, avg_loss/200), f=f)
                avg_loss = 0.0
            # clear memory
            torch.cuda.empty_cache()
            data_batch = data_batch.detach()
            label_batch = label_batch.detach()
            del data_batch, label_batch
            del loss

        train_loss, train_distance = test_validation(
            model, criterion, train_dataloader)
        val_loss, val_distance = test_validation(
            model, criterion, valid_dataloader)
        print_file_and_screen('Train Loss: {:.4f}\tTrain Distance: {:.4f}\tVal Loss: {:.4f}\tVal Distance: {:.4f}'.format(
            train_loss, train_distance, val_loss, val_distance), f=f)

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


def test_validation(model, criterion, valid_dataloader):
    # TODO: fix L distance calculation
    model.eval()
    num_batches = len(valid_dataloader)
    avg_loss = 0.0
    avg_distance = 0.0
    for idx, batch in enumerate(valid_dataloader):
        text, text_lengths = batch.text
        predictions = model(text, text_lengths, testing=False)
        predictions = predictions.squeeze(1) if len(
            predictions.size()) > 1 else predictions  # prepare for batch size == 1
        loss = criterion.forward(predictions, batch.label)
        distance = None
        avg_loss += loss.item()
        avg_distance += 0
    return avg_loss/num_batches, 0/num_batches


def predict(model, test_dataloader, DEVICE):
    model.to(DEVICE)
    model.eval()
    prediction = []
    for i, batch in enumerate(test_dataloader):
        if i % 400 == 0:
            print(i)
        text, text_lengths = batch.text
        predictions_batch = model(text, text_lengths, testing=True)
        predictions_batch = predictions_batch.squeeze(1) if len(
            predictions_batch.size()) > 1 else predictions_batch  # prepare for batch size == 1
        rounded_preds = torch.round(torch.sigmoid(predictions_batch))
#         rounded_preds = np.round(1 / (1 + np.exp(-predictions_batch.detach().cpu().numpy())))
        prediction.append(rounded_preds)
#     return np.hstack(prediction)
    return torch.cat(prediction, dim=0).detach().cpu().numpy()
