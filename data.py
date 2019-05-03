import paths
import config
import UtteranceDataset

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import char_language_model

import importlib
reload_packages = [paths, config, UtteranceDataset, char_language_model]
for package in reload_packages:
    importlib.reload(package)

class Data():
    def __init__(self):
        self.LANG = None
    
    def get_loader(self, mode="train"):
        batch_size = config.batch_size
        print('Batch Size: ', batch_size)
        # load paths
        if mode == "train":
            data_path = paths.train_data_path
            labels_path = paths.train_labels_path
            shuffle = True
            drop_last = True # when trainning, we need to drop the last imcomplete batch because we need a fix batch_size through training
        if mode == "val":
            data_path = paths.valid_data_path
            labels_path = paths.valid_labels_path
            shuffle = False
            drop_last = False
        if mode == "test":
            data_path = paths.test_data_path
            labels_path = None
            shuffle = False
            batch_size = config.test_batch_size
            drop_last = False
        
        # load data
        data = np.load(data_path, encoding='bytes')
        if config.sanity:
            data = data[:150]

        if labels_path:
            if mode == "train":
                labels = self.prepare_training_transcripts(labels_path)
            else:
                assert self.LANG is not None
                labels = self.prepare_testing_transcripts(labels_path)
            if config.sanity:
                labels = labels[:150]

            print(data.shape, labels.shape)

            dataset = UtteranceDataset.FrameDataset(data, labels)
        else:
    #         dataset = TensorDataset(torch.tensor(data, dtype=torch.float))
            dataset = UtteranceDataset.FrameDataset(data)

        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, drop_last=drop_last, num_workers=0, collate_fn=self.collate_fn) # multiprocessing set to 0 because the data is already on GPU; putting data onto GPU in dataset because various length and we are using customized collate function. if using multiprocessing, need to manually iterate through one batch of data, put them onto gpu before putting input model.
        
        return dataloader


    def prepare_training_transcripts(self, lang_path):
        '''prepare training transcripts, built lang, process'''
        self.LANG, lines = self.read_lang(lang_path)
        # build corpus, decode bytes to strings and concat
        concatenated_decoded_lines = self.LANG.build_corpus(lines)
        # lines to indexes and add EOS token
        processed_lines = np.array([self.LANG.string2indexes(line_string) for line_string in concatenated_decoded_lines])
        return processed_lines

    def prepare_testing_transcripts(self, label_path):
        '''prepare testing transcripts, process'''
        _, lines = self.read_lang(label_path)
        # decode bytes to strings and concat
        # lines to indexes and add EOS token
        processed_lines = np.array([self.LANG.string2indexes(self.LANG.decode_line2string(line_string)) for line_string in lines])
        return processed_lines
    
    @ staticmethod
    def read_lang(lang_path):
        '''assume the dataset has already been tokenized
        Source: recitation 9'''
        print('Preparing corpus...')
        
        lines = np.load(lang_path, encoding='bytes')
        lang = char_language_model.Lang('English Labels')
        return lang, lines
    
    @ staticmethod
    def collate_fn(seq_list):
        """Function put into dataloader to solve the various length issue
        by default, dataloader will stack the batch of data into a tensor, which will cause error when sequence length is different"""
        if len(seq_list[0])==2:
            inputs, targets = zip(*seq_list)
            return list(inputs), list(targets)
        else:
            return seq_list
        # inputs = [item[0] for item in seq_list]
        # targets = [item[1] for item in seq_list]