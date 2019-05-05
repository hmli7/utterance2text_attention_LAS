import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import char_language_model

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import util
from loss import *
from s2s_speller_Gumbel import *
from s2s_speller_Baseline import *
from s2s_listener import *


import pdb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Sequence2Sequence(nn.Module):
    def __init__(self, embed_size, hidden_size, n_layers, n_plstm, mlp_hidden_size, mlp_output_size,
                 decoder_vocab_size, decoder_embed_size, decoder_hidden_size, decoder_n_layers, decoder_mlp_hidden_size, decoder_padding_value, GUMBEL_SOFTMAX=False, batch_size=64, INIT_STATE=True, device=DEVICE):
        super(Sequence2Sequence, self).__init__()
        
        self.decoder_padding_value = decoder_padding_value

        self.encoder = Encoder_RNN(
            embed_size, hidden_size, n_layers, n_plstm, mlp_hidden_size, mlp_output_size)
        if GUMBEL_SOFTMAX:
            self.decoder = Decoder_RNN_Gumbel(
                decoder_vocab_size, decoder_embed_size, mlp_output_size, decoder_hidden_size, decoder_n_layers, decoder_mlp_hidden_size, decoder_padding_value)
        else:
            self.decoder = Decoder_RNN(
				decoder_vocab_size, decoder_embed_size, mlp_output_size, decoder_hidden_size, decoder_n_layers, decoder_mlp_hidden_size, decoder_padding_value, batch_size, INIT_STATE, device)

    def forward(self, seq_batch, TEACHER_FORCING_Ratio=None, TEST=False, VALIDATE=False, NUM_CONFIG=0, SEARCH_MODE='greedy'):
        '''NUM_CONFIG means MAX_SEQ_LEN for greedy search, means (MAX_SEQ_LEN, N_CONDIDATES) for random search'''
        if TEST:
            # inferencing
            encoder_argument_list = [seq_batch, TEST]
            keys, values, final_seq_lens, sequence_order, reverse_sequence_order = self.encoder(
                encoder_argument_list)
            argument_list = [keys, values, final_seq_lens, sequence_order,
                             reverse_sequence_order, char_language_model.SOS_token, char_language_model.EOS_token, NUM_CONFIG, SEARCH_MODE]
            y_hat, y_hat_labels, attentions = self.decoder.inference(argument_list)
            return y_hat, y_hat_labels, sequence_order, reverse_sequence_order
        
        elif VALIDATE:
            # validation
            assert TEST == False
            encoder_argument_list = [seq_batch, TEST]
            keys, values, labels, final_seq_lens, sequence_order, reverse_sequence_order = self.encoder(
                encoder_argument_list)
            argument_list = [keys, values, final_seq_lens, sequence_order,
                             reverse_sequence_order, char_language_model.SOS_token, char_language_model.EOS_token, NUM_CONFIG, SEARCH_MODE] # if search mode is greedy, max_seq_len is a number, if randome, it's a tuple (max_seq_len, n_candidates)
            y_hat, y_hat_labels, attentions = self.decoder.inference(argument_list)
            labels_lens = [len(label) for label in labels]
            # pad
            # sorted N, Label L; use -1 to pad, this will be initialized to zero in embedding
            labels_padded = pad_sequence(labels, padding_value=self.decoder_padding_value).permute(1,0) # N, L
            labels_padded.requires_grad=False
            
            if SEARCH_MODE == 'random':
                return y_hat, y_hat_labels, labels_padded, labels_lens, sequence_order
            return y_hat, y_hat_labels, labels_padded, labels_lens, attentions
        
        else:
            assert TEST == False and VALIDATE == False
            encoder_argument_list = [seq_batch, TEST]
            keys, values, labels, final_seq_lens, sequence_order, reverse_sequence_order = self.encoder(
                encoder_argument_list)
            argument_list = [keys, values, labels, final_seq_lens, char_language_model.SOS_token, char_language_model.EOS_token, TEACHER_FORCING_Ratio, NUM_CONFIG, SEARCH_MODE]
            y_hat, y_hat_labels, labels_padded, labels_lens, attentions = self.decoder(argument_list)

            return y_hat, y_hat_labels, labels_padded, labels_lens, attentions
