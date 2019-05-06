import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import char_language_model

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import util

import pdb

from dropouts import LockedDropout, WeightDrop

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder_RNN(nn.Module):
    def __init__(self, embed_size, hidden_size, n_layers, n_plstm, mlp_hidden_size, mlp_output_size):
        super(Encoder_RNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.rnns = nn.ModuleList()
        self.rnns.append(nn.LSTM(input_size=embed_size,
                                 hidden_size=hidden_size, num_layers=n_layers, bidirectional=True))
        for i in range(n_plstm):
            # add plstm layers
            self.rnns.append(pBLSTM(hidden_size*2, hidden_size))
#         self.mlp1 = MLP(hidden_size*2, mlp_hidden_size, mlp_output_size)
#         self.mlp2 = MLP(hidden_size*2, mlp_hidden_size, mlp_output_size)
        self.fc1 = nn.Linear(hidden_size*2, mlp_output_size)
        self.fc2 = nn.Linear(hidden_size*2, mlp_output_size)

    def forward(self, arguments_list):
        input_sequences, TEST = arguments_list
        # x: N, L, E
        # sort sequence in descendent order
        # if batch_size == 1 and no batch dim
        if TEST:
            x, sorted_sequence_lens, sequence_order, reverse_sequence_order = util.sort_instances(
                input_sequences)
        else:
            x, labels, sorted_sequence_lens, sequence_order, reverse_sequence_order = util.sort_instances(
                input_sequences)

        # put to cuda
#         x = x.to(DEVICE) data have already been put onto cuda
#         sorted_sequence_lens = sorted_sequence_lens.to(DEVICE)
#         labels = labels.to(DEVICE)
        # pad sequence
        x = pad_sequence(x) # L, N, H
        x = pack_padded_sequence(x, sorted_sequence_lens)
        for rnn in self.rnns:
            x, _ = rnn(x)  # L/8, B, hidden_size * 2
        # unpack
        x, final_seq_lens = pad_packed_sequence(x)
        keys = F.leaky_relu(self.fc1(x.permute(1, 0, 2)), negative_slope=1, inplace=False)
        values = F.leaky_relu(self.fc2(x.permute(1, 0, 2)), negative_slope=1, inplace=False)
        if TEST:
            return keys, values, final_seq_lens, sequence_order, reverse_sequence_order
        return keys, values, labels, final_seq_lens, sequence_order, reverse_sequence_order


class pBLSTM(nn.Module):
    '''pyramid structure with a factor of 0.5 (L, B, F) -> (L/2, B, F*2)'''

    def __init__(self, input_dim, hidden_size):
        super(pBLSTM, self).__init__()
        
        lstm = nn.LSTM(
            input_size=input_dim*2, hidden_size=hidden_size, num_layers=1, bidirectional=True)
        self.dropped_lstm = WeightDrop(lstm, ['weight_hh_l0','weight_hh_l0_reverse'],
                              dropout=0.1, variational=True)
        self.locked_dropout = LockedDropout()

    def forward(self, x):
        # unpack
        x, sorted_sequence_lens = pad_packed_sequence(x)  # L, B, F
        seq_length, batch_size, feature_dim = x.size()

        # change original sequence lengths
        x = x[:(seq_length//2)*2]
        sorted_sequence_lens = sorted_sequence_lens//2
        x = self.locked_dropout(x, dropout=.05) # L, N, H

        # reduce the timestep by 2
        x = x.permute(1, 0, 2)  # L, B, F -> B, L, F
        x = x.contiguous().view(batch_size, int(seq_length//2), feature_dim*2)  # B, L/2, F*2
#         # pad
#         x = pad_sequence(x)
        # pack
        x = pack_padded_sequence(x.permute(1,0,2), sorted_sequence_lens)
        return self.dropped_lstm(x)
