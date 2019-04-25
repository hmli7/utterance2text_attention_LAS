import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import char_language_model

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import util


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TODO:
# 1. teacher forcing
# 2. how to mask to loss
# 3. how to loss

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
        self.mlp1 = MLP(hidden_size*2, mlp_hidden_size, mlp_output_size)
        self.mlp2 = MLP(hidden_size*2, mlp_hidden_size, mlp_output_size)

    def forward(self, input_sequences):
        # x: N, L, E
        # sort sequence in descendent order
        # if batch_size == 1 and no batch dim
        x, labels, sorted_sequence_lens, sequence_order, reverse_sequence_order = util.collate_lines_for_test(
            input_sequences)

        # put to cuda
#         x = x.to(DEVICE) data have already been put onto cuda
#         sorted_sequence_lens = sorted_sequence_lens.to(DEVICE)
#         labels = labels.to(DEVICE)
        # pad sequence
        x = pad_sequence(x)
        x = pack_padded_sequence(x, sorted_sequence_lens)
        for rnn in self.rnns:
            x, _ = rnn(x)  # L/8, B, hidden_size * 2
        # unpack
        x, final_seq_lens = pad_packed_sequence(x)
        keys = self.mlp1(x.permute(1, 0, 2))
        values = self.mlp2(x.permute(1, 0, 2))
        return keys, values, labels, final_seq_lens, sequence_order, reverse_sequence_order


class pBLSTM(nn.Module):
    '''pyramid structure with a factor of 0.5 (L, B, F) -> (L/2, B, F*2)'''

    def __init__(self, input_dim, hidden_size):
        super(pBLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim*2, hidden_size=hidden_size, num_layers=1, bidirectional=True)

    def forward(self, x):
        # unpack
        x, sorted_sequence_lens = pad_packed_sequence(x)  # L, B, F
        seq_length, batch_size, feature_dim = x.size()

        # change original sequence lengths
        x = x[:(seq_length//2)*2]
        sorted_sequence_lens = sorted_sequence_lens//2

        # reduce the timestep by 2
        x = x.permute(1, 0, 2)  # L, B, F -> B, L, F
        x = x.contiguous().view(batch_size, int(seq_length//2), feature_dim*2)  # B, L/2, F*2
        # pad
        x = pad_sequence(x)
        # pack
        x = pack_padded_sequence(x, sorted_sequence_lens)
        return self.lstm(x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    '''different mechanisms to get attention
    energy = bmm(keys, values)
    attention = softmax(energy)
    use dot
    assume taking in key and score pair after two fc layers'''

    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, key, query):
        # (N, key length , dimension)  (N, query_len, dimensions)
        energy = self.score(key, query)
        # normalize
        attention = F.softmax(energy, dim=1)# apply softmax on key time len
        return attention

    def score(self, key, query):
        # use dot product
        energy = torch.bmm(key, query.transpose(1, 2).contiguous())  # B, key_len, query_len
        return energy


class Decoder_RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, key_value_size, hidden_size, n_layers, mlp_hidden_size, padding_value):
        super(Decoder_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.key_value_size = key_value_size # key and value pair hidden size
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.padding_value = padding_value

        assert self.hidden_size == self.key_value_size # when calculating attention score, we need key_value_size equals to rnn output size

        # for transcripts
        self.embedding = nn.Embedding(
            self.vocab_size+1, self.embed_size) # , padding_idx=self.padding_value vocab_size + 1 for <padding value>; using index of <EOS> to pad

        # rnn cells
        self.rnns = nn.ModuleList()
        self.rnns.append(nn.LSTMCell(self.embed_size +
                                     self.key_value_size, self.hidden_size))
        for i in range(n_layers-1):
            self.rnns.append(nn.LSTMCell(self.hidden_size, self.hidden_size))

        self.attention = Attention()

        # fc layer
        self.mlp = MLP(self.hidden_size+self.key_value_size,
                       mlp_hidden_size, self.vocab_size)

    def forward(self, argument_list):
        key, value, labels, final_seq_lens, seq_order, reversed_seq_order, SOS_token, TEACHER_FORCING_Ratio, TEST = argument_list
        # labels have been sorted by seq_order
        # label lens for loss masking
        labels_lens = [len(label) for label in labels]
        # pad
        # sorted N, Label L; use -1 to pad, this will be initialized to zero in embedding
        labels_padded = pad_sequence(labels, padding_value=self.padding_value)
        # embedding
        labels_embedding = self.embedding(labels_padded)  # N, label L, emb

        # ------ Init ------
        # initialize attention, hidden states, memory states
        attention_context = self.init_states(
            key, hidden_size=key.size(2))  # N, Key/query_len
        # initialize hidden and memory states
        hidden_states = [self.init_states(key)]  # N, hidden_size
        memory_states = [self.init_states(key)]  # N, hidden_size
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]
        for i in range(self.n_layers-1):
            # we use single direction lstm cell in this case
            hidden_states.append(self.init_states(key))
            # [n_layers, N, Hidden_size]
            memory_states.append(self.init_states(key))

        max_label_len = labels_embedding.size(0)
        
        # query_mask, (0, max_length)
        batch_size, max_len = key.size(0), key.size(1)
        key_mask = torch.stack([torch.arange(0, max_len)
                                for i in range(batch_size)]).int()
        key_lens = torch.stack(
            [torch.full((1, max_len), length).squeeze(0) for length in final_seq_lens]).int()
        key_mask = key_mask < key_lens  # a matrix of 1 & 0; N, max_key_len
        key_mask = key_mask.unsqueeze(2).repeat(
            (1, 1, max_label_len)).float().to(DEVICE)  # expend to N, max_key_len, max_label_len; float for matrix mul when applying attention

        # ------ LAS ------
        # initialze for the first time step input
        y_hat_t_label = torch.LongTensor([SOS_token]*batch_size).to(DEVICE)  # N
        y_hat = []
        y_hat_label = []
        attentions = []
        # iterate through max possible label length
        for time_index in range(max_label_len):
            # decoding rnn
            # decide whether to use teacher forcing in this time step
            use_teacher_forcing = True if random.random() < TEACHER_FORCING_Ratio else False
            if not use_teacher_forcing:
                y_hat_t_embedding = self.embedding(y_hat_t_label) # N, Embedding
                rnn_input = torch.cat(
                    (y_hat_t_embedding, attention_context), dim=1)
            else:
                rnn_input = torch.cat(
                    (labels_embedding[time_index, :, :], attention_context), dim=1)

            # rnn
            hidden_states[0], memory_states[0] = self.rnns[0](
                rnn_input, (hidden_states[0], memory_states[0])
            )

            for hidden_layer in range(1, self.n_layers):
                hidden_states[hidden_layer], memory_states[hidden_layer] = self.rnns[hidden_layer](
                    hidden_states[hidden_layer - 1], (hidden_states[hidden_layer], memory_states[hidden_layer])
                )
            rnn_output = hidden_states[-1]  # N, hidden_size

            # apply attention to generate context
            attention_softmax_energy = self.attention(key, rnn_output.unsqueeze(1)) # N, key_len, query_len
            masked_softmax_energy = torch.mul(attention_softmax_energy, key_mask[:, :, time_index].unsqueeze(2)) # N, key_len, 1
            masked_softmax_energy = F.normalize(masked_softmax_energy, p=1, dim=1)
#             masked_softmax_energy = masked_softmax_energy/masked_softmax_energy.sum(dim=1, keepdim=True) # normalize key_time_len dimension by devided by sum
            # N, key_len, query_len * N, key_len, key_size
            attention_context = torch.bmm(masked_softmax_energy.permute(0,2,1), value) # N, 1, key_len * N, key_len, key_size => N, 1, key_size
            attention_context = attention_context.squeeze(1) # N, key_len

            y_hat_t = self.mlp(torch.cat((rnn_output, attention_context), dim=1)) # N, vocab_size
            y_hat.append(y_hat_t) # N, vocab_size
            attentions.append(masked_softmax_energy.detach())
            
            y_hat_t_label = torch.argmax(y_hat_t, dim=1).long() # N ; long tensor for embedding inputs
            y_hat_label.append(y_hat_t_label.detach().cpu())
        # concat predictions
        y_hat = torch.stack(y_hat, dim=1) # N, label_L, vocab_size
        if TEST:
            attentions = torch.cat(attentions, dim=2) # N, key_len, query_len
            return y_hat_label, labels_padded.permute(1,0).detach().cpu(), labels_lens, attentions
        return y_hat, labels_padded.permute(1,0), labels_lens, attentions

    def init_states(self, source_outputs, hidden_size=None):
        """initiate state using source outputs and hidden size"""
        batch_size = source_outputs.size(0)
        hidden_size = self.hidden_size if hidden_size is None else hidden_size
        initiation = source_outputs.new_zeros(batch_size, hidden_size)
        return initiation


class Sequence2Sequence(nn.Module):
    def __init__(self, embed_size, hidden_size, n_layers, n_plstm, mlp_hidden_size, mlp_output_size,
                 decoder_vocab_size, decoder_embed_size, decoder_hidden_size, decoder_n_layers, decoder_mlp_hidden_size, decoder_padding_value):
        super(Sequence2Sequence, self).__init__()

        self.encoder = Encoder_RNN(
            embed_size, hidden_size, n_layers, n_plstm, mlp_hidden_size, mlp_output_size)
        self.decoder = Decoder_RNN(
            decoder_vocab_size, decoder_embed_size, mlp_output_size, decoder_hidden_size, decoder_n_layers, decoder_mlp_hidden_size, decoder_padding_value)

    def forward(self, seq_batch, TEACHER_FORCING_Ratio=0, TEST=False):
        keys, values, labels, final_seq_lens, sequence_order, reverse_sequence_order = self.encoder(
            seq_batch)
        argument_list = [keys, values, labels, final_seq_lens, sequence_order,reverse_sequence_order, char_language_model.SOS_token, TEACHER_FORCING_Ratio, TEST]
        y_hat, labels_padded, labels_lens, attentions = self.decoder(argument_list)
        if TEST:
            pass
            # unsort y_hat and labels
        return y_hat, labels_padded, labels_lens, attentions
