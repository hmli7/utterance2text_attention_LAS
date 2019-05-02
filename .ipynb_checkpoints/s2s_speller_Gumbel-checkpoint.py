import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import char_language_model

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import util
from loss import *
import pdb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu = nn.LeakyReLU(negative_slope=0.9, inplace=False)
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

    def forward(self, key, query, key_mask):
        # (N, key length , dimension)  (N, query_len, dimensions)
        energy = self.score(key, query)
        
        #---softmax before masking
#         # normalize
#         attention_softmax_energy = F.softmax(energy, dim=1)# apply softmax on key time len
# #             masked_softmax_energy = torch.mul(attention_softmax_energy, key_mask[:, :, time_index].unsqueeze(2)) # N, key_len, 1
# #             masked_softmax_energy = attention_softmax_energy * key_mask[:, :, time_index].unsqueeze(2)
#         masked_softmax_energy = attention_softmax_energy * key_mask
#         masked_softmax_energy = F.normalize(masked_softmax_energy, p=1, dim=1)
# #             masked_softmax_energy = masked_softmax_energy/masked_softmax_energy.sum(dim=1, keepdim=True) # normalize key_time_len dimension by devided by sum

        #----softmax after masking
        masked_energy = energy * key_mask
        masked_softmax_energy = F.softmax(masked_energy, dim=1)
        return masked_softmax_energy

    def score(self, key, query):
        # use dot product
        energy = torch.bmm(key, query.permute(0,2,1))  # B, key_len, query_len
        return energy

class Decoder_RNN_Gumbel(nn.Module):
    def __init__(self, vocab_size, embed_size, key_value_size, hidden_size, n_layers, mlp_hidden_size, padding_value):
        super(Decoder_RNN_Gumbel, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.key_value_size = key_value_size  # key and value pair hidden size
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.padding_value = padding_value

#         assert self.hidden_size == self.key_value_size # when calculating attention score, we need key_value_size equals to rnn output size
        #gumble softmax for onehot smoothing
        self.gumble_softmax = Gumbel_Softmax()
        
#         # for transcripts
#         self.embedding_1 = nn.Embedding(
#             self.vocab_size, self.embed_size) # , padding_idx=self.padding_value  using index of <EOS> to pad
# #         self.embedding_1.requres_grad = False # do not train label embedding
        # for y_hat
        self.embedding = nn.Linear(
            self.vocab_size, self.embed_size)  # , padding_idx=self.padding_value vocab_size + 1 for <padding value>; using index of <EOS> to pad
        
        # rnn cells
        self.rnns = nn.ModuleList()
        self.rnns.append(nn.LSTMCell(self.embed_size +
                                     self.key_value_size, self.hidden_size))
        for i in range(n_layers-1):
            self.rnns.append(nn.LSTMCell(self.hidden_size, self.hidden_size))

        self.attention = Attention()

        # fc layer to map demention of query with dimention of key and value
        self.fc = nn.Linear(self.hidden_size, self.key_value_size)

        # fc layer
        self.mlp = MLP(self.key_value_size*2, mlp_hidden_size, self.vocab_size)
#         self.fc2 = nn.Linear(self.key_value_size*2, self.vocab_size)

    def forward(self, argument_list):
        '''control which forward method to use'''
        SEARCH_MODE = argument_list[-1]
        if SEARCH_MODE == 'greedy':
            return self.forward_greedy_search(argument_list[:-2])
        elif SEARCH_MODE == 'beam_search':
            NotImplemented
            return self.forward_beam_search(argument_list[:-1])
        else:
            NotImplemented

    def forward_greedy_search(self, argument_list):
        """only used to train other than inferencing"""
        key, value, labels, final_seq_lens, SOS_token, _, TEACHER_FORCING_Ratio = argument_list
        batch_size, max_key_len = key.size(0), key.size(1)
        # labels have been sorted by seq_order
        # label lens for loss masking
        labels_lens = [len(label) for label in labels]
        # pad
        # sorted N, Label L; use -1 to pad, this will be initialized to zero in embedding
        labels_padded = pad_sequence(labels, padding_value=self.padding_value) # L, N
        
        # <--- previous add noise to ground truth label version --->
        # onehot encoding for ground truth
        y_onehot = torch.FloatTensor(labels_padded.size(0), batch_size, self.vocab_size).to(DEVICE)
        # initialize to zeros
        y_onehot.zero_()
        # onehot_encoding
        labels_onehot = y_onehot.scatter_(2, labels_padded.unsqueeze(2), 1) # onehot on dim2 with 1; L, N, Vocabsize
        labels_onehot.requires_grad = False # don't learn onehot labels
#         # onehot smoothing
#         # labels_smoothed = F.gumbel_softmax(labels_onehot, tau=1, hard=False, eps=1e-10) # functional gumble softmax only support 2 dimensions input
#         labels_smoothed = self.gumble_softmax(
#             labels_onehot, temperature=1.0, eps=1e-10) # L, N, Vocabsize
        labels_embedding = self.embedding(labels_onehot)

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

        max_label_len = labels_padded.size(0)

        # query_mask, (0, max_length)
        
#         key_mask = torch.stack([torch.arange(0, max_len)
#                                 for i in range(batch_size)]).int()
#         key_lens = torch.stack(
#             [torch.full((1, max_len), length).squeeze(0) for length in final_seq_lens]).int()
#         key_mask = key_mask < key_lens  # a matrix of 1 & 0; N, max_key_len
#         key_mask = key_mask.unsqueeze(2).repeat(
#             (1, 1, max_label_len)).float().to(DEVICE)  # expend to N, max_key_len, max_label_len; float for matrix mul when applying attention
        key_mask = torch.arange(0, max_key_len) < final_seq_lens.view(
            len(final_seq_lens), -1)
        key_mask = key_mask.unsqueeze(2).float().to(
            DEVICE)  # N, max_key_len, 1
        key_mask.requires_grad = False

        # ------ LAS ------
        # initialze for the first time step input
        # initialze for the first time step input
        y_hat_t_label = torch.LongTensor(
            [SOS_token]*batch_size).to(DEVICE)  # N

        # onehot encoding for <SOS>
        y_onehot = torch.FloatTensor(batch_size, self.vocab_size).to(DEVICE)
        # initialize to zeros
        y_onehot.zero_()
        # onehot_encoding
        y_hat_t_onehot = y_onehot.scatter_(1, y_hat_t_label.unsqueeze(1), 1) # onehot on dim2 with 1; L, N, Vocabsize
        y_hat_t_onehot.requires_grad = False # don't learn onehot labels
#         # onehot smoothing
#         # labels_smoothed = F.gumbel_softmax(labels_onehot, tau=1, hard=False, eps=1e-10) # functional gumble softmax only support 2 dimensions input
#         y_hat_t = self.gumble_softmax(
#             y_hat_t_onehot, temperature=1.0, eps=1e-10) # L, N, Vocabsize
#         labels_embedding = self.embedding(y_hat_t)  # N, Embedding
        y_hat_t_embedding = self.embedding(y_hat_t_onehot)
        rnn_input = torch.cat(
            (y_hat_t_embedding, attention_context), dim=1)
        y_hat = []
        y_hat_label = []
        attentions = []
        # iterate through max possible label length
        for time_index in range(max_label_len):
            # decoding rnn

            # rnn
            hidden_states[0], memory_states[0] = self.rnns[0](
                rnn_input, (hidden_states[0], memory_states[0])
            )

            for hidden_layer in range(1, self.n_layers):
                hidden_states[hidden_layer], memory_states[hidden_layer] = self.rnns[hidden_layer](
                    hidden_states[hidden_layer -
                                  1], (hidden_states[hidden_layer], memory_states[hidden_layer])
                )
            rnn_output = hidden_states[-1]  # N, hidden_size

            # N, query_len, key_value_size
            query = self.fc(rnn_output.unsqueeze(1))

            # apply attention to generate context
            masked_softmax_energy = self.attention(
                key, query, key_mask)  # N, key_len, query_len

            # N, key_len, query_len * N, key_len, key_size
            # N, 1, key_len * N, key_len, key_size => N, 1, key_size
            attention_context = torch.bmm(
                masked_softmax_energy.permute(0, 2, 1), value)
            attention_context = attention_context.squeeze(1)  # N, key_len

            y_hat_t = self.mlp(
                torch.cat((query.squeeze(1), attention_context), dim=1))  # N, vocab_size
#             y_hat_t = self.fc2(torch.cat((query.squeeze(1), attention_context), dim=1)) # N, vocab_size
            y_hat.append(y_hat_t)  # N, vocab_size
            attentions.append(masked_softmax_energy.detach().cpu())
            y_hat_smoothed = self.gumble_softmax(y_hat_t, temperature=1.0, eps=1e-10) # L, N, Vocabsize

            # N ; long tensor for embedding inputs
            y_hat_t_label = torch.argmax(y_hat_smoothed, dim=1).long()
            y_hat_label.append(y_hat_t_label.detach().cpu())
            
            # decide whether to use teacher forcing in this time step
            use_teacher_forcing = True if random.random() < TEACHER_FORCING_Ratio else False
            if not use_teacher_forcing:
                y_hat_t_embedding = self.embedding(
                    y_hat_smoothed)  # N, Embedding
                rnn_input = torch.cat(
                    (y_hat_t_embedding, attention_context), dim=1)
            else:
                # embedding
#                 label_embedding = self.embedding(
#                     labels_smoothed[time_index, :, :])  # N, emb
                y_hat_t_embedding = labels_embedding[time_index, :, :]
                rnn_input = torch.cat(
                    (y_hat_t_embedding, attention_context), dim=1)


        # concat predictions
        y_hat = torch.stack(y_hat, dim=1)  # N, label_L, vocab_size
        attentions = torch.cat(attentions, dim=2)  # N, key_len, query_len
        y_hat_label = torch.stack(y_hat_label, dim=1)  # N, label_L

        return y_hat, y_hat_label, labels_padded.permute(1, 0), labels_lens, attentions

    def inference(self, argument_list):
        '''the inference function allocation right function using search mode'''
        SEARCH_MODE = argument_list[-1]
        with torch.no_grad():
            if SEARCH_MODE == 'greedy':
                y_hat, y_hat_label, attentions = self.inference_greedy_search(argument_list)
                return y_hat, y_hat_label, attentions  
            elif SEARCH_MODE == 'random':
                candidate_y_hats, candidate_labels, attentions = self.inference_random_search(argument_list)
                return candidate_y_hats, candidate_labels, attentions
            else:
                NotImplemented

    def inference_greedy_search(self, argument_list):
        """ when tesing"""
        key, value, final_seq_lens, seq_order, reversed_seq_order, SOS_token, EOS_token, MAX_SEQ_LEN, SEARCH_MODE = argument_list
        assert SEARCH_MODE == 'greedy'

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

        max_label_len = MAX_SEQ_LEN

        # query_mask, (0, max_length)
        batch_size, max_key_len = key.size(0), key.size(1)
#         key_mask = torch.stack([torch.arange(0, max_key_len)
#                                 for i in range(batch_size)]).int()
#         key_lens = torch.stack(
#             [torch.full((1, max_key_len), length).squeeze(0) for length in final_seq_lens]).int()
#         key_mask = key_mask < key_lens  # a matrix of 1 & 0; N, max_key_len
#         key_mask = key_mask.unsqueeze(2).repeat(
#             (1, 1, max_label_len)).float().to(DEVICE)  # expend to N, max_key_len, max_label_len; float for matrix mul when applying attention
        key_mask = torch.arange(0, max_key_len) < final_seq_lens.view(
            len(final_seq_lens), -1)
        key_mask = key_mask.unsqueeze(2).float().to(
            DEVICE)  # N, max_key_len, 1
        key_mask.requires_grad = False

        # ------ LAS ------
        # initialze for the first time step input
        y_hat_t_label = torch.LongTensor(
            [SOS_token]*batch_size).to(DEVICE)  # N
        
        # onehot encoding for <SOS>
        y_onehot = torch.FloatTensor(batch_size, self.vocab_size).to(DEVICE)
        # initialize to zeros
        y_onehot.zero_()
        # onehot_encoding
        y_hat_t_onehot = y_onehot.scatter_(1, y_hat_t_label.unsqueeze(1), 1) # onehot on dim2 with 1; L, N, Vocabsize
        y_hat_t_onehot.requires_grad = False # don't learn onehot labels

        y_hat_t_embedding = self.embedding(y_hat_t_onehot)
        rnn_input = torch.cat(
            (y_hat_t_embedding, attention_context), dim=1)

        y_hat = []
        y_hat_label = []
        attentions = []
        finished_instances = [] # use to end inferencing before max label len
        # iterate through max possible label length
        for time_index in range(max_label_len):
            # decoding rnn

            # rnn
            hidden_states[0], memory_states[0] = self.rnns[0](
                rnn_input, (hidden_states[0], memory_states[0])
            )

            for hidden_layer in range(1, self.n_layers):
                hidden_states[hidden_layer], memory_states[hidden_layer] = self.rnns[hidden_layer](
                    hidden_states[hidden_layer -
                                  1], (hidden_states[hidden_layer], memory_states[hidden_layer])
                )
            rnn_output = hidden_states[-1]  # N, hidden_size

            # N, query_len, key_value_size
            query = self.fc(rnn_output.unsqueeze(1))

            # apply attention to generate context
            masked_softmax_energy = self.attention(
                key, query, key_mask)  # N, key_len, query_len

            # N, key_len, query_len * N, key_len, key_size
            # N, 1, key_len * N, key_len, key_size => N, 1, key_size
            attention_context = torch.bmm(
                masked_softmax_energy.permute(0, 2, 1), value)
            attention_context = attention_context.squeeze(1)  # N, key_len

            y_hat_t = self.mlp(
                torch.cat((query.squeeze(1), attention_context), dim=1))  # N, vocab_size
#             y_hat_t = self.fc2(torch.cat((query.squeeze(1), attention_context), dim=1)) # N, vocab_size
            y_hat.append(y_hat_t)  # N, vocab_size
            attentions.append(masked_softmax_energy.detach().cpu())
        
            y_hat_smoothed = self.gumble_softmax(y_hat_t, temperature=1.0, eps=1e-10) # L, N, Vocabsize
            
            # N ; long tensor for embedding inputs: greedy output
            y_hat_t_label = torch.argmax(y_hat_smoothed, dim=1).long()
            
            y_hat_label.append(y_hat_t_label.detach().cpu())

            if EOS_token in y_hat_t_label: # if some one ended
                finished_instances.extend((y_hat_t_label == EOS_token).nonzero().squeeze(1).detach().cpu().numpy().tolist())
                finished_instances = list(set(finished_instances))
                if len(finished_instances) == batch_size: # inferencing of all instances meet the end token, stop
                    break
            
            # use output as the input of next time step, teacher_forcing_ratio == 0
            y_hat_t_embedding = self.embedding(y_hat_smoothed)  # N, Embedding
            rnn_input = torch.cat(
                (y_hat_t_embedding, attention_context), dim=1)

        # concat predictions
        y_hat = torch.stack(y_hat, dim=1)  # N, label_L, vocab_size
        attentions = torch.cat(attentions, dim=2)  # N, key_len, query_len
        y_hat_label = torch.stack(y_hat_label, dim=1)  # N, label_L

        return y_hat, y_hat_label, attentions
    
#     def inference_random_search(self, argument_list):
#         """ return y_hat adding gumbel noises, y_hat_labels using gumbel noises"""
#         key, value, final_seq_lens, seq_order, reversed_seq_order, SOS_token, EOS_token, (max_label_len, num_candidates), SEARCH_MODE = argument_list
#         assert SEARCH_MODE == 'random'

#         # ------ Init ------
#         # initialize attention, hidden states, memory states
#         attention_context = self.init_states(
#             key, hidden_size=key.size(2))  # N, Key/query_len
#         # initialize hidden and memory states
#         hidden_states = [self.init_states(key)]  # N, hidden_size
#         memory_states = [self.init_states(key)]  # N, hidden_size
#         # hidden = [num layers * num directions, batch size, hid dim]
#         # cell = [num layers * num directions, batch size, hid dim]
#         for i in range(self.n_layers-1):
#             # we use single direction lstm cell in this case
#             hidden_states.append(self.init_states(key))
#             # [n_layers, N, Hidden_size]
#             memory_states.append(self.init_states(key))


#         # query_mask, (0, max_length)
#         batch_size, max_key_len = key.size(0), key.size(1)
# #         key_mask = torch.stack([torch.arange(0, max_key_len)
# #                                 for i in range(batch_size)]).int()
# #         key_lens = torch.stack(
# #             [torch.full((1, max_key_len), length).squeeze(0) for length in final_seq_lens]).int()
# #         key_mask = key_mask < key_lens  # a matrix of 1 & 0; N, max_key_len
# #         key_mask = key_mask.unsqueeze(2).repeat(
# #             (1, 1, max_label_len)).float().to(DEVICE)  # expend to N, max_key_len, max_label_len; float for matrix mul when applying attention
#         key_mask = torch.arange(0, max_key_len) < final_seq_lens.view(
#             len(final_seq_lens), -1)
#         key_mask = key_mask.unsqueeze(2).float().to(
#             DEVICE)  # N, max_key_len, 1
#         key_mask.requires_grad = False
        
#         # softmax for random search
#         gumble_softmax = Gumbel_Softmax()

#         candidate_labels = []
#         candidate_y_hats = []
        
#         # ------ generate number of output with gumbel noise ------
#         for candidate_index in range(num_candidates):

#             # ------ LAS ------
#             # initialze for the first time step input
#             y_hat_t_label = torch.LongTensor(
#                 [SOS_token]*batch_size).to(DEVICE)  # N

#             # onehot encoding for <SOS>
#             y_onehot = torch.FloatTensor(batch_size, self.vocab_size).to(DEVICE)
#             # initialize to zeros
#             y_onehot.zero_()
#             # onehot_encoding
#             y_hat_t_onehot = y_onehot.scatter_(1, y_hat_t_label.unsqueeze(1), 1) # onehot on dim2 with 1; L, N, Vocabsize
#             y_hat_t_onehot.requires_grad = False # don't learn onehot labels
#             # onehot smoothing
#             # labels_smoothed = F.gumbel_softmax(labels_onehot, tau=1, hard=False, eps=1e-10) # functional gumble softmax only support 2 dimensions input
#             y_hat_t = self.gumble_softmax(
#                 y_hat_t_onehot, temperature=1.0, eps=1e-10) # L, N, Vocabsize

#             # use last output as input, teacher_forcing_ratio == 0
#             y_hat_t_embedding = self.embedding(y_hat_t)  # N, Embedding
#             rnn_input = torch.cat(
#                 (y_hat_t_embedding, attention_context), dim=1)

#             y_hat = []
#             y_hat_label = []
#             attentions = []
#             finished_instances = [] # use to end inferencing before max label len
#             # iterate through max possible label length
#             for time_index in range(max_label_len):
#                 # decoding rnn

#                 # rnn
#                 hidden_states[0], memory_states[0] = self.rnns[0](
#                     rnn_input, (hidden_states[0], memory_states[0])
#                 )

#                 for hidden_layer in range(1, self.n_layers):
#                     hidden_states[hidden_layer], memory_states[hidden_layer] = self.rnns[hidden_layer](
#                         hidden_states[hidden_layer -
#                                       1], (hidden_states[hidden_layer], memory_states[hidden_layer])
#                     )
#                 rnn_output = hidden_states[-1]  # N, hidden_size

#                 # N, query_len, key_value_size
#                 query = self.fc(rnn_output.unsqueeze(1))

#                 # apply attention to generate context
#                 masked_softmax_energy = self.attention(
#                     key, query, key_mask)  # N, key_len, query_len

#                 # N, key_len, query_len * N, key_len, key_size
#                 # N, 1, key_len * N, key_len, key_size => N, 1, key_size
#                 attention_context = torch.bmm(
#                     masked_softmax_energy.permute(0, 2, 1), value)
#                 attention_context = attention_context.squeeze(1)  # N, key_len

#                 y_hat_t = self.mlp(
#                     torch.cat((query.squeeze(1), attention_context), dim=1))  # N, vocab_size
#     #             y_hat_t = self.fc2(torch.cat((query.squeeze(1), attention_context), dim=1)) # N, vocab_size
#                 attentions.append(masked_softmax_energy.detach().cpu())
                
#                 # add gumbel noise
#                 y_hat_t_gumbel = gumble_softmax(y_hat_t, temperature=1.0, eps=1e-10) # L, N, Vocabsize
#                 y_hat.append(y_hat_t_gumbel.detach()) # N, vocab_size
#                 y_hat_t_label = torch.argmax(y_hat_t_gumbel, dim=1).long() # N ; long tensor for embedding inputs: greedy output

#                 y_hat_label.append(y_hat_t_label.detach().cpu())

#                 if EOS_token in y_hat_t_label: # if some one ended
#                     finished_instances.extend((y_hat_t_label == EOS_token).nonzero().squeeze(1).detach().cpu().numpy().tolist())
#                     finished_instances = list(set(finished_instances))
#                     if len(finished_instances) == batch_size: # inferencing of all instances meet the end token, stop
#                         break

#                 # use output as the input of next time step, teacher_forcing_ratio == 0
#                 y_hat_t_embedding = self.embedding(y_hat_t_gumbel)  # N, Embedding
#                 rnn_input = torch.cat(
#                     (y_hat_t_embedding, attention_context), dim=1)


#             # concat predictions
#             y_hat = torch.stack(y_hat, dim=1)  # N, label_L, vocab_size
#             attentions = torch.cat(attentions, dim=2)  # N, key_len, query_len
#             y_hat_label = torch.stack(y_hat_label, dim=1)  # N, label_L

#             # store results to pool
#             candidate_labels.append(y_hat_label) # NumCandidate, N, Prediction_L # Prediction_L varies for different instances
#             candidate_y_hats.append(y_hat) # # NumCandidate, N, Prediction_L, vocab_size
#             # not storing attention to save memory

#         # TODO: currently lazily return none for attentions for memory consideration
#         return candidate_y_hats, candidate_labels, None
    def inference_random_search(self, argument_list):
        """ return y_hat adding gumbel noises, y_hat_labels using gumbel noises"""
        key, value, final_seq_lens, seq_order, reversed_seq_order, SOS_token, EOS_token, (max_label_len, num_candidates), SEARCH_MODE = argument_list
        assert SEARCH_MODE == 'random'
        
        # ------ generate duplicated data as candidate inputs ------
        dup_key = self.duplicate(key, num_candidates)
        dup_value = self.duplicate(value, num_candidates)
        dup_final_seq_lens = self.duplicate(final_seq_lens, num_candidates)

        # ------ Init ------
        # initialize attention, hidden states, memory states
        attention_context = self.init_states(dup_key, hidden_size=dup_key.size(2))  # N, Key/query_len
        # initialize hidden and memory states
        hidden_states = [self.init_states(dup_key)]  # N, hidden_size
        memory_states = [self.init_states(dup_key)]  # N, hidden_size
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]
        for i in range(self.n_layers-1):
            # we use single direction lstm cell in this case
            hidden_states.append(self.init_states(dup_key))
            # [n_layers, N, Hidden_size]
            memory_states.append(self.init_states(dup_key))


        # query_mask, (0, max_length)
        batch_size, max_len = dup_key.size(0), dup_key.size(1)
    #         key_mask = torch.stack([torch.arange(0, max_len)
    #                                 for i in range(batch_size)]).int()
    #         key_lens = torch.stack(
    #             [torch.full((1, max_len), length).squeeze(0) for length in final_seq_lens]).int()
    #         key_mask = key_mask < key_lens  # a matrix of 1 & 0; N, max_key_len
    #         key_mask = key_mask.unsqueeze(2).repeat(
    #             (1, 1, max_label_len)).float().to(DEVICE)  # expend to N, max_key_len, max_label_len; float for matrix mul when applying attention
        key_mask = torch.arange(0, max_len) < dup_final_seq_lens.view(len(dup_final_seq_lens),-1)
        key_mask = key_mask.unsqueeze(2).float().to(DEVICE) # N, max_key_len, 1
        key_mask.requires_grad = False

        gumble_softmax = Gumbel_Softmax()


        # ------ LAS ------
        # initialze for the first time step input
        y_hat_t_label = torch.LongTensor([SOS_token]*batch_size).to(DEVICE)  # N
        # use last output as input, teacher_forcing_ratio == 0
        y_hat_t_embedding = self.embedding(y_hat_t_label) # N, Embedding
        rnn_input = torch.cat(
            (y_hat_t_embedding, attention_context), dim=1)

        y_hat = []
        y_hat_label = []
        attentions = []
        finished_instances = []
        # iterate through max possible label length
        for time_index in range(max_label_len):
            # decoding rnn
            # decide whether to use teacher forcing in this time step

            # rnn
            hidden_states[0], memory_states[0] = self.rnns[0](
                rnn_input, (hidden_states[0], memory_states[0])
            )

            for hidden_layer in range(1, self.n_layers):
                hidden_states[hidden_layer], memory_states[hidden_layer] = self.rnns[hidden_layer](
                    hidden_states[hidden_layer - 1], (hidden_states[hidden_layer], memory_states[hidden_layer])
                )
            rnn_output = hidden_states[-1]  # N, hidden_size

            query = self.fc(rnn_output.unsqueeze(1)) # N, query_len, key_value_size

            # apply attention to generate context
            masked_softmax_energy = self.attention(dup_key, query, key_mask) # N, key_len, query_len

            # N, key_len, query_len * N, key_len, key_size
            attention_context = torch.bmm(masked_softmax_energy.permute(0,2,1), dup_value) # N, 1, key_len * N, key_len, key_size => N, 1, key_size
            attention_context = attention_context.squeeze(1) # N, key_len

            y_hat_t = self.mlp(torch.cat((query.squeeze(1), attention_context), dim=1)) # N, vocab_size
    #             y_hat_t = self.fc2(torch.cat((query.squeeze(1), attention_context), dim=1)) # N, vocab_size

            attentions.append(masked_softmax_energy.detach().cpu())

            # add gumbel noise
            y_hat_t_gumbel = gumble_softmax(y_hat_t, temperature=1.0, eps=1e-10) # L, N, Vocabsize
            y_hat.append(y_hat_t_gumbel.detach()) # N, vocab_size
            y_hat_t_label = torch.argmax(y_hat_t_gumbel, dim=1).long() # N ; long tensor for embedding inputs: greedy output

            y_hat_label.append(y_hat_t_label.detach().cpu())
            # determine whether to stop
            if EOS_token in y_hat_t_label: # if some one ended
                finished_instances.extend((y_hat_t_label == EOS_token).nonzero().squeeze(1).detach().cpu().numpy().tolist())
                finished_instances = list(set(finished_instances))
                if len(finished_instances) == batch_size: # inferencing of all instances meet the end token, stop
                    break

            # prepare input of next time step
            # use last output as input, teacher_forcing_ratio == 0
            y_hat_t_embedding = self.embedding(y_hat_t_label.detach()) # N, Embedding
            rnn_input = torch.cat(
                (y_hat_t_embedding, attention_context), dim=1)


        # concat predictions
        y_hat = torch.stack(y_hat, dim=1) # N, label_L, vocab_size
        attentions = torch.cat(attentions, dim=2) # N, key_len, query_len
        y_hat_label = torch.stack(y_hat_label, dim=1) # N, label_L


        # TODO: currently lazily return none for attentions for memory consideration
        return y_hat, y_hat_label, attentions

    def init_states(self, source_outputs, hidden_size=None):
        """initiate state using source outputs and hidden size"""
        batch_size = source_outputs.size(0)
        hidden_size = self.hidden_size if hidden_size is None else hidden_size
        initiation = source_outputs.new_zeros(batch_size, hidden_size)
        return initiation
    
    def duplicate(self, tensor, num):
        '''duplicate data within tensor by num time [[0],[1]] -num=2-> [[0],[0],[1],[1]]'''
        new_tensor = []
        for i in range(tensor.size(0)):
            new_tensor.extend([tensor[i]]*num)
        return torch.stack(new_tensor, dim=0)
