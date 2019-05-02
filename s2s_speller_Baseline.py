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


class Decoder_RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, key_value_size, hidden_size, n_layers, mlp_hidden_size, padding_value):
        super(Decoder_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.key_value_size = key_value_size # key and value pair hidden size
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.padding_value = padding_value


        # for transcripts
        self.embedding = nn.Embedding(
            self.vocab_size, self.embed_size) # , padding_idx=self.padding_value  using index of <EOS> to pad

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
            return self.forward_beam_search(argument_list[:-1])
        else:
            NotImplemented
        

    def forward_greedy_search(self, argument_list):
        """only used to train other than inferencing"""
        key, value, labels, final_seq_lens, SOS_token,_, TEACHER_FORCING_Ratio = argument_list
        # labels have been sorted by seq_order
        # label lens for loss masking
        labels_lens = [len(label) for label in labels]
        # pad
        # sorted N, Label L; use -1 to pad, this will be initialized to zero in embedding
        labels_padded = pad_sequence(labels, padding_value=self.padding_value)
        labels_padded.requires_grad = False
        # embedding
        labels_embedding = self.embedding(labels_padded)  # label L, N, emb

        # ------ Init ------
        # initialize attention, hidden states, memory states
        attention_context = self.init_states(key, hidden_size=key.size(2))  # N, Key/query_len
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
#         key_mask = torch.stack([torch.arange(0, max_len)
#                                 for i in range(batch_size)]).int()
#         key_lens = torch.stack(
#             [torch.full((1, max_len), length).squeeze(0) for length in final_seq_lens]).int()
#         key_mask = key_mask < key_lens  # a matrix of 1 & 0; N, max_key_len
#         key_mask = key_mask.unsqueeze(2).repeat(
#             (1, 1, max_label_len)).float().to(DEVICE)  # expend to N, max_key_len, max_label_len; float for matrix mul when applying attention
        key_mask = torch.arange(0, max_len) < final_seq_lens.view(len(final_seq_lens),-1)
        key_mask = key_mask.unsqueeze(2).float().to(DEVICE) # N, max_key_len, 1
        key_mask.requires_grad = False

        # ------ LAS ------
        # initialze for the first time step input
        y_hat_t_label = torch.LongTensor([SOS_token]*batch_size).to(DEVICE)  # N
        y_hat = []
        y_hat_label = []
        attentions = []
        
        # initialize for the first time step
        y_hat_t_embedding = self.embedding(y_hat_t_label) # N, Embedding
        rnn_input = torch.cat((y_hat_t_embedding, attention_context), dim=1)
        
        # iterate through max possible label length
        for time_index in range(max_label_len):
            # decoding rnn

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
            masked_softmax_energy = self.attention(key, query, key_mask) # N, key_len, query_len

            # N, key_len, query_len * N, key_len, key_size
            attention_context = torch.bmm(masked_softmax_energy.permute(0,2,1), value) # N, 1, key_len * N, key_len, key_size => N, 1, key_size
            attention_context = attention_context.squeeze(1) # N, key_len

            y_hat_t = self.mlp(torch.cat((query.squeeze(1), attention_context), dim=1)) # N, vocab_size
#             y_hat_t = self.fc2(torch.cat((query.squeeze(1), attention_context), dim=1)) # N, vocab_size
            y_hat.append(y_hat_t) # N, vocab_size
            attentions.append(masked_softmax_energy.detach().cpu())
            
            y_hat_t_label = torch.argmax(y_hat_t, dim=1).long() # N ; long tensor for embedding inputs
            y_hat_label.append(y_hat_t_label.detach().cpu())
            
            # decide whether to use teacher forcing in this time step
            use_teacher_forcing = True if random.random() < TEACHER_FORCING_Ratio else False
            if not use_teacher_forcing:
                y_hat_t_embedding = self.embedding(y_hat_t_label) # N, Embedding
                rnn_input = torch.cat(
                    (y_hat_t_embedding, attention_context), dim=1)
            else:
                rnn_input = torch.cat(
                    (labels_embedding[time_index, :, :], attention_context), dim=1)
            
        # concat predictions
        y_hat = torch.stack(y_hat, dim=1) # N, label_L, vocab_size
        attentions = torch.cat(attentions, dim=2) # N, key_len, query_len
        y_hat_label = torch.stack(y_hat_label, dim=1) # N, label_L
        
        return y_hat, y_hat_label, labels_padded.permute(1,0), labels_lens, attentions
    
    def beam_search_decode(self, utterance_key, utterance_value, utterance_label, utterance_seq_len, SOS_token, EOS_token, TEACHER_FORCING_Ratio, MAX_SEQ_LEN, beam_size, num_candidates):
        '''do beam search decoding for training for one utterance a time
        source: https://github.com/kaituoxu/Listen-Attend-Spell/blob/master/src/models/decoder.py'''
        utterance_key = utterance_key.unsqueeze(0) # 1, key_len, key_hidden_size
        utterance_value = utterance_value.unsqueeze(0)
        utterance_label = utterance_label.unsqueeze(0) # 1, label_len

        # embedding
        labels_embedding = self.embedding(utterance_label)  # 1, label L, emb

        # ------ Init ------
        # initialize attention, hidden states, memory states
        attention_context = self.init_states(
            utterance_key, hidden_size=utterance_key.size(2))  # 1, Key/query_len
        # initialize hidden and memory states
        hidden_states = [self.init_states(utterance_key)]  # 1, hidden_size
        memory_states = [self.init_states(utterance_key)]  # 1, hidden_size
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]
        for i in range(self.n_layers-1):
            # we use single direction lstm cell in this case
            hidden_states.append(self.init_states(utterance_key))
            # [n_layers, N, Hidden_size]
            memory_states.append(self.init_states(utterance_key))

        max_label_len = utterance_label.size(1)

        # query_mask, (0, max_length)
        max_key_len = utterance_key.size(1)

        key_mask = torch.arange(0, max_key_len).view(1,-1) < utterance_seq_len
        key_mask = key_mask.unsqueeze(2).float().to(DEVICE)  # 1, max_key_len, 1
        key_mask.requires_grad = False

        # ------ LAS ------
        # initialze for the first time step input
        y_hat_t_label = torch.tensor(SOS_token).long().to(DEVICE)  # N

        start_beam_node = {
            'y_hat_t_labels':[y_hat_t_label],
            'y_hats':[],
            'attentions':[attention_context],
            'previous_hidden':hidden_states,
            'previous_memory':memory_states,
            'previous_attention_c':attention_context,    
            'prob':0.0
        }
        beam_nodes = [start_beam_node]
        stopped_beam_nodes = []

        # iterate through max possible label length
        for time_index in range(max_label_len):
            # best buffer
            beam_nodes_candidates = []

            for beam_node in beam_nodes:
                # for each beam node in the buffer
                # decide whether to use teacher forcing in this time step
                use_teacher_forcing = True if random.random() < TEACHER_FORCING_Ratio else False
                if not use_teacher_forcing:
                    y_hat_t_embedding = self.embedding(
                        beam_node['y_hat_t_labels'][-1]).unsqueeze(0)  # N, Embedding
                    rnn_input = torch.cat(
                        (y_hat_t_embedding, attention_context), dim=1)
                else:
                    rnn_input = torch.cat(
                        (labels_embedding[:, time_index, :], attention_context), dim=1)


                # rnn
                hidden_states[0], memory_states[0] = self.rnns[0](
                    rnn_input, (beam_node['previous_hidden']
                                [0], beam_node['previous_memory'][0])
                )

                for hidden_layer in range(1, self.n_layers):
                    hidden_states[hidden_layer], memory_states[hidden_layer] = self.rnns[hidden_layer](
                        hidden_states[hidden_layer - 1], (beam_node['previous_hidden']
                                                          [hidden_layer], beam_node['previous_memory'][hidden_layer])
                    )
                rnn_output = hidden_states[-1]  # 1, hidden_size

                # 1, query_len, key_value_size
                utterance_query = self.fc(rnn_output.unsqueeze(1))

                # apply attention to generate context
                masked_softmax_energy = self.attention(
                    utterance_key, utterance_query, key_mask)  # 1, key_len, query_len

                # 1, key_len, query_len * 1, key_len, key_size
                # 1, 1, key_len * 1, key_len, key_size => 1, 1, key_size
                attention_context = torch.bmm(
                    masked_softmax_energy.permute(0, 2, 1), utterance_value)
                attention_context = attention_context.squeeze(1)  # 1, key_len

                y_hat_t = self.mlp(
                    torch.cat((utterance_query.squeeze(1), attention_context), dim=1))  # 1, vocab_size

                # get top n candidate
                logits = F.log_softmax(y_hat_t, dim=1)
                candidate_probs, candidate_pred_labels = torch.topk(logits, beam_size, dim=1)

                # push to buffer
                for candidate_index in range(beam_size):
                    new_beam_node = {
                        'y_hat_t_labels': beam_node['y_hat_t_labels'] + [candidate_pred_labels[0][candidate_index]],
                        'y_hats': beam_node['y_hats']+[y_hat_t] if len(beam_node['y_hats'])!=0 else [torch.zeros_like(y_hat_t), y_hat_t],
                        'attentions': beam_node['attentions']+[masked_softmax_energy.detach().cpu()],
                        'previous_hidden': hidden_states[:],
                        'previous_memory': memory_states[:],
                        'previous_attention_c': attention_context[:],
                        'prob': beam_node['prob'] + candidate_probs[0][candidate_index]
                    }
                    beam_nodes_candidates.append(new_beam_node)
                
                # rank by prob
                beam_nodes_candidates = sorted(beam_nodes_candidates, key=lambda x: x['prob'], reverse=True)[:beam_size]

            beam_nodes = beam_nodes_candidates
            if time_index == max_label_len-1:
                for beam_node in beam_nodes:
                    beam_node['y_hat_t_labels'].append(torch.tensor(EOS_token).to(DEVICE))
            
            left_beam_nodes = []
            for beam_node in beam_nodes:
                if beam_node['y_hat_t_labels'][-1] == EOS_token:
                    stopped_beam_nodes.append(beam_node)
                else:
                    left_beam_nodes.append(beam_node)
            beam_nodes = left_beam_nodes

            if len(beam_nodes) == 0:
                # stop decoding
                break
        # need to detach y_hat_label in each beam node to free memory
        return sorted(stopped_beam_nodes, key=lambda x: x['prob'], reverse=True)[:min(len(stopped_beam_nodes), num_candidates)]

    def forward_beam_search(self, argument_list):
        """only used to train other than inferencing"""
        keys, values, labels, final_seq_lens, SOS_token, EOS_token, TEACHER_FORCING_Ratio, (MAX_SEQ_LEN, beam_size, num_candidates) = argument_list
        batch_size = len(keys)
        
        # labels have been sorted by seq_order
        # label lens for loss masking
        labels_lens = [len(label) for label in labels]
        # pad
        # sorted N, Label L; use -1 to pad, this will be initialized to zero in embedding
        labels_padded = pad_sequence(labels, padding_value=self.padding_value).permute(1,0)
        labels_padded.requires_grad = False

        # concat predictions
        y_hat = []  # N, label_L, vocab_size
        attentions = []  # N, key_len, query_len
        y_hat_label = []  # N, label_L

        for utterance_index in range(batch_size):
            list_beam_candidates = self.beam_search_decode(keys[utterance_index], values[utterance_index], labels_padded[utterance_index], final_seq_lens[utterance_index], SOS_token, EOS_token, TEACHER_FORCING_Ratio, MAX_SEQ_LEN, beam_size, num_candidates)
            winner_beam_node = list_beam_candidates[0]
           
            y_hat.append(torch.cat(
                winner_beam_node['y_hats'][1:], dim=0)) # label_L, vocab_size
            y_hat_label.append(torch.stack(
                winner_beam_node['y_hat_t_labels'][1:]).detach().cpu().numpy())  # label_L
            attentions.append(torch.cat(
                winner_beam_node['attentions'][1:], dim=-1).squeeze(0))  # key_len, query_len
#         pdb.set_trace()
#         y_hat = torch.stack(y_hat, dim=0)
#         y_hat_label = torch.stack(y_hat_label, dim=0)
#         attentions = torch.stack(attentions, dim=0)
#         pdb.set_trace()
        return y_hat, y_hat_label, labels_padded, labels_lens, attentions

    
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

        # ------ Init ------
        # initialize attention, hidden states, memory states
        attention_context = self.init_states(key, hidden_size=key.size(2))  # N, Key/query_len
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
        batch_size, max_len = key.size(0), key.size(1)
#         key_mask = torch.stack([torch.arange(0, max_len)
#                                 for i in range(batch_size)]).int()
#         key_lens = torch.stack(
#             [torch.full((1, max_len), length).squeeze(0) for length in final_seq_lens]).int()
#         key_mask = key_mask < key_lens  # a matrix of 1 & 0; N, max_key_len
#         key_mask = key_mask.unsqueeze(2).repeat(
#             (1, 1, max_label_len)).float().to(DEVICE)  # expend to N, max_key_len, max_label_len; float for matrix mul when applying attention
        key_mask = torch.arange(0, max_len) < final_seq_lens.view(len(final_seq_lens),-1)
        key_mask = key_mask.unsqueeze(2).float().to(DEVICE) # N, max_key_len, 1
        key_mask.requires_grad = False

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
            masked_softmax_energy = self.attention(key, query, key_mask) # N, key_len, query_len

            # N, key_len, query_len * N, key_len, key_size
            attention_context = torch.bmm(masked_softmax_energy.permute(0,2,1), value) # N, 1, key_len * N, key_len, key_size => N, 1, key_size
            attention_context = attention_context.squeeze(1) # N, key_len

            y_hat_t = self.mlp(torch.cat((query.squeeze(1), attention_context), dim=1)) # N, vocab_size
#             y_hat_t = self.fc2(torch.cat((query.squeeze(1), attention_context), dim=1)) # N, vocab_size
            y_hat.append(y_hat_t) # N, vocab_size
            attentions.append(masked_softmax_energy.detach().cpu())
            if SEARCH_MODE == 'greedy':
                y_hat_t_label = torch.argmax(y_hat_t, dim=1).long() # N ; long tensor for embedding inputs: greedy output
            elif SEARCH_MODE == 'beam_search':
                NotImplemented
            else:
                NotImplemented
            y_hat_label.append(y_hat_t_label.detach().cpu())
            
            # determine whether to stop
            if EOS_token in y_hat_t_label: # if some one ended
                finished_instances.extend((y_hat_t_label == EOS_token).nonzero().squeeze(1).detach().cpu().numpy().tolist())
                finished_instances = list(set(finished_instances))
                if len(finished_instances) == batch_size: # inferencing of all instances meet the end token, stop
                    break
            
            # prepare input of next time step
            # use last output as input, teacher_forcing_ratio == 0
            y_hat_t_embedding = self.embedding(y_hat_t_label) # N, Embedding
            rnn_input = torch.cat(
                (y_hat_t_embedding, attention_context), dim=1)


        # concat predictions
        y_hat = torch.stack(y_hat, dim=1) # N, label_L, vocab_size
        attentions = torch.cat(attentions, dim=2) # N, key_len, query_len
        y_hat_label = torch.stack(y_hat_label, dim=1) # N, label_L
        
        return y_hat, y_hat_label, attentions
    
#     def inference_random_search(self, argument_list):
#         """ return y_hat adding gumbel noises, y_hat_labels using gumbel noises"""
#         key, value, final_seq_lens, seq_order, reversed_seq_order, SOS_token, EOS_token, (max_label_len, num_candidates), SEARCH_MODE = argument_list
#         assert SEARCH_MODE == 'random'

#         # ------ Init ------
#         # initialize attention, hidden states, memory states
#         attention_context = self.init_states(key, hidden_size=key.size(2))  # N, Key/query_len
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
#         batch_size, max_len = key.size(0), key.size(1)
#     #         key_mask = torch.stack([torch.arange(0, max_len)
#     #                                 for i in range(batch_size)]).int()
#     #         key_lens = torch.stack(
#     #             [torch.full((1, max_len), length).squeeze(0) for length in final_seq_lens]).int()
#     #         key_mask = key_mask < key_lens  # a matrix of 1 & 0; N, max_key_len
#     #         key_mask = key_mask.unsqueeze(2).repeat(
#     #             (1, 1, max_label_len)).float().to(DEVICE)  # expend to N, max_key_len, max_label_len; float for matrix mul when applying attention
#         key_mask = torch.arange(0, max_len) < final_seq_lens.view(len(final_seq_lens),-1)
#         key_mask = key_mask.unsqueeze(2).float().to(DEVICE) # N, max_key_len, 1
#         key_mask.requires_grad = False

#         gumble_softmax = Gumbel_Softmax()

#         candidate_labels = []
#         candidate_y_hats = []

#         # ------ generate number of output with gumbel noise ------
#         for candidate_index in range(num_candidates):

#             # ------ LAS ------
#             # initialze for the first time step input
#             y_hat_t_label = torch.LongTensor([SOS_token]*batch_size).to(DEVICE)  # N
#             # use last output as input, teacher_forcing_ratio == 0
#             y_hat_t_embedding = self.embedding(y_hat_t_label) # N, Embedding
#             rnn_input = torch.cat(
#                 (y_hat_t_embedding, attention_context), dim=1)

#             y_hat = []
#             y_hat_label = []
#             attentions = []
#             finished_instances = []
#             # iterate through max possible label length
#             for time_index in range(max_label_len):
#                 # decoding rnn
#                 # decide whether to use teacher forcing in this time step

#                 # rnn
#                 hidden_states[0], memory_states[0] = self.rnns[0](
#                     rnn_input, (hidden_states[0], memory_states[0])
#                 )

#                 for hidden_layer in range(1, self.n_layers):
#                     hidden_states[hidden_layer], memory_states[hidden_layer] = self.rnns[hidden_layer](
#                         hidden_states[hidden_layer - 1], (hidden_states[hidden_layer], memory_states[hidden_layer])
#                     )
#                 rnn_output = hidden_states[-1]  # N, hidden_size

#                 query = self.fc(rnn_output.unsqueeze(1)) # N, query_len, key_value_size

#                 # apply attention to generate context
#                 masked_softmax_energy = self.attention(key, query, key_mask) # N, key_len, query_len

#                 # N, key_len, query_len * N, key_len, key_size
#                 attention_context = torch.bmm(masked_softmax_energy.permute(0,2,1), value) # N, 1, key_len * N, key_len, key_size => N, 1, key_size
#                 attention_context = attention_context.squeeze(1) # N, key_len

#                 y_hat_t = self.mlp(torch.cat((query.squeeze(1), attention_context), dim=1)) # N, vocab_size
#         #             y_hat_t = self.fc2(torch.cat((query.squeeze(1), attention_context), dim=1)) # N, vocab_size
                
#                 attentions.append(masked_softmax_energy.detach().cpu())

#                 # add gumbel noise
#                 y_hat_t_gumbel = gumble_softmax(y_hat_t, temperature=1.0, eps=1e-10) # L, N, Vocabsize
#                 y_hat.append(y_hat_t_gumbel.detach()) # N, vocab_size
#                 y_hat_t_label = torch.argmax(y_hat_t_gumbel, dim=1).long() # N ; long tensor for embedding inputs: greedy output

#                 y_hat_label.append(y_hat_t_label.detach().cpu())
#                 # determine whether to stop
#                 if EOS_token in y_hat_t_label: # if some one ended
#                     finished_instances.extend((y_hat_t_label == EOS_token).nonzero().squeeze(1).detach().cpu().numpy().tolist())
#                     finished_instances = list(set(finished_instances))
#                     if len(finished_instances) == batch_size: # inferencing of all instances meet the end token, stop
#                         break

#                 # prepare input of next time step
#                 # use last output as input, teacher_forcing_ratio == 0
#                 y_hat_t_embedding = self.embedding(y_hat_t_label.detach()) # N, Embedding
#                 rnn_input = torch.cat(
#                     (y_hat_t_embedding, attention_context), dim=1)


#             # concat predictions
#             y_hat = torch.stack(y_hat, dim=1) # N, label_L, vocab_size
#             attentions = torch.cat(attentions, dim=2) # N, key_len, query_len
#             y_hat_label = torch.stack(y_hat_label, dim=1) # N, label_L

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