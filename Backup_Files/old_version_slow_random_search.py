    def inference_random_search(self, argument_list):
        """ return y_hat adding gumbel noises, y_hat_labels using gumbel noises"""
        key, value, final_seq_lens, seq_order, reversed_seq_order, SOS_token, EOS_token, (max_label_len, num_candidates), SEARCH_MODE = argument_list
        assert SEARCH_MODE == 'random'

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

        gumble_softmax = Gumbel_Softmax()

        candidate_labels = []
        candidate_y_hats = []

        # ------ generate number of output with gumbel noise ------
        for candidate_index in range(num_candidates):

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

            # store results to pool
            candidate_labels.append(y_hat_label) # NumCandidate, N, Prediction_L # Prediction_L varies for different instances
            candidate_y_hats.append(y_hat) # # NumCandidate, N, Prediction_L, vocab_size
            # not storing attention to save memory

        # TODO: currently lazily return none for attentions for memory consideration
        return candidate_y_hats, candidate_labels, None