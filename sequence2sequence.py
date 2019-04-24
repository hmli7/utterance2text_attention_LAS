import torch
import torch.nn as nn
import torch.nn.functional as F
import sequence2sequence_Atten_modules
import char_language_model


class Sequence2Sequence(nn.Module):
    def __init__(self, embed_size, hidden_size, n_plstm, mlp_hidden_size, mlp_output_size,
                 decoder_vocab_size, decoder_embed_size, decoder_key_value_size, decoder_hidden_size, decoder_n_layers, decoder_mlp_hidden_size):
        super(Sequence2Sequence, self).__init__()
        self.encoder = sequence2sequence_Atten_modules.Encoder_RNN(
            embed_size, hidden_size, n_plstm, mlp_hidden_size, mlp_output_size)
        self.decoder = sequence2sequence_Atten_modules.Decoder_RNN(
            decoder_vocab_size, decoder_embed_size, decoder_key_value_size, decoder_hidden_size, decoder_n_layers, decoder_mlp_hidden_size)

    def forward(self, seq_batch):
        keys, values, labels, final_seq_lens, sequence_order, reverse_sequence_order = self.encoder(
            seq_batch)
        y_hat = self.decoder(keys, values, labels, final_seq_lens, sequence_order,
                             reverse_sequence_order, char_language_model.SOS_token, TEACHER_FORCING_Ratio=0, TEST=False)
        return y_hat
