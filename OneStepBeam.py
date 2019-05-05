import torch
import char_language_model


class OneStepBeam:
    def __init__(self, model, SOS_token, EOS_token, max_label_len, beam_size, num_candidates):
        self.model = model
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token
        self.max_label_len = max_label_len
        self.beam_size = beam_size
        self.num_candidates = num_candidates
        self.model.eval()

    def do_one_step(self, argument_tuple):
        '''do one step beam search and return results'''
        key, value, seq_len = argument_tuple
        list_beam_candidates = model.beam_search_decode_inference(
            key, value, seq_len, SOS_token, EOS_token, max_label_len, beam_size, num_candidates)
        winner_beam_node = list_beam_candidates[0]

        # label_L, vocab_size
        y_hat = torch.cat(winner_beam_node['y_hats'][1:], dim=0)
        y_hat_label = torch.stack(
            winner_beam_node['y_hat_t_labels'][1:]).detach().cpu().numpy()  # label_L
        # key_len, query_len
        attention = torch.cat(
            winner_beam_node['attentions'][1:], dim=-1).squeeze(0)

        # decode to string
        if char_language_model.EOS_token in y_hat_label:
            end_position = y_hat_label.tolist().index(char_language_model.EOS_token)
            if end_position < len(y_hat_label):  # not the last one
                pred = y_hat_label[:end_position+1]
            else:
                pred = y_hat_label
        else:
            pred = y_hat_label  # if no EOS, use the entire prediction

        return pred
