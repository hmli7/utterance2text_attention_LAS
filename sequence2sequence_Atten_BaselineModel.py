import torch
import torch.nn as nn

class Encoder_RNN(nn.Module):
    def __init__(self, embed_size, hidden_size, n_layers, n_plstm, mlp_layers):
        return super(Encoder_RNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        

        self.rnns = nn.ModuleList()
        self.rnns.append(nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=nlayers, bidirectional=True))
        for i in range(n_plstm):
            # add plstm layers
            self.rnns.append(pBLSTM(hidden_size*2, hidden_size))
        self.mlp1 = MLP(hidden_size*2, mlp_layers[0], mlp_layers[1])
        self.mlp2 = MLP(hidden_size*2, mlp_layers[0], mlp_layers[1])

    def forward(self, x):
        for rnn in self.rnns:
            x, _ = rnn(x)
        keys = self.mlp1(x)
        values = self.mlp2(x)
        return keys, values
        
class pBLSTM(nn.Module):
    '''pyramid structure with a factor of 0.5 (L, B, F) -> (L/2, B, F*2)'''
    def __init__(self, input_dim, hidden_size):
        super(pBLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim*2, hidden_size=hidden_size, num_layers=1, bidirectional=True)

    def forward(self, x):

        # reduce the timestep by 2
        x = x.permute(1, 0, 2) # L, B, F -> B, L, F
        batch_size, seq_length, feature_dim = x.size()

        x = x.contiguous().view(batch_size, int(seq_length//2), feature_dim*2) # B, L/2, F*2
        return self.lstm(x.permute(1, 0, 2))

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