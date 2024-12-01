import torch.nn as nn
import json
import torch

import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FirePredictor(nn.Module):
    def __init__(self,  event_vec_dim, seq_len, hidden_dim, num_layers):
        super(FirePredictor, self).__init__()
        self.event_vec_dim = event_vec_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len

        self.lstm = nn.LSTM(
            self.event_vec_dim,
            self.hidden_dim,
            num_layers = self.num_layers,
            dropout=0.2,
            batch_first=True,
            device=device
        )
        self.flatten = nn.Flatten()
        self.output = nn.Linear(hidden_dim*seq_len, event_vec_dim, device=device)

    def forward(self, x):
        h1 = self.lstm(x)[0]
        flat = self.flatten(h1)
        return self.output(flat).reshape(x.shape[0], self.event_vec_dim)

    def to_json(self, path):
        dir = path.split('/')[0:-1]
        dir = '/'.join(dir)
        os.makedirs(dir, exist_ok=True)

        listified_dict = {}
        for key,val in self.state_dict().items():
            listified_dict[key] = val.tolist()

        with open(path, mode='w') as file:
            json.dump(listified_dict, file)


    def from_json(self, path):
        # read the JSON
        with open(path) as file:
            text = file.read()
            obj = json.loads(text)

            # turn the JSON into a dict of tensors
            tensorified = {}
            for key,val in obj.items():
                tensorified[key] = torch.Tensor(val)

            # set the models state_dict to the dict of tensors
            self.load_state_dict(tensorified)

