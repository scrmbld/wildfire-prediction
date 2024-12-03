import torch.nn as nn
import json
import torch

import os

# use a gpu if we can
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FirePredictor(nn.Module):
    def __init__(self,  event_vec_dim, seq_len, hidden_dim, num_layers):
        super(FirePredictor, self).__init__()

        # store some information about the sizes of various things
        # not all of this is actually used but it doesn't hurt to hold onto it
        self.event_vec_dim = event_vec_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len

        # define some LSTM layers
        self.lstm = nn.LSTM(
            self.event_vec_dim,
            self.hidden_dim,
            num_layers = self.num_layers,
            dropout=0.2,
            batch_first=True, # if this is False (the default), the first dimension will be the sequnce, not the batch
            device=device
        )
        self.flatten = nn.Flatten()
        self.output = nn.Linear(hidden_dim*seq_len, event_vec_dim, device=device)

    def forward(self, x):
        h1 = self.lstm(x)[0] # only use the actual outputs, not the "outputs" for the next recurrent layer
        flat = self.flatten(h1)
        return self.output(flat).reshape(x.shape[0], self.event_vec_dim)

    def to_json(self, path):
        """
        Write the model's state_dict to a JSON file.
        """
        dir = path.split('/')[0:-1]
        dir = '/'.join(dir)
        os.makedirs(dir, exist_ok=True)

        listified_dict = {}
        for key,val in self.state_dict().items():
            listified_dict[key] = val.tolist()

        with open(path, mode='w') as file:
            json.dump(listified_dict, file)


    def from_json(self, path):
        """
        Set the model's state_dict by reading a JSON file. Everything (namely weight tensors) must be the same size for this to work.
        """
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

