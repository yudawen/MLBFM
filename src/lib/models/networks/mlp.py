import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,predict=False):
        super(MLP, self).__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.predict = predict

    def forward(self, x):
        B, N, D = x.size()
        x = x.reshape(B*N, D)
        for i, layer in enumerate(self.layers):
            if i < self.num_layers - 1:
                x = F.relu(layer(x))
            elif  self.predict==False:
                x = F.relu(layer(x))
            elif i==self.num_layers - 1 and self.predict==True:
                x = layer(x)
            elif  i==self.num_layers - 1:
                x = F.relu(layer(x))
        x = x.view(B, N, self.output_dim)
        return x
