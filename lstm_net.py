import torch.nn as nn
from connectivity_lstm import *
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from scipy import stats
from numpy import linalg

class Net_lstm(torch.nn.Module):
    # PyTorch module for implementing an LSTM to be trained on cognitive tasks.
    def __init__(self, n, alpha=0.2, sigma_rec=0.15, input_size=4, output_size=2, dale=True,
                 activation=torch.nn.ReLU(), dropout_prob=0.0, use_layer_norm=False):
        super(Net_lstm, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.sigma_rec = torch.tensor(sigma_rec)
        self.n = n
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.dale = dale
        self.device = 'gpu'
        self.use_dropout = dropout_prob > 0
        self.use_layer_norm = use_layer_norm
        
        if self.use_dropout:
            self.lstm_dropout = nn.Dropout(p=dropout_prob)
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(n)

        if torch.cuda.is_available():
            self.device = 'cuda'
        
        if dale:
            n_exc = int(self.n * .85)
            n_inh = self.n - n_exc

            dale_mask = torch.ones(self.n)
            dale_mask[n_exc:] = -1
            self.dale_mask = dale_mask.to(self.device)
            
            # Increase the magnitude of inhibitory weights
            w_scale = 1.0  # Scale factor for weight initialization           
    
        # Initialize LSTM
        self.lstm = nn.LSTM(input_size, self.n, batch_first=True)

        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.orthogonal_(self.lstm.weight_ih_l0)

        nn.init.zeros_(self.lstm.bias_ih_l0)
        nn.init.zeros_(self.lstm.bias_hh_l0)

        self.output_layer = nn.Linear(self.n, self.output_size, bias=False)
        nn.init.xavier_normal_(self.output_layer.weight, gain=0.1)

        # Apply Dale's law and balance network
        if dale:
            self.lstm.weight_hh_l0.data, self.lstm.weight_ih_l0.data, self.output_layer.weight.data, \
            self.dale_mask_hh, self.dale_mask_ih, self.output_mask, self.input_mask = \
                init_connectivity_lstm(self.n, self.input_size, self.output_size, radius=1.5)

        # Apply connectivity constraints
        self.connectivity_constraints()

    def to(self, device):
        """Override the to() method to ensure masks are moved to the correct device"""
        super().to(device)
        if self.dale:
            self.dale_mask = self.dale_mask.to(device)
            self.dale_mask_hh = self.dale_mask_hh.to(device)
            self.dale_mask_ih = self.dale_mask_ih.to(device)
            self.output_mask = self.output_mask.to(device)
            self.input_mask = self.input_mask.to(device)
        return self
        
    # Dynamics
    def forward(self, u, hidden = None):
        # Get trial length
        t = u.shape[1]

        # Initialize hidden states to zero.
        
        h0 = torch.zeros(1, u.size(0), self.n).to(self.device)
        c0 = torch.zeros(1, u.size(0), self.n).to(self.device)
        hidden = (h0, c0)

        # Noise to be applied at each time step.
        noise = torch.sqrt(2 * self.alpha * self.sigma_rec ** 2) * torch.empty(u.size(0), t, self.input_size).normal_(mean=0, std=1).to(self.device)
        

        # Forward propagate LSTM
        out, hidden = self.lstm(u + noise, hidden)

        if self.use_layer_norm:
            out = self.layer_norm(out)
        if self.use_dropout:
            out = self.lstm_dropout(out)
        
        # Apply output layer
        out = self.output_layer(out)
        return out, hidden

    def connectivity_constraints(self):
        # Constrain input and output to be positive
        self.output_layer.weight.data = torch.relu(self.output_layer.weight.data)

        # Constrain network to satisfy Dale's law
        if self.dale:
            self.output_layer.weight.data = self.output_mask * torch.relu(self.output_layer.weight.data)
            self.lstm.weight_hh_l0.data = torch.relu(self.lstm.weight_hh_l0.data * self.dale_mask_hh) * self.dale_mask_hh
            self.lstm.weight_ih_l0.data = torch.relu(self.lstm.weight_ih_l0.data * self.dale_mask_ih) * self.dale_mask_ih

    def l2_ortho(self):
        # Penalty to enforce orthogonality of input and output columns.

        input_weights = self.lstm.weight_ih_l0[:self.n, :]
        b = torch.cat((input_weights, self.output_layer.weight.t()), dim=1)
        b = b / torch.norm(b, dim=0)
        return torch.norm(b.t() @ b - torch.diag(torch.diag(b.t() @ b)), p=2)

    def loss_function(self, x, z, mask):
        return self.mse_z(x, z, mask) + self.l2_ortho() + 0.05 * torch.mean(x ** 2)

    def mse_z(self, x, z, mask):
        # Mean squared error for task performance.
        mse = nn.MSELoss()
        return mse(x * mask, z * mask)

    # Function for fitting LSTM to task
    def fit(self, u, z, mask, epochs=10000, lr=0.01, verbose=False, weight_decay=0, maintain_hidden=False):
        # Wrap training data as PyTorch dataset.
        my_dataset = TensorDataset(u, z, mask)
        my_dataloader = DataLoader(my_dataset, batch_size=128)
        # Initialize optimizer.
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        # Initialize hidden to zero between batches, maintain hidden should be true for context switch analysis
        hidden = None if not maintain_hidden else (
            torch.zeros(1, u.size(0), self.n).to(self.device),
            torch.zeros(1, u.size(0), self.n).to(self.device)
        )
        
    # Training loop
        epoch = 0
        while epoch < epochs:
            for batch_idx, (u_batch, z_batch, mask_batch) in enumerate(my_dataloader):
                optimizer.zero_grad()
                x_batch, hidden = self.forward(u_batch, hidden if maintain_hidden else None)
                loss = self.loss_function(x_batch, z_batch, mask_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                # Apply connectivity constraints after each gradient step.
                self.connectivity_constraints()

            epoch += 1
            if verbose:
                if epoch % 5 == 0:
                    x, _ = self.forward(u, hidden if maintain_hidden else None)
                    print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
                    print("mse_z: {:.4f}".format(self.mse_z(x, z, mask).item()))

""" Original training loop
Origingal training loop
# Training loop
        epoch = 0
        while epoch < epochs:
            for batch_idx, (u_batch, z_batch, mask_batch) in enumerate(my_dataloader):
                optimizer.zero_grad()
                x_batch = self.forward(u_batch)
                loss = self.loss_function(x_batch, z_batch, mask_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                # Apply connectivity constraints after each gradient step.
                self.connectivity_constraints()

            epoch += 1
            if verbose:
                if epoch % 5 == 0:
                    x = self.forward(u)
                    print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
                    print("mse_z: {:.4f}".format(self.mse_z(x, z, mask).item()))
"""