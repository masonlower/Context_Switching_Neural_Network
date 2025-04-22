
import torch
import numpy as np
import torch
from scipy.sparse import random
from scipy import stats
from numpy import linalg



def init_connectivity( N,input_size,output_size,radius=1.5):
        
        '''
        Initialize connectivity of RNN
        :param N: network size
        :param input_size: number of input channels
        :param output_size: number of output channels
        :param radius: spectral radius
        :return: Connectivity and masks
        '''
        Ne = int(N * 0.8)
        Ni = int(N * 0.2)

        # Initialize W_rec
        W_rec = torch.empty([0, N])

        # Balancing parameters  
        mu_E = 1 / np.sqrt(N)
        mu_I = 4 / np.sqrt(N)

        var = 1 / N

        rowE = torch.empty([Ne, 0])
        rowI = torch.empty([Ni, 0])

        rowE = torch.cat((rowE, torch.tensor(
            random(Ne, Ne, density=1, data_rvs=stats.norm(scale=var, loc=mu_E).rvs).toarray()).float()), 1)
        rowE = torch.cat((rowE, -torch.tensor(
            random(Ne, Ni, density=1, data_rvs=stats.norm(scale=var, loc=mu_I).rvs).toarray()).float()), 1)
        rowI = torch.cat((rowI, torch.tensor(
            random(Ni, Ne, density=1, data_rvs=stats.norm(scale=var, loc=mu_E).rvs).toarray()).float()), 1)
        rowI = torch.cat((rowI, -torch.tensor(
            random(Ni, Ni, density=1, data_rvs=stats.norm(scale=var, loc=mu_I).rvs).toarray()).float()), 1)

        W_rec = torch.cat((W_rec, rowE), 0)
        W_rec = torch.cat((W_rec, rowI), 0)

        W_rec = W_rec - torch.diag(torch.diag(W_rec))
        w, v = linalg.eig(W_rec)
        spec_radius = np.max(np.absolute(w))
        W_rec = radius * W_rec / spec_radius

        W_in = torch.zeros([N, input_size]).float()
        W_in[:, :] = radius * torch.tensor(
            random(N, input_size, density=1, data_rvs=stats.norm(scale=var, loc=mu_E).rvs).toarray()).float()

        W_out = torch.zeros([output_size, N])
        W_out[:, :Ne] = torch.tensor(random(output_size, Ne, density=1,
                                            data_rvs=stats.norm(scale=var, loc=mu_E).rvs).toarray()).float()

        dale_mask = torch.sign(W_rec).float()
        output_mask = (W_out != 0).float()
        input_mask = (W_in != 0).float()

        return W_rec.float(), W_in.float(), W_out.float(), dale_mask, output_mask, input_mask
        

def init_connectivity_lstm(N, input_size, output_size, radius=1.5):
    """
    Initialize connectivity for LSTM networks
    """
    # Get the basic connectivity matrices
    W_rec, W_in, W_out, dale_mask, output_mask, input_mask = init_connectivity(N, input_size, output_size, radius)
    
    # For LSTM, we need to expand W_in and dale_mask for the 4 gates
    # LSTM weight_ih_l0 has shape (4*hidden_size, input_size)
    W_in_lstm = torch.zeros([4*N, input_size]).float()
    for i in range(4):
        W_in_lstm[i*N:(i+1)*N, :] = W_in
    
    # Create a dale_mask for the input-to-hidden weights
    dale_mask_ih = torch.zeros([4*N, input_size]).float()
    for i in range(4):
        dale_mask_ih[i*N:(i+1)*N, :] = input_mask
    
    # For hidden-to-hidden weights (weight_hh_l0), also shape (4*hidden_size, hidden_size)
    W_rec_lstm = torch.zeros([4*N, N]).float()
    for i in range(4):
        W_rec_lstm[i*N:(i+1)*N, :] = W_rec
    
    # Create a dale_mask for the hidden-to-hidden weights
    dale_mask_hh = torch.zeros([4*N, N]).float()
    for i in range(4):
        dale_mask_hh[i*N:(i+1)*N, :] = dale_mask
    
    return W_rec_lstm, W_in_lstm, W_out, dale_mask_hh, dale_mask_ih, output_mask, input_mask