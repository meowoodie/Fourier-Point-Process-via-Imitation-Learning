#!/usr/bin/env python
# -*- coding: utf-8 -*-

import arrow
import torch
import numpy as np
import fourierpp as fpp
import matplotlib.pyplot as plt

from dataloader import Dataloader4TemporalOnly
from fourierpp import FourierPointProcess, train



def log_callback(model, dataloader):
    """callback function invoked at every log interval"""
    # evaluation mode
    model.eval()
    
    # kernel function evaluation
    xi  = torch.from_numpy(np.zeros((100, 1))).float()                        # [ batch_size, dsize ]
    xj  = torch.from_numpy(np.linspace(0, 10, num=100)).unsqueeze_(1).float() # [ batch_size, dsize ]
    kij = model._fourier_kernel(xi, xj, nf=10000)                              # [ batch_size, 1 ]
    kij = kij.detach().numpy()
    plt.plot(kij)
    plt.show()

    # lambda function evaluation
    T          = 10.
    n_points   = 100
    X = torch.from_numpy(np.linspace(0, T, num=n_points)).\
        unsqueeze_(1).unsqueeze_(0).float()                 # [ 1, n_points, dsize ]
    H = dataloader.data[0].unsqueeze_(0)                    # [ 1, seq_len, dsize ]
    lam = []
    for i in range(n_points):
        xi = X[:, i, :].clone()                             # [ 1, dsize ] 
        ht = H[(H > 0.) * (H < xi)]\
            .unsqueeze_(1).unsqueeze_(0)                    # [ 1, seq_len < i, dsize ]
        lami = model._lambda(xi, ht, nf=10000)
        lam.append(lami.detach().numpy())
    plt.plot(lam)
    plt.show()
            

if __name__ == "__main__":

    torch.manual_seed(0)

    dl = Dataloader4TemporalOnly(path="data/hawkes_data.npy", batch_size=50)

    nsize, fsize, dsize = 10, 20, 1
    fpp = FourierPointProcess(nsize, fsize, dsize)
        
    train(fpp, dl, n_epoch=10, log_interval=25, lr=1e-4, log_callback=log_callback)