#!/usr/bin/env python
# -*- coding: utf-8 -*-

import arrow
import torch
import numpy as np
import torch.optim as optim

def evaluate_kernel(model, T=10., ngrid=100, nf=10000):
    """evaluate kernel function in point process model from 0 to T"""
    xi  = torch.from_numpy(np.zeros((ngrid, 1))).float()                       # [ batch_size, dsize ]
    xj  = torch.from_numpy(np.linspace(0, T, num=ngrid)).unsqueeze_(1).float() # [ batch_size, dsize ]
    kij = model._fourier_kernel(xi, xj, nf=nf)                                 # [ batch_size, 1 ]
    kij = kij.detach().numpy()
    return kij

def evaluate_lambda(model, dataloader, T=10., ngrid=100, nf=10000):
    """evaluate lambda function in point process function"""
    X = torch.from_numpy(np.linspace(0, T, num=ngrid)).\
        unsqueeze_(1).unsqueeze_(0).float()                 # [ 1, ngrid, dsize ]
    H = dataloader.data[0].unsqueeze_(0)                    # [ 1, seq_len, dsize ]
    lam = []
    for i in range(ngrid):
        xi = X[:, i, :].clone()                             # [ 1, dsize ] 
        ht = H[(H > 0.) * (H < xi)]\
            .unsqueeze_(1).unsqueeze_(0)                    # [ 1, seq_len < i, dsize ]
        lami = model._lambda(xi, ht, nf=nf)
        lam.append(lami.detach().numpy())
    return np.array(lam)

def log_callback(model, dataloader):
    """callback function invoked at every log interval"""
    # evaluation mode
    model.eval()
    
    # kernel function evaluation
    kij = evaluate_kernel(model)
    plt.plot(kij)
    plt.show()

    # lambda function evaluation
    lam = evaluate_lambda(model, dataloader)
    plt.plot(lam)
    plt.show()