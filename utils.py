#!/usr/bin/env python
# -*- coding: utf-8 -*-

import arrow
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm

def evaluate_kernel(model, T=10., ngrid=100, nf=10000):
    """evaluate kernel function in point process model from 0 to T"""
    xi  = torch.from_numpy(np.zeros((ngrid, 1))).float()                       # [ batch_size, dsize ]
    xj  = torch.from_numpy(np.linspace(0, T, num=ngrid)).unsqueeze_(1).float() # [ batch_size, dsize ]
    kij = model._fourier_kernel(xi, xj, nf=nf)                                 # [ batch_size, 1 ]
    kij = kij.detach().numpy()
    return kij

def evaluate_lambda(model, H, T=10., ngrid=100, nf=10000):
    """
    evaluate lambda function in point process function

    Args:
    - H: history points 
    """
    X = torch.from_numpy(np.linspace(0, T, num=ngrid)).\
        unsqueeze_(1).unsqueeze_(0).float()                 # [ 1, ngrid, dsize ]
    lam = []
    for i in range(ngrid):
        xi = X[:, i, :].clone()                             # [ 1, dsize ] 
        ht = H[(H > 0.) * (H < xi)]\
            .unsqueeze_(1).unsqueeze_(0)                    # [ 1, seq_len < i, dsize ]
        lami = model._lambda(xi, ht, nf=nf)
        lam.append(lami.detach().numpy())
    return np.array(lam)

def evaluate_loglik(model, H, T=10., nf=10000):
    """
    evaluate lambda function in point process function

    Args:
    - H: history points [ batch_size, seq_len, dszie ]
    """
    print("evaluation log-likelihood")
    mask    = (H[:, 1:, 0].clone() > 0).numpy()
    seq_len = H.shape[1]
    logliks = []
    for i in tqdm(range(1, seq_len)):
        X         = H[:, :i, :].clone()  # [ batch_size, i, dsize ]
        _, loglik = model(X, nf=nf, T=T) # [ batch_size ]
        logliks.append(loglik)           
    logliks = torch.stack(logliks, dim=1).detach().numpy()
    return logliks, mask

def log_callback(model, dataloader):
    """callback function invoked at every log interval"""
    # evaluation mode
    model.eval()
    
    # kernel function evaluation
    kij = evaluate_kernel(model)
    plt.plot(kij)
    plt.show()

    # lambda function evaluation
    H   = dataloader.data[0].unsqueeze_(0)   # [ 1, seq_len, dsize ]
    lam = evaluate_lambda(model, H)
    plt.plot(lam)
    plt.show()