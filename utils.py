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

def evaluate_loglik(model, X, T=10., nf=10000):
    """
    evaluate lambda function in point process function

    Args:
    - X: history points [ batch_size, seq_len, dszie ]
    """
    _, loglik     = model(X, nf=nf, T=T) # [ batch_size, seq_len + 1 ]
    mask          = (loglik != 0).numpy()[:, :-1]
    cumsum_loglik = torch.cumsum(loglik, dim=1)[:, :-1]
    return cumsum_loglik.detach().numpy(), mask

def random_seqs(batch_size, seq_len, mu, T):
    """
    generate random sequences
    """
    seqs = np.zeros((batch_size, seq_len))
    Ns   = np.random.poisson(size=batch_size, lam=mu)
    for i in range(batch_size):
        seq             = np.random.uniform(0, T, Ns[i])
        seq.sort()
        seqs[i, :Ns[i]] = seq
    seqs = torch.FloatTensor(seqs).unsqueeze_(2)
    return seqs


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