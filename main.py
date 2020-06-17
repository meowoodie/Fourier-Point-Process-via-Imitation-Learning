#!/usr/bin/env python
# -*- coding: utf-8 -*-

import arrow
import torch
import numpy as np
import fourierpp as fpp
import matplotlib.pyplot as plt

from dataloader import Dataloader4TemporalOnly
from fourierpp import FourierPointProcess, train
from stochlstm import StochasticLSTM, advtrain



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

    nsize = 10 # noise dimension
    fsize = 20 # fourier feature dimension
    dsize = 1  # data dimension
    hsize = 10 # hidden state dimension
    fpp   = FourierPointProcess(nsize, fsize, dsize)
    slstm = StochasticLSTM(dsize, hsize)

    # # learn fpp by MLE
    # train(fpp, dl, n_epoch=10, log_interval=25, lr=1e-4, log_callback=log_callback)

    # learn fpp and slstm by adversarial learning
    advtrain(slstm, fpp, dl, seq_len=50, K=10,
        n_epoch=3, log_interval=20, glr=1e-2, clr=1e-4, 
        log_callback=lambda x, y, z: None)


    torch.save(fpp.state_dict(), "savedmodels/fpp.pt")
    torch.save(slstm.state_dict(), "savedmodels/slstm.pt")

    # fpp.load_state_dict(torch.load("savedmodels/fpp.pt"))
    # slstm.load_state_dict(torch.load("savedmodels/slstm.pt"))

    