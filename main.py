#!/usr/bin/env python
# -*- coding: utf-8 -*-

import arrow
import torch
import numpy as np
import fourierpp as fpp

from dataloader import Dataloader4TemporalOnly
from fourierpp import FourierPointProcess, train
from stochlstm import StochasticLSTM, advtrain
from utils import log_callback



if __name__ == "__main__":

    torch.manual_seed(0)

    dl = Dataloader4TemporalOnly(path="data/hawkes_data.npy", batch_size=50)

    nsize = 10 # noise dimension
    fsize = 20 # fourier feature dimension
    dsize = 1  # data dimension
    hsize = 50 # hidden state dimension
    fpp   = FourierPointProcess(nsize, fsize, dsize)
    slstm = StochasticLSTM(dsize, hsize)

    # # learn fpp by MLE
    # train(fpp, dl, n_epoch=10, log_interval=20, lr=1e-4) # , log_callback=log_callback)
    # torch.save(fpp.state_dict(), "savedmodels/fpp-v2.pt")

    # learn fpp and slstm by adversarial learning
    recloglik, recloglikhat = advtrain(slstm, fpp, dl, seq_len=50, K=2,
        n_epoch=3, log_interval=20, glr=1e-7, clr=1e-5)
        # log_callback=lambda x, y, z: None)
    
    np.save("loginfo/adv-loglik-v2.npy", recloglik)
    np.save("loginfo/adv-loglikhat-v2.npy", recloglikhat)

    torch.save(fpp.state_dict(), "savedmodels/adv-fpp-v2.pt")
    torch.save(slstm.state_dict(), "savedmodels/adv-slstm-v2.pt")