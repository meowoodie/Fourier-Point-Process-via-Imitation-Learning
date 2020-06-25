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

    # import random
    # data  = np.load("data/macy_data.npy")
    # macy  = np.count_nonzero(data[:, :, 0], axis=1)

    # # idx_len1 = [i for i, x in enumerate(macy == 1) if x]
    # idx_len2 = [i for i, x in enumerate(macy == 2) if x]
    # idx_len3 = [i for i, x in enumerate(macy == 3) if x]
    # idx_len4 = [i for i, x in enumerate(macy == 4) if x]
    # idx_len5 = [i for i, x in enumerate(macy == 5) if x]
    # idx_ge6  = [i for i, x in enumerate(macy > 5) if x]
    
    # # sidx_len1 = random.choices(idx_len1, k=int(len(idx_len1)/10))
    # sidx_len2 = random.choices(idx_len2, k=int(len(idx_len2)/10))
    # sidx_len3 = random.choices(idx_len3, k=int(len(idx_len3)/5))
    # sidx_len4 = random.choices(idx_len4, k=int(len(idx_len4)/2))
    # sidx_len5 = random.choices(idx_len5, k=int(len(idx_len5)/2))
    
    # idx = sidx_len2 + sidx_len3 + sidx_len4 + sidx_len5 + idx_ge6
    # plt.hist(macy[idx], bins=40)
    # plt.show()

    # data = data[idx, 1:, :1] * 10
    # np.save("data/upt_macy_data.npy", data)
    data  = np.load("data/upt_macy_data.npy", data)

    dl    = Dataloader4TemporalOnly(data, batch_size=20) 

    nsize = 10 # noise dimension
    fsize = 20 # fourier feature dimension
    dsize = 1  # data dimension
    hsize = 50 # hidden state dimension
    fpp   = FourierPointProcess(nsize, fsize, dsize)
    slstm = StochasticLSTM(dsize, hsize)

    # # # learn fpp by MLE
    # # train(fpp, dl, n_epoch=10, log_interval=20, lr=1e-4) # , log_callback=log_callback)
    # # torch.save(fpp.state_dict(), "savedmodels/fpp-v2.pt")

    # learn fpp and slstm by adversarial learning
    recloglik, recloglikhat = advtrain(slstm, fpp, dl, seq_len=50, K=2,
        n_epoch=3, log_interval=5, glr=1e-7, clr=1e-5)

    np.save("loginfo/macy-adv-loglik.npy", recloglik)
    np.save("loginfo/macy-adv-loglikhat.npy", recloglikhat)

    torch.save(fpp.state_dict(), "savedmodels/macy-adv-fpp.pt")
    torch.save(slstm.state_dict(), "savedmodels/macy-adv-slstm.pt")