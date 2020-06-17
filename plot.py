#!/usr/bin/env python
# -*- coding: utf-8 -*-

import arrow
import torch
import numpy as np
import fourierpp as fpp
import matplotlib.pyplot as plt

from dataloader import Dataloader4TemporalOnly
from fourierpp import FourierPointProcess, train
from stochlstm import StochasticLSTM, advtrain, truncatebyT
from utils import evaluate_kernel, evaluate_lambda, evaluate_loglik

from matplotlib.patches import Polygon
from matplotlib.backends.backend_pdf import PdfPages

def visualize_training_iterations(rloglik, floglik):
    n  = len(floglik)
    x  = list(range(n))
    yr = rloglik
    yf = floglik

    plt.rc('text', usetex=True)
    # plt.rc("font", family="serif")
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 15}
    plt.rc('font', **font)
    with PdfPages("result/iter_vs_loglik.pdf") as pdf:
        fig, ax = plt.subplots()
        plt.plot(x, yr, 'r', linewidth=2)
        plt.plot(x, yf, 'b', linewidth=2)
        plt.ylim(ymin=0)

        # Make the shaded region
        ix    = list(range(n))
        iyr   = rloglik
        iyf   = floglik
        vert1 = list(zip(ix, iyr))
        vert2 = list(zip(ix, iyf))
        vert2.reverse()
        verts = vert1 + vert2
        poly  = Polygon(verts, facecolor='0.8', edgecolor='0.5')
        ax.add_patch(poly)

        hloc  = 0.8 * n
        vloc  = 150 # (max(yr) + max(yf))/2
        plt.text(hloc, vloc, r"$J(\theta_0; \theta_1)$",
                horizontalalignment='center', fontsize=20)

        # plt.figtext(0.9, 0.05, '$x$')
        # plt.figtext(0.1, 0.9, '$y$')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xlabel("iteration")
        plt.ylabel(r"log-likelihood $\ell$")
        # ax.xaxis.set_ticks_position('bottom')

        # ax.set_xticks((a, b))
        # ax.set_xticklabels(('$a$', '$b$'))
        # ax.set_yticks([])

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        pdf.savefig(fig)


if __name__ == "__main__":

    torch.manual_seed(0)

    dl = Dataloader4TemporalOnly(path="data/hawkes_data.npy", batch_size=50)

    nsize = 10 # noise dimension
    fsize = 20 # fourier feature dimension
    dsize = 1  # data dimension
    hsize = 10 # hidden state dimension
    fpp   = FourierPointProcess(nsize, fsize, dsize)
    slstm = StochasticLSTM(dsize, hsize)

    # ------------------------------------------------------------------------
    #
    # Visualize log information
    # 
    # ------------------------------------------------------------------------

    # rloglik = np.load("loginfo/loglik.npy")
    # floglik = np.load("loginfo/loglikhat.npy")
    # visualize_training_iterations(rloglik, floglik)

    # ------------------------------------------------------------------------
    #
    # Visualize model output
    # 
    # ------------------------------------------------------------------------

    fpp.load_state_dict(torch.load("savedmodels/fpp.pt"))
    slstm.load_state_dict(torch.load("savedmodels/slstm.pt"))

    # loglikelihood trajectories evaluation
    rseq        = dl.data[:100, :15]                          # [ batch_size, seq_len, dsize ]
    # fseq        = slstm(batch_size=100, seq_len=15)           # [ batch_size, seq_len, dsize ]
    # fseq        = truncatebyT(fseq)
    fseq        = torch.FloatTensor(100, 15).uniform_(0, 10)
    rloglik, rm = evaluate_loglik(fpp, rseq, T=10., nf=10000) # [ batch_size, seq_len ]
    floglik, fm = evaluate_loglik(fpp, fseq, T=10., nf=10000) # [ batch_size, seq_len ]

    for i in range(10):
        rx = np.where(rm[i])[0]
        fx = np.where(fm[i])[0]
        plt.plot(rx, rloglik[i, :len(rx)], c="r")
        plt.plot(fx, floglik[i, :len(fx)], c="b")
        
    plt.show()

    # # kernel function evaluation
    # kij   = evaluate_kernel(fpp)
    # plt.plot(kij)
    # plt.show()

    # # generate fake sequences
    # fseq  = slstm(batch_size=1, seq_len=20)
    # fseq  = truncatebyT(fseq)
    # # lambda function evaluation for fake sequence
    # lam   = evaluate_lambda(fpp, fseq)
    # plt.plot(lam)
    # plt.show()

    # # lambda function evaluation for real sequence
    # rseq  = dl.data[2].unsqueeze_(0) # [ 1, seq_len, dsize ]
    # lam   = evaluate_lambda(fpp, rseq)
    # plt.plot(lam)
    # plt.show()