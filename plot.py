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
from utils import evaluate_kernel, evaluate_lambda, evaluate_loglik, random_seqs

from matplotlib.patches import Polygon
from matplotlib.backends.backend_pdf import PdfPages

def visualize_training_iterations(rloglik, floglik):

    n  = len(floglik)
    x  = list(range(n))
    yr = rloglik
    yf = floglik

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 15}
    plt.rc('font', **font)
    with PdfPages("result/iter_vs_loglik.pdf") as pdf:
        fig, ax = plt.subplots()
        plt.plot(x, yr, 'r', linewidth=2)
        plt.plot(x, yf, 'b', linewidth=2)

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
        vloc  = (max(yr) + max(yf))/2
        plt.text(hloc, vloc, r"$J(\theta_0; \theta_1)$",
                horizontalalignment='center', fontsize=20)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xlabel("iteration")
        plt.ylabel(r"log-likelihood $\ell$")

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        pdf.savefig(fig)



def visualize_detection_statistics(clf, rseq, fseq, seq_len=30):

    rloglik, rmask = evaluate_loglik(clf, rseq, T=10., nf=1000)  # [ batch_size, seq_len ]
    floglik, fmask = evaluate_loglik(clf, fseq, T=10., nf=1000)  # [ batch_size, seq_len ]

    rloglik = rloglik / np.arange(dl.seq_len)
    floglik = floglik / np.arange(50)
    rm      = rloglik.mean(0)[:seq_len]
    fm      = floglik.mean(0)[:seq_len]
    rc      = 1 * np.std(rloglik, axis=0)[:seq_len]
    fc      = 1 * np.std(floglik, axis=0)[:seq_len]
    rx      = np.arange(dl.seq_len)[:seq_len]
    fx      = np.arange(50)[:seq_len]

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 15}
    plt.rc('font', **font)
    with PdfPages("result/detection-stats.pdf") as pdf:
        fig, ax = plt.subplots()
        ax.plot(rx, rm, c="r", linewidth=2, linestyle="--", label="mean for anomalous event")
        ax.plot(fx, fm, c="b", linewidth=2, linestyle="--", label="mean for normal event")
        ax.axhline(y=-15, linewidth=2, linestyle=":", color='g', label=r"$\eta$ detection threshold")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.fill_between(rx, (rm-rc), (rm+rc), color='r', alpha=.1, label=r"$1\sigma$ region for anomalous event")
        ax.fill_between(fx, (fm-fc), (fm+fc), color='b', alpha=.1, label=r"$1\sigma$ region for normal event")
        plt.xlabel(r"$j$-th event")
        plt.ylabel(r"detection statistics $\ell(x_{1:j};\theta_0)/j$")
        plt.legend(loc='lower right')
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()
        pdf.savefig(fig)


if __name__ == "__main__":

    torch.manual_seed(0)

    dl = Dataloader4TemporalOnly(path="data/hawkes_data.npy", batch_size=50)

    nsize = 10 # noise dimension
    fsize = 20 # fourier feature dimension
    dsize = 1  # data dimension
    hsize = 50 # hidden state dimension
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
    batch_size = 500
    rseq = dl.data[:batch_size, :]                     # [ batch_size, seq_len, dsize ]
    fseq = slstm(batch_size=batch_size, seq_len=50)    # [ batch_size, seq_len, dsize ]
    fseq = truncatebyT(fseq)
    # fseq = random_seqs(batch_size, seq_len=50, mu=15, T=10)
    visualize_detection_statistics(fpp, rseq, fseq, seq_len=30)

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