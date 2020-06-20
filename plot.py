#!/usr/bin/env python
# -*- coding: utf-8 -*-

import arrow
import torch
import numpy as np
import fourierpp as fpp
import matplotlib
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
        vloc  = (yr.mean() + yf.mean())/2
        plt.text(hloc, vloc, r"$J(\theta_0; \theta_1)$",
                horizontalalignment='center', fontsize=20)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xlabel("iteration")
        plt.ylabel(r"log-likelihood $\ell$")

        fig.tight_layout() # otherwise the right y-label is slightly clipped
        pdf.savefig(fig)



def visualize_detection_statistics(clf, rseq, fseq, bseq, seq_len=30):

    rloglik, rmask = evaluate_loglik(clf, rseq, T=10., nf=1000) # [ batch_size, seq_len ]
    floglik, fmask = evaluate_loglik(clf, fseq, T=10., nf=1000) # [ batch_size, seq_len ]
    bloglik, bmask = evaluate_loglik(clf, bseq, T=10., nf=1000) # [ batch_size, seq_len ]

    rloglik = rloglik / (np.arange(dl.seq_len) + 1)
    floglik = floglik / (np.arange(floglik.shape[1]) + 1)
    bloglik = bloglik / (np.arange(bloglik.shape[1]) + 1)
    
    # rm      = ((rloglik * rmask).sum(0) / rmask.sum(0))[:seq_len]
    # fm      = ((floglik * fmask).sum(0) / fmask.sum(0))[:seq_len]
    # bm      = ((bloglik * bmask).sum(0) / bmask.sum(0))[:seq_len]
    rm      = rloglik.mean(0)[:seq_len]
    fm      = floglik.mean(0)[:seq_len]
    bm      = bloglik.mean(0)[:seq_len]
    rc      = 1 * np.std(rloglik, axis=0)[:seq_len]
    fc      = 1 * np.std(floglik, axis=0)[:seq_len]
    bc      = 1 * np.std(bloglik, axis=0)[:seq_len]
    rx      = np.arange(dl.seq_len)[:seq_len] + 1
    fx      = np.arange(floglik.shape[1])[:seq_len] + 1
    bx      = np.arange(bloglik.shape[1])[:seq_len] + 1

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 15}
    plt.rc('font', **font)
    with PdfPages("result/detection-stats-withrand.pdf") as pdf:
        fig, ax = plt.subplots()
        ax.axhline(y=-7.5, linewidth=2, linestyle=":", color='g', label=r"$\eta$ detection threshold")
        ax.plot(rx, rm, c="red", linewidth=2, linestyle="--", label="mean for anomalous seq")
        ax.plot(fx, fm, c="blue", linewidth=2, linestyle="--", label="mean for normal seq")
        ax.plot(bx, bm, c="grey", linewidth=2, linestyle="--", label="mean for random seq")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.fill_between(rx, (rm-rc), (rm+rc), color='red', alpha=.1, label=r"$1\sigma$ for anomalous seq")
        ax.fill_between(fx, (fm-fc), (fm+fc), color='blue', alpha=.1, label=r"$1\sigma$ for normal seq")
        ax.fill_between(bx, (bm-bc), (bm+bc), color='grey', alpha=.1, label=r"$1\sigma$ for random seq")
        plt.ylim(-25, 0)
        plt.xlabel(r"Detect at the $j$-th event")
        plt.ylabel(r"Detection statistics $\ell(x_{1:j};\theta_0)/j$")
        plt.legend(loc='lower right')
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()
        pdf.savefig(fig)



def visualize_detection_threshold(clf, rseq, fseq, bseq, seq_len=30):
    
    rloglik, rmask = evaluate_loglik(clf, rseq, T=10., nf=1000) # [ batch_size, seq_len ]
    floglik, fmask = evaluate_loglik(clf, fseq, T=10., nf=1000) # [ batch_size, seq_len ]
    bloglik, bmask = evaluate_loglik(clf, bseq, T=10., nf=1000) # [ batch_size, seq_len ]

    rloglik = (rloglik / (np.arange(dl.seq_len) + 1))[:, :seq_len]
    floglik = (floglik / (np.arange(floglik.shape[1]) + 1))[:, :seq_len]
    bloglik = (bloglik / (np.arange(bloglik.shape[1]) + 1))[:, :seq_len]
    
    rnalert = np.stack([ (rloglik > eta).sum(0) for eta in np.linspace(-15, -2, 100) ], axis=1)
    fnalert = np.stack([ (floglik > eta).sum(0) for eta in np.linspace(-15, -2, 100) ], axis=1)
    bnalert = np.stack([ (bloglik > eta).sum(0) for eta in np.linspace(-15, -2, 100) ], axis=1)

    prc = rnalert / (rnalert + fnalert)
    rec = rnalert / rloglik.shape[0]
    f1  = 2 * (prc * rec) / (prc + rec)

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 15}
    plt.rc('font', **font)
    with PdfPages("result/detection-recall.pdf") as pdf:
        fig, ax = plt.subplots()
        cmap    = matplotlib.cm.get_cmap('magma') # matplotlib.cm.get_cmap('plasma')
        img     = ax.imshow(rec, 
            interpolation='nearest', origin='lower', cmap=cmap, 
            extent=[-15,-2,0,30], aspect=0.42)
        ax.set_xlabel(r'Threshold $\eta$')
        ax.set_ylabel(r'Detect at the $j$-th event')
        cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label(r'Recall')
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()
        pdf.savefig(fig)
    # region  = np.s_[5:50, 5:50]
    # x, y, z = x[region], y[region], z[region]

    # fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    # ls = LightSource(270, 45)
    # # To use a custom hillshading mode, override the built-in shading and pass
    # # in the rgb colors of the shaded surface calculated from "shade".
    # rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    # surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
    #                     linewidth=0, antialiased=False, shade=False)



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

    # rloglik = np.load("loginfo/adv-loglik.npy")
    # floglik = np.load("loginfo/adv-loglikhat.npy")
    # visualize_training_iterations(rloglik, floglik)

    # ------------------------------------------------------------------------
    #
    # Visualize model output
    # 
    # ------------------------------------------------------------------------

    fpp.load_state_dict(torch.load("savedmodels/adv-fpp.pt"))
    slstm.load_state_dict(torch.load("savedmodels/adv-slstm.pt"))

    # loglikelihood trajectories evaluation
    batch_size = 500
    rseq = dl.data[:batch_size, :]                     # [ batch_size, seq_len, dsize ]
    fseq = slstm(batch_size=batch_size, seq_len=50)    # [ batch_size, seq_len, dsize ]
    fseq = truncatebyT(fseq, T=10)
    bseq = random_seqs(batch_size, seq_len=30, mu=5, T=10)
    
    visualize_detection_statistics(fpp, rseq, fseq, bseq, seq_len=30)
    # visualize_detection_threshold(fpp, rseq, fseq, bseq, seq_len=30)
    

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