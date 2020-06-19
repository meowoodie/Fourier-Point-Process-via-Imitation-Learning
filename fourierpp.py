#!/usr/bin/env python
# -*- coding: utf-8 -*-

import arrow
import torch
import numpy as np
import torch.optim as optim



class DeepNoise2Fourier(torch.nn.Module):
    """
    Deep Fourier Feature Generator
    """
    def __init__(self, nsize, fsize):
        """
        Args:
        - nsize: size of input noise
        - fsize: size of output fourier feature
        """
        super(DeepNoise2Fourier, self).__init__()
        self.nsize = nsize
        self.n2f   = torch.nn.Sequential(
            torch.nn.Linear(nsize, 100), # [ nf, 100 ]
            torch.nn.Tanh(), 
            torch.nn.Linear(100, fsize), # [ nf, fsize ]
            torch.nn.Tanh())

    def forward(self, nf, mean=0, std=10):
        """
        custom forward function returning fourier features with number of nf.
        """
        noise   = torch.FloatTensor(nf, self.nsize).normal_(mean=mean,std=std)
        fourier = self.n2f(noise)
        return fourier



class FourierPointProcess(torch.nn.Module):
    """
    Point Process with Deep Fourier Triggering Kernel
    """
    def __init__(self, nsize, fsize, dsize):
        """
        Args:
        - fsize: size of fourier feature dimension
        - dsize: size of data dimension
        """
        super(FourierPointProcess, self).__init__()
        # sub model
        self.n2f   = DeepNoise2Fourier(nsize, fsize)
        # model parameters
        self.W     = torch.nn.Parameter(torch.FloatTensor(fsize, dsize).uniform_(0, 1))
        self.alpha = torch.nn.Parameter(torch.ones(1))

    def forward(self, X, nf=100, T=10.):
        """
        custom forward function returning conditional intensities and corresponding log-likelihood
        """
        # pre-computed fourier features for fast log-likelihood calculation
        # fourier [ nf, fsize ] x W [ fsize, dsize ] = Womg.t [ nf, dsize ]
        self.Womg = torch.matmul(self.n2f(nf), self.W).transpose(0, 1) # [ dsize, nf ]
        # return conditional intensities and corresponding log-likelihood
        return self._log_likelihood(X, T)

    def _fourier_kernel(self, xi, xj, nf=None):
        """
        return the kernel value evaluated at xi and xj. i.e., 
            kernel K(xi,xj) = sum(cos(x1*omega)*cos(x2*omega))

        Args:
        - xi, xj: evaluation point [ batch_size, dsize ]
        - nf:     (optional) if nf is not None, then recalculate the Womg

        Return:
        - kij   : kernel value     [ batch_size, 1 ]
        """
        Womg = self.Womg \
            if nf is None else \
            torch.matmul(self.n2f(nf), self.W).transpose(0, 1)
        cos1 = torch.cos(torch.matmul(xi, Womg))            # [ batch_size, nf ]
        cos2 = torch.cos(torch.matmul(xj, Womg))            # [ batch_size, nf ]
        sin1 = torch.sin(torch.matmul(xi, Womg))            # [ batch_size, nf ]
        sin2 = torch.sin(torch.matmul(xj, Womg))            # [ batch_size, nf ]
        kij  = torch.mean(cos1 * cos2 + sin1 * sin2, dim=1) # [ batch_size ]
        kij  = self.alpha * kij.unsqueeze_(1)               # [ batch_size ]
        return kij
    
    def _lambda(self, xi, ht, nf=None):
        """
        return conditional intensity given x

        Args:
        - xi:   current ith point [ batch_size, dsize ]
        - ht:   history points    [ batch_size, seq_len, dsize ]
        - nf:   number of sampled fourier features

        Return:
        - lami: ith lambda        [ batch_size ]
        """
        batch_size, seq_len, dsize = ht.shape
        if seq_len > 0:
            xi   = xi.unsqueeze_(1).repeat(1, seq_len, 1).reshape(-1, dsize)     # [ batch_size * seq_len, dsize ]
            ht   = ht.reshape(-1, dsize)                                         # [ batch_size * seq_len, dsize ]
            k    = self._fourier_kernel(xi, ht, nf).reshape(batch_size, seq_len) # [ batch_size, seq_len ]
            lami = k.sum(1) + self._mu()                                         # [ batch_size ]
            return lami
        else:
            return torch.ones(batch_size) * self._mu()

    def _log_likelihood(self, X, T):
        """
        return log-likelihood given sequence X

        Args:
        - X:      input points sequence [ batch_size, seq_len, dsize ]
        - T:      time horizon

        Return:
        - lam:    sequence of lambda    [ batch_size, seq_len ]
        - loglik: log-likelihood        [ batch_size ]
        """
        batch_size, seq_len, dsize = X.shape
        lam     = [ self._lambda(X[:, i, :].clone(), X[:, :i, :].clone()) 
            for i in range(seq_len) ]
        lam     = torch.stack(lam, dim=1)                                 # [ batch_size, seq_len ]
        # log-likelihood
        mask    = X[:, :, 0] > 0                                          # [ batch_size, seq_len ]
        loglik1 = torch.log(lam) * mask                                   # [ batch_size, seq_len ]
        loglik1 = torch.cat((loglik1, torch.zeros(batch_size, 1)), dim=1) # [ batch_size, seq_len + 1 ]
        loglik2 = - self._mu() * T * (2 * np.pi) ** (dsize - 1) \
            if dsize > 1 else self._integral4temporal(X, T)               # [ batch_size, seq_len ]
        loglik  = loglik1 + loglik2                                       # [ batch_size, seq_len + 1 ]
        return lam, loglik

    def _integral4temporal(self, X, T):
        """
        the integral term calculation only for one-dimensional point

        Args:
        - X: input points sequence [ batch_size, seq_len, dsize ]
        """
        batch_size, seq_len, dsize = X.shape
        assert dsize == 1, "dsize = %d is not 1." % dsize
        # first mask for shifting zero-paddings to T
        mask1      = (X[:, :, 0] <= 0).float()                            # [ batch_size, seq_len ]
        X[:, :, 0] = X[:, :, 0].clone() + mask1 * T
        # calculate the integral
        nf    = self.Womg.shape[1]
        x0    = torch.zeros(batch_size, 1, dsize)                         # [ batch_size, 1, 1 ]
        xn    = torch.ones(batch_size, 1, dsize) * T                      # [ batch_size, 1, 1 ]
        X     = torch.cat((x0, X, xn), dim=1)                             # [ batch_size, seq_len + 2, 1 ]
        # second mask for masking integral sub-terms
        m0    = torch.ones(batch_size, 1)                                 # [ batch_size, 1 ]
        mask2 = torch.cat((m0, (1. - mask1)), dim=1)                      # [ batch_size, seq_len + 1 ]
        intg0 = torch.zeros(batch_size, 1)
        intgi = []
        for i in range(1, seq_len + 1):
            xi   = X[:, i, :].clone()                                     # [ batch_size, 1 ]
            xi1  = X[:, i+1, :].clone()                                   # [ batch_size, 1 ]
            ht   = X[:, :i, :].clone()                                    # [ batch_size, seq_len=i, 1 ]
            cos1 = torch.cos(- torch.matmul(ht, self.Womg))               # [ batch_size, seq_len=i, nf ]
            cos1 = cos1.sum(1)                                            # [ batch_size, nf ]
            cos2 = torch.cos((xi1 + xi) / 2)                              # [ batch_size, 1 ]
            sin1 = torch.sin((xi1 - xi) / 2)                              # [ batch_size, 1 ]
            sinc = 2 * torch.exp(self.Womg) / self.Womg                   # [ 1, nf ]
            sinc = sinc.squeeze_(0)                                       # [ nf ]
            intg = (cos1 * sinc).mean(1).unsqueeze_(1) * cos2 * sin1      # [ batch_size, 1 ]
            intgi.append(intg)
        intgs = [ intg0 ] + intgi
        intgs = torch.stack(intgs, dim=1).squeeze_() * mask2              # [ batch_size, seq_len + 1 ]
        base  = self._mu() * (X[:, 1:, 0].clone() - X[:, :-1, 0].clone()) # [ batch_size, seq_len + 1 ]
        intgs = base + intgs / nf                                         # [ batch_size, seq_len + 1 ]
        return - intgs                                                    # [ batch_size, seq_len + 1 ]

    def _mu(self):
        """
        return base intensity
        """
        return 10.



def train(model, dataloader, 
    n_epoch=10, log_interval=10, lr=1e-4, log_callback=lambda x, y: None):
    """training procedure"""
    # NOTE: gradient for loss is expected to be None, 
    #       since it is not leaf node. (it's root node)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    for e in range(n_epoch):
        avgloss = []
        logloss = []
        dataloader.shuffle()
        for i in range(len(dataloader)):
            X = dataloader[i]
            model.train()
            optimizer.zero_grad()              # init optimizer (set gradient to be zero)
            _, loglik = model(X)               # inference
            loss      = - loglik.sum(1).mean() # negative log-likelihood
            avgloss.append(loss.item())
            logloss.append(loss.item())
            loss.backward()                    # gradient descent
            optimizer.step()                   # update optimizer
            if i % log_interval == 0 and i != 0:
                print("[%s] Train batch: %d\tLoss: %.3f" % (arrow.now(), i, sum(logloss) / log_interval))
                # callback 
                log_callback(model, dataloader)
                logloss = []
        
        # log loss
        print("[%s] Train epoch: %d\tAvg loss: %.3f" % (arrow.now(), e, sum(avgloss) / len(dataloader)))