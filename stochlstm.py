#!/usr/bin/env python
# -*- coding: utf-8 -*-

import arrow
import torch
import numpy as np
import torch.optim as optim



class StochasticLSTM(torch.nn.Module):
    """
    Stochastic LSTM
    """
    def __init__(self, dsize, hsize):
        super(StochasticLSTM, self).__init__()
        self.dsize = dsize
        self.hsize = hsize
        self.Wif   = torch.nn.Parameter(torch.FloatTensor(dsize, hsize * 4).uniform_(0, 1))
        self.Whf   = torch.nn.Parameter(torch.FloatTensor(hsize, hsize * 4).uniform_(0, 1))
        self.h2z   = torch.nn.Sequential(
            torch.nn.Linear(hsize, 100),
            torch.nn.Softplus(), 
            torch.nn.Linear(100, 2 * dsize),
            torch.nn.Softplus())

    def forward(self, batch_size, seq_len):
        """
        custom forward function generating sequences
        """
        # initial hidden and cell state
        h = torch.FloatTensor(batch_size, self.hsize).uniform_(0, 1) # [ batch_size, hsize ]
        c = torch.FloatTensor(batch_size, self.hsize).uniform_(0, 1) # [ batch_size, hsize ]
        # initial output / input
        o = self._stochastic_forward(h)                              # [ batch_size, dsize ]
        # perform recurrent structure
        output = []
        for i in range(seq_len):
            o, h, c = self._stochastic_LSTM_cell(o, h, c)
            output.append(o)
        output = torch.stack(output, dim=1)                          # [ batch_size, seq_len, dsize ]
        return output

    def _stochastic_LSTM_cell(self, _input, h, c):
        """
        custom stochastic LSTM cell

        Args: 
        - _input: input point  [ batch_size, dsize ]
        - h:      hidden state [ batch_size, hsize ]
        - c:      cell state   [ batch_size, hsize ]
        
        Return:
        - output: output point [ batch_size, dsize ]
        - h:      hidden state [ batch_size, hsize ]
        - c:      cell state   [ batch_size, hsize ]
        """
        batch_size = _input.shape[0]
        h, c       = self.LSTMCell(_input, (h, c), self.Wif, self.Whf) # [ batch_size, hsize ]
        # stochastic forward
        output     = self._stochastic_forward(h)                       # [ batch_size, dsize ]
        return output, h, c

    def _stochastic_forward(self, h):
        """
        stochastic forward

        Reference:
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Drawing_values_from_the_distribution
        https://discuss.pytorch.org/t/multivariate-normal-sampling-function/1615/5

        Args:
        - h: hidden state [ batch_size, hsize ]
        """
        batch_size = h.shape[0]
        z          = self.h2z(h)                                           # [ batch_size, 2 * dsize ]
        mean, var  = z[:, :self.dsize].clone(), z[:, self.dsize:].clone()  # [ batch_size, dsize ]
        output     = []
        # draw samples from Gaussian distribution specified by mean and var
        for b in range(batch_size):
            # take target mean and covariance
            m = mean[b, :].clone()                                         # [ dsize ]
            c = var[b, :].clone()                                          # [ dsize ]
            c = np.array(torch.diag(c).detach().numpy(), dtype=np.float32) # [ dsize, dsize ]
            # compute cholesky factor in numpy
            l = torch.from_numpy(np.linalg.cholesky(c))                    # [ dsize, dsize ]
            # sample standard normal random and o = m + l * randn
            o = torch.mm(l, torch.randn(self.dsize, 1)) + m.unsqueeze_(1)  # [ dsize, 1 ]
            output.append(o)
        output = torch.cat(output, dim=1).transpose(0, 1)                  # [ batch_size, dsize ]
        return torch.nn.functional.softplus(output)                        # [ batch_size, dsize ]

    @staticmethod
    def LSTMCell(_input, hc, w_if, w_hf, b_if=None, b_hf=None):
        """
        standard LSTM cell

        Reference:
        https://github.com/pytorch/pytorch/blob/c62490bf597ec93f308a8b0108522aa9b40701d9/torch/nn/_functions/rnn.py#L23
        https://medium.com/@andre.holzner/lstm-cells-in-pytorch-fab924a78b1c

        Args:
        - _input: input point           [ batch_size, dsize ]
        - hc:     hidden and cell state [ batch_size, hsize ]
        - w_if:   weight                [ dsize, hsize ]      
        - w_hf:   weight                [ hsize, hsize ]    
        - b_if:   (optional) bias       [ hsize ]  
        - b_hf:   (optional) bias       [ hsize ]

        Returns:
        - hy:     updated hidden state  [ batch_size, hsize ]
        - cy:     updated cell state    [ batch_size, hsize ]
        """
        hx, cx     = hc                                           # [ batch_size, hsize ]
        w_if, w_hf = w_if.transpose(0, 1), w_hf.transpose(0, 1)   # [ hsize, dsize ] and [ hsize, hsize ]
        gates      = torch.nn.functional.linear(_input, w_if, b_if) + \
            torch.nn.functional.linear(hx, w_hf, b_hf)            # [ batch_size, hsize * 4 ]

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1) # [ batch_size, hsize ]

        ingate     = torch.sigmoid(ingate)          # [ batch_size, hsize ]
        forgetgate = torch.sigmoid(forgetgate)      # [ batch_size, hsize ]
        cellgate   = torch.tanh(cellgate)           # [ batch_size, hsize ]
        outgate    = torch.sigmoid(outgate)         # [ batch_size, hsize ]

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy



def truncatebyT(X, T=10.):
    """
    truncate sequences by time horizon T

    Args:
    - X: input sequences where the first dimension represents time [ batch_size, seq_len, dsize ]
    """
    dsize      = X.shape[-1]
    X[:, :, 0] = torch.cumsum(X[:, :, 0].clone(), dim=1)                    # [ batch_size, seq_len ]
    mask       = (X[:, :, 0].clone() < T).unsqueeze_(2).repeat(1, 1, dsize) # [ batch_size, seq_len, dsize ]
    X          = X * mask                                                   # [ batch_size, seq_len, dsize ]
    return X
 


def advtrain(generator, classifier, dataloader, seq_len=100, K=1,
    n_epoch=10, log_interval=10, glr=1e-4, clr=1e-4, log_callback=lambda x, y, z: None):
    """
    adversarial learning

    NOTE: here we use stochastic LSTM as our generator and a point process based model as our classifier
    """
    goptimizer = optim.Adadelta(generator.parameters(), lr=glr)
    coptimizer = optim.Adadelta(classifier.parameters(), lr=clr)
    for e in range(n_epoch):
        avgloglik, avgloglikhat  = [], []
        logloglik, logloglikehat = [], []
        dataloader.shuffle()
        for i in range(len(dataloader)):
            # collect real and fake sequences
            X    = dataloader[i]                             # real sequences [ batch, seq_len1, dszie ]
            Xhat = generator(dataloader.batch_size, seq_len) # fake sequences [ batch, seq_len2, dszie ]
            Xhat = truncatebyT(Xhat)                         # truncate generated sequence by time horizon T
            
            _, loglik    = classifier(X)                     # log-likelihood of real sequences [ batch ]
            _, loglikhat = classifier(Xhat)                  # log-likelihood of real sequences [ batch ]
            exploglik    = loglik.mean()
            exploglikhat = loglikhat.mean()
            closs        = exploglikhat - exploglik          # log-likelihood discrepancy
            gloss        = exploglik - exploglikhat          # log-likelihood discrepancy
            # average epoch loss 
            avgloglik.append(exploglik.item())
            avgloglikhat.append(exploglikhat.item())
            # average log loss
            logloglik.append(exploglik.item())
            logloglikehat.append(exploglikhat.item())

            # train classifier
            classifier.train()        
            coptimizer.zero_grad()   
            closs.backward(retain_graph=True)                # gradient descent
            coptimizer.step()                                # update optimizer
            # train generator
            for k in range(K):
                generator.train()
                goptimizer.zero_grad()
                gloss.backward(retain_graph=True)                # gradient descent
                goptimizer.step()                                # update optimizer

            if i % log_interval == 0 and i != 0:
                print("[%s] Train batch: %d\tLoglik: %.3f\tLoglik hat: %.3f\tdiff: %.3f" % \
                    (arrow.now(), i, 
                    sum(logloglik) / log_interval, 
                    sum(logloglikehat) / log_interval, 
                    sum(logloglik) / log_interval - sum(logloglikehat) / log_interval))
                # callback 
                log_callback(generator, classifier, dataloader)
                logloglik, logloglikehat = [], []
        
        # log loss
        print("[%s] Train epoch: %d\tAvg Loglik: %.3f\tAvg loglik hat: %.3f\tdiff: %.3f" % \
            (arrow.now(), e, 
            sum(avgloglik) / len(dataloader), 
            sum(avgloglikhat) / len(dataloader),
            sum(avgloglik) / len(dataloader) - sum(avgloglikhat) / len(dataloader)))



if __name__ == "__main__":

    batch_size = 2
    seq_len    = 10
    dsize      = 3
    hsize      = 5

    slstm = StochasticLSTM(dsize=dsize, hsize=hsize)
    seqs  = slstm(batch_size, seq_len)
    print(seqs)
    print(truncatebyT(seqs))