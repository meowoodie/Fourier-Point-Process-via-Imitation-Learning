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

        ingate     = torch.nn.functional.sigmoid(ingate)          # [ batch_size, hsize ]
        forgetgate = torch.nn.functional.sigmoid(forgetgate)      # [ batch_size, hsize ]
        cellgate   = torch.nn.functional.tanh(cellgate)           # [ batch_size, hsize ]
        outgate    = torch.nn.functional.sigmoid(outgate)         # [ batch_size, hsize ]

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.nn.functional.tanh(cy)

        return hy, cy

 

def advtrain(generator, classifier, dataloader, seq_len=100,
    n_epoch=10, log_interval=10, lr=1e-4, log_callback=lambda x, y: None):
    """
    adversarial learning

    NOTE: here we use stochastic LSTM as our generator and a point process based model as our classifier
    """
    goptimizer = optim.Adadelta(generator.parameters(), lr=lr)
    coptimizer = optim.Adadelta(classifier.parameters(), lr=lr)
    for e in range(n_epoch):
        avgcloss, avggloss = [], []
        dataloader.shuffle()
        for i in range(int(len(dataloader)/2)):
            # collect real and fake sequences
            generator.eval()
            X    = dataloader[i]                             # real sequences [ batch, seq_len1, dszie ]
            Xhat = generator(dataloader.batch_size, seq_len) # fake sequences [ batch, seq_len2, dszie ]
            Xhat = truncatebyT(Xhat)                         # truncate generated sequence by time horizon T

            classifier.train()
            coptimizer.zero_grad()                           # init optimizer (set gradient to be zero)
            generator.train()
            goptimizer.zero_grad()
            
            _, loglik    = classifier(X)                     # log-likelihood of real sequences [ batch ]
            _, loglikhat = classifier(Xhat)                  # log-likelihood of real sequences [ batch ]

            # train classifier
            closs        = loglik.mean() - loglikhat.mean()  # log-likelihood discrepancy
            avgcloss.append(closs.item())
            closs.backward()                                 # gradient descent
            coptimizer.step()                                # update optimizer
            # train generator
            gloss        = - closs                           # log-likelihood discrepancy
            avggloss.append(gloss.item())
            gloss.backward()                                 # gradient descent
            goptimizer.step()                                # update optimizer

            if i % log_interval == 0 and i != 0:
                print("[%s] Train batch: %d\tgLoss: %.3f\tcLoss: %.3f" % \
                    (arrow.now(), i, gcloss.item(), ccloss.item()))
                # callback 
                log_callback(generator, classifier, dataloader)
        
        # log loss
        print("[%s] Train epoch: %d\tAvg gLoss: %.3f\tAvg cLoss: %.3f" % \
            (arrow.now(), e, sum(avggloss) / len(dataloader), sum(avgcloss) / len(dataloader)))



if __name__ == "__main__":

    batch_size = 2
    seq_len    = 10
    dsize      = 3
    hsize      = 5

    slstm = StochasticLSTM(dsize=dsize, hsize=hsize)
    print(slstm(batch_size, seq_len))
    
        