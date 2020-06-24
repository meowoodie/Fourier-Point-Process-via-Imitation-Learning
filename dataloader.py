#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import arrow
import numpy as np

class Dataloader4TemporalOnly(torch.utils.data.Dataset):
    """
    Data loader for sequential point data
    """
    def __init__(self, data, batch_size=20):
        self.data       = torch.from_numpy(data).float()
        self.n_seq      = self.data.shape[0]
        self.seq_len    = self.data.shape[1] 
        self.batch_size = batch_size

    def shuffle(self):
        self.data = self.data[torch.randperm(self.n_seq)]

    def __len__(self):
        """return number of mini-batches"""
        return int(self.n_seq / self.batch_size)

    def __getitem__(self, idx):
        """return the idx-th mini-batch"""
        idx      += 1
        start_idx = (idx - 1) * self.batch_size
        start_idx = start_idx if start_idx > 0 else 0
        end_idx   = idx * self.batch_size
        end_idx   = end_idx if end_idx < self.n_seq else self.n_seq - 1
        minibatch = self.data[start_idx:end_idx]
        return minibatch