import torch

import numpy as np

from torch.utils import data


class RegressionDataset(data.Dataset):
    '''
    Data set for the experiment in Section 4 of the paper
    '''

    def __init__(self, sigmas, epsilons):
        '''
        Initialize the dataset
        Inputs:
            sigmas: ($\sigma_i$) fixed scalars that set the scales of the outputs of each function $f_i$
            epsilons: ($\epsilon_i$) task-specific information
        '''

        # B is a constant matrix with its elemenets generated IID
        # from a normal distribution N(0,10)
        self.B = np.random.normal(scale=10, size=(100, 250)).astype(np.float32)
        
        # check if the given epsilons have the appropriate size
        assert epsilons.shape == (len(sigmas), 100, 250)

        # assign the epsilons and the sigmas
        self.sigmas = np.array(sigmas).astype(np.float32)
        self.epsilons = np.array(epsilons).astype(np.float32)

    
    def __len__(self):
        return 100


    def __getitem__(self, index):
        
        # retrieve a single input sample with d=250, normalized
        x = np.random.uniform(-1, 1, size=(250,)).astype(np.float32)
        x = x / np.linalg.norm(x)

        # retrieve one target value for each of the tasks
        ys = []
        for i in range(len(self.sigmas)):
            # eq (3) on the paper:
            # each target is $\sigma_i \tanh((B + \epsilon_i)) \mathbf{x}) $
            ys.append(
                self.sigmas[i] * np.tanh((self.B + self.epsilons[i]).dot(x))
            )
        ys = np.stack(ys)
        
        # move everything to torch variables
        x = torch.from_numpy(x).float()
        ys = torch.from_numpy(ys).float()

        return x, ys