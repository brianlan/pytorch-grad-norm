import torch

from torch.nn.modules.loss import MSELoss

import torch.nn.functional as F





class RegressionTrain(torch.nn.Module):
    '''
    '''

    def __init__(self, model):
        '''
        '''

        # initialize the module using super() constructor
        super(RegressionTrain, self).__init__()
        # assign the architectures
        self.model = model
        # assign the weights for each task
        self.weights = torch.nn.Parameter(torch.ones(model.n_tasks).float())
        # loss function
        self.mse_loss = MSELoss()

    
    def forward(self, x, ts):
        B, n_tasks = ts.shape[:2]
        ys = self.model(x)
        
        # check if the number of tasks is equal to this size
        assert(ys.size()[1] == n_tasks)
        task_loss = []
        for i in range(n_tasks):
            task_loss.append( self.mse_loss(ys[:,i,:], ts[:,i,:]) )
        task_loss = torch.stack(task_loss)

        return task_loss


    def get_last_shared_layer(self):
        return self.model.get_last_shared_layer()



class RegressionModel(torch.nn.Module):
    '''
    '''

    def __init__(self, n_tasks):
        '''
        Constructor of the architecture.
        Input:
            n_tasks: number of tasks to solve ($T$ in the paper)
        '''

        # initialize the module using super() constructor
        super(RegressionModel, self).__init__()
        
        # number of tasks to solve
        self.n_tasks = n_tasks
        # fully connected layers
        self.l1 = torch.nn.Linear(250, 100)
        self.l2 = torch.nn.Linear(100, 100)
        self.l3 = torch.nn.Linear(100, 100)
        self.l4 = torch.nn.Linear(100, 100)
        # branches for each task
        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), torch.nn.Linear(100, 100))

    
    def forward(self, x):
        # forward pass through the common fully connected layers
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = F.relu(self.l4(h))

        # forward pass through each output layer
        outs = []
        for i in range(self.n_tasks):
            layer = getattr(self, 'task_{}'.format(i))
            outs.append(layer(h))

        return torch.stack(outs, dim=1)


    def get_last_shared_layer(self):
        return self.l4

        