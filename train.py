
import argparse
import torch

import numpy as np

from dataset import RegressionDataset
from model import RegressionModel, RegressionTrain

import matplotlib.pyplot as plt

from torch.utils import data
from torch.autograd import Variable

import torch.nn.functional as F


def train_toy_example(args):

    # set the random seeds for reproducibility
    np.random.seed(123)
    torch.cuda.manual_seed_all(123)
    torch.manual_seed(123)

    # define the sigmas, the number of tasks and the epsilons
    # for the toy example
    sigmas = [1.0, 100.0]
    n_tasks = len(sigmas)
    epsilons = np.random.normal(scale=3.5, size=(n_tasks, 100, 250)).astype(np.float32)

    # initialize the data loader
    dataset = RegressionDataset(sigmas, epsilons)
    data_loader = data.DataLoader(dataset, batch_size=200, num_workers=4, shuffle=False)

    # initialize the model and use CUDA if available
    model = RegressionTrain(RegressionModel(n_tasks))
    if torch.cuda.is_available():
        model.cuda()

    # initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    n_iterations = int(args.n_iter)
    weights = []
    task_losses = []
    loss_ratios = []
    final_layer_names = ['task_{}'.format(i) for i in range(n_tasks)]

    # run n_iter iterations of training
    for t in range(n_iterations):

        # get a single batch
        for (it, batch) in enumerate(data_loader):

            # get the X and the targets values
            X = batch[0]
            ts = batch[1]
            if torch.cuda.is_available():
                X = X.cuda()
                ts = ts.cuda()
            X = torch.tensor(X, requires_grad=True)

            # clear the gradients
            optimizer.zero_grad()

            # evaluate each task loss L_i(t)
            task_loss = model(X, ts) # this will do a forward pass in the model and will also evaluate the loss
            # compute the weighted loss L(t)
            weighted_task_loss = torch.mul(model.weights, task_loss)

            # initialize the initial loss L(0) if t=0
            if t == 0:
                # set L(0)
                initial_task_loss = task_loss.clone()
            # get the mean loss
            loss = weighted_task_loss.mean()
            # do the backward pass
            loss.backward(retain_graph=True) # need to retain the graph to be able to use autograd

            # ignore a gradient to the coefficient vector,
            # which is computed from the standard loss
            model.weights.grad = model.weights.grad * 0.0

            # switch for each weighting algorithm:
            # --> grad norm
            if args.mode == 'grad_norm':
                
                # use $| \nabla_W w_i * L_i  | = w_i | \nabla_W L_i |$
                
                # get the GradNorm gradients
                gygw_norms = []
                for i, layer_name in enumerate(final_layer_names):
                    layer = getattr(model.model, layer_name)
                    gygw = torch.autograd.grad(task_loss[i],layer.parameters(), retain_graph=True)[0]
                    gygw_norms.append(torch.norm(gygw))
                gygw_norms = torch.stack(gygw_norms)
                norms = torch.mul(model.weights, gygw_norms)

                mean_norm = torch.mean(norms)
                loss_ratio = task_loss / initial_task_loss
                inverse_train_rate = loss_ratio / torch.mean(loss_ratio)

                diff = (norms - (inverse_train_rate ** torch.tensor(args.alpha)) * mean_norm)
                grad_norm_loss = torch.mean(torch.abs(diff))
                grad_norm_loss.backward(retain_graph=True)

            # do a step with the optimizer
            optimizer.step()

        # renormalize
        normalize_coeff = n_tasks / torch.sum(model.weights.data, dim=0)
        model.weights.data = model.weights.data * normalize_coeff

        # record
        task_losses.append(task_loss.data.numpy())
        loss_ratios.append(np.mean(task_losses[-1] / task_losses[0]))
        weights.append(model.weights.data.numpy())

        if t % 100 == 0:
            print('{}/{}: loss_ratio={}, weights={}, task_loss={}'.format(
                t, args.n_iter, loss_ratios[-1], model.weights.data.numpy(), task_loss.data.numpy()))

    task_losses = np.array(task_losses)
    weights = np.array(weights)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.set_title('loss (task 0)')
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.set_title('loss (task 1')
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.set_title('sum of normalized losses')
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.set_title('change of weights over time')

    ax1.plot(task_losses[:, 0])
    ax2.plot(task_losses[:, 1])
    ax3.plot(loss_ratios)
    ax4.plot(weights[:, 0])
    ax4.plot(weights[:, 1])
    plt.show()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GradNorm')
    parser.add_argument('--n-iter', '-it', type=int, default=25000)
    parser.add_argument('--mode', '-m', choices=('grad_norm', 'equal_weight'), default='grad_norm')
    parser.add_argument('--alpha', '-a', type=float, default=0.12)
    args = parser.parse_args()

    train_toy_example(args)