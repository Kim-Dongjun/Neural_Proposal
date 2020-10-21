import numpy as np
import torch
import sys

class ToyModel():
    def __init__(self, thetas):
        self.thetas = thetas

    def executeSimulation(self, device, xDim, thetaDim):
        mean = self.thetas[:,:2] ** 2
        if thetaDim != 2:
            diag1 = self.thetas[:,2].reshape(-1,1) ** 2
            diag2 = self.thetas[:,3].reshape(-1,1) ** 2
            corr = torch.tanh(self.thetas[:,4]).reshape(-1,1)
        else:
            diag1 = torch.Tensor([-1.] * self.thetas.shape[0]).to(device).reshape(-1,1) ** 2
            diag2 = torch.Tensor([-0.9] * self.thetas.shape[0]).to(device).reshape(-1,1) ** 2
            corr = torch.tanh(torch.Tensor([-0.6] * self.thetas.shape[0])).to(device).reshape(-1,1)
        cov = torch.cat((diag1 ** 2, corr * diag1 * diag2, corr * diag1 * diag2, diag2 ** 2), 1).reshape(-1,2,2)
        distribution = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov + 1e-6 * torch.eye(2).to(device))
        return distribution.sample([int(xDim/mean.shape[1])]).transpose(1,0).reshape(-1,xDim)

    def log_prob(self, context='', inputs=''):
        if context.get_device() == -1:
            device = 'cpu'
        else:
            device = 'cuda:' + str(context.get_device())
        mean = context[:, :2] ** 2
        if context.shape[1] != 2:
            diag1 = context[:, 2].reshape(-1, 1) ** 2
            diag2 = context[:, 3].reshape(-1, 1) ** 2
            corr = torch.tanh(context[:, 4]).reshape(-1, 1)
        else:
            diag1 = torch.Tensor([-1.] * context.shape[0])
            diag2 = torch.Tensor([-0.9] * context.shape[0])
            corr = torch.tanh(torch.Tensor([-0.6] * context.shape[0]))
        cov = torch.cat((diag1 ** 2, corr * diag1 * diag2, corr * diag1 * diag2, diag2 ** 2), 1).reshape(-1, 2, 2)
        distribution = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov + 1e-6 * torch.eye(2).to(device))
        ll = torch.zeros((context.shape[0])).to(device)
        for k in range(int(inputs.shape[1] / 2)):
            ll = ll + distribution.log_prob(inputs[:,2 * k: 2 * (k + 1)])
        return ll.detach()

if __name__ == '__main__':
    #thetas = torch.randn((100,5))
    thetas = []
    for _ in range(100):
        thetas.append([0.3,0.7,0.6,0.4,0.1])
    thetas = torch.Tensor(thetas).reshape(-1,5)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    toy = ToyModel(thetas, device)
    print(toy.executeSimulation())