import numpy as np
import torch
import sys

class ToyModel():
    def __init__(self, thetas):
        self.thetas = thetas

    def executeSimulation(self, device, xDim, thetaDim):
        self.device = device
        mean = self.thetas ** 2
        cov = torch.eye(self.thetas.shape[1]).to(device)
        distribution = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)
        return distribution.sample([int(xDim/mean.shape[1])]).transpose(1,0).reshape(-1,xDim)

    def log_prob(self, context='', inputs=''):
        mean = context ** 2
        cov = torch.eye(context.shape[1]).to(self.device)
        distribution = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)
        ll = torch.zeros((context.shape[0])).to(self.device)
        #print("!!! : ", x.shape, thetas.shape)
        for k in range(int(inputs.shape[1] / 8)):
            ll = ll + distribution.log_prob(inputs[:,8 * k: 8 * (k + 1)])
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