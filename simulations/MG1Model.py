import torch
import numpy as np
from scipy.stats import expon

class MG1Model():
    def __init__(self, thetas, device, numTimeStep):
        self.thetas = thetas
        self.device = device
        self.numTimeStep = numTimeStep

    def executeSimulation(self):
        y_t = torch.Tensor([0.] * self.thetas.shape[0]).reshape(-1, self.thetas.shape[0]).to(self.device)
        thetas = torch.t(self.thetas).to(self.device)
        length = thetas.shape[1]
        a = torch.zeros(1, length).to(self.device)
        d = torch.zeros(1, length).to(self.device)
        zero = torch.zeros(1, length).to(self.device)
        for t in range(self.numTimeStep):
            s = thetas[0] + thetas[1] * torch.FloatTensor(1, length).uniform_(0, 1).to(self.device)
            a = a + torch.distributions.exponential.Exponential(thetas[2]).rsample().reshape(1, -1).to(self.device)
            #a = a + torch.from_numpy(np.array(expon.rvs(thetas[2].cpu().detach().numpy()),dtype=np.float32)).reshape(1,-1).to(self.device)
            obs = s + torch.max(zero, a - d)
            #print("a, d : ", thetas, a-d)
            d = d + obs
            y_t = torch.cat((y_t, obs))
        y_t = y_t[1:]
        #return torch.t(torch.quantile(y_t, 0.0625 * torch.arange(17), dim=0)).to(self.device)
        mean = torch.t(torch.Tensor(np.quantile(y_t.cpu().detach().numpy(), 0.25 * np.arange(5), axis=0)).to(self.device))
        return mean
        Sigma = 0.1 * torch.eye((mean.shape[1])).repeat(mean.shape[0], 1).reshape(mean.shape[0], mean.shape[1], mean.shape[1]).to(self.device)
        distribution = torch.distributions.multivariate_normal.MultivariateNormal(mean, Sigma)
        return distribution.sample([1]).transpose(1,0).reshape(mean.shape[0],-1)
        #return torch.t(
        #    torch.Tensor(np.quantile(y_t.cpu().detach().numpy(), 0.25 * np.arange(5), axis=0)).to(self.device))

    def log_prob(self, context='', inputs=''):
        pass
        '''y_t = torch.Tensor([0.] * context.shape[0]).reshape(-1, context.shape[0]).to(self.device)
        thetas = torch.t(context).to(self.device)
        length = thetas.shape[1]
        a = torch.zeros(1, length).to(self.device)
        d = torch.zeros(1, length).to(self.device)
        zero = torch.zeros(1, length).to(self.device)
        for t in range(self.numTimeStep):
            s = thetas[0] + thetas[1] * torch.FloatTensor(1, length).uniform_(0, 1).to(self.device)
            a = a + torch.distributions.exponential.Exponential(thetas[2]).rsample().reshape(1, -1).to(
                self.device)
            obs = s + torch.max(zero, a - d)
            d = d + obs
            y_t = torch.cat((y_t, obs))
        y_t = y_t[1:]
        # return torch.t(torch.quantile(y_t, 0.0625 * torch.arange(17), dim=0)).to(self.device)
        mean = torch.t(
            torch.Tensor(np.quantile(y_t.cpu().detach().numpy(), 0.1 * np.arange(11), axis=0)[3:8]).to(self.device))
        Sigma = 0.1 * torch.eye((mean.shape[1])).repeat(mean.shape[0], 1).reshape(mean.shape[0], mean.shape[1],
                                                                                    mean.shape[1]).to(self.device)
        distribution = torch.distributions.multivariate_normal.MultivariateNormal(mean, Sigma)

        ll = torch.zeros((mean.shape[0])).to(self.device)
        for k in range(int(inputs.shape[1] / inputs.shape[1])):
            ll = ll + distribution.log_prob(inputs[:, inputs.shape[1] * k: inputs.shape[1] * (k + 1)])
        return ll.detach()'''