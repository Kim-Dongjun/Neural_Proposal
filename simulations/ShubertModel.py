import matplotlib.pyplot as plt
import torch
import numpy as np

class Model():
    def __init__(self, thetas):
        self.thetas = thetas

    def executeSimulation(self, device, xDim, thetaDim):
        self.xDim = xDim
        self.device = device
        term1 = torch.zeros((5, self.thetas.shape[0])).to(device)
        term2 = torch.zeros((5, self.thetas.shape[0])).to(device)
        for k in range(1,6):
            term1[k-1] = k * torch.cos((k+1) * self.thetas[:,0] + k).to(device)
            term2[k-1] = k * torch.cos((k+1) * self.thetas[:,1] + k).to(device)
            #term2[k-1] = k * torch.cos((k+1) * (0.5 * self.thetas[:,1] + self.thetas[:,1] ** 2) + k)
        val = torch.sum(term1,0) * torch.sum(term2,0)
        thetas = val.reshape(-1,1).repeat(1,xDim)
        cov = 49. * torch.eye((xDim)).repeat(thetas.shape[0],1).reshape(thetas.shape[0],xDim,xDim).to(device)
        distribution = torch.distributions.multivariate_normal.MultivariateNormal(thetas, cov)
        return distribution.sample([1]).transpose(1,0).reshape(thetas.shape[0],-1)

    def log_prob(self, context='', inputs=''):
        #print("input : ", inputs)
        term1 = torch.zeros((5, context.shape[0])).to(self.device)
        term2 = torch.zeros((5, context.shape[0])).to(self.device)
        for k in range(1, 6):
            term1[k - 1] = k * torch.cos((k + 1) * context[:, 0] + k).to(self.device)
            term2[k - 1] = k * torch.cos((k + 1) * context[:, 1] + k).to(self.device)
            # term2[k-1] = k * torch.cos((k+1) * (0.5 * self.thetas[:,1] + self.thetas[:,1] ** 2) + k)
        val = torch.sum(term1, 0) * torch.sum(term2, 0)
        thetas = val.reshape(-1, 1).repeat(1, self.xDim)
        cov = 49. * torch.eye((self.xDim)).repeat(thetas.shape[0], 1).reshape(thetas.shape[0], self.xDim, self.xDim).to(self.device)
        distribution = torch.distributions.multivariate_normal.MultivariateNormal(thetas, cov)

        ll = torch.zeros((thetas.shape[0])).to(self.device)
        for k in range(int(inputs.shape[1]/thetas.shape[1])):
            #ll = ll + distribution.log_prob(300. * inputs[:,thetas.shape[1] * k: thetas.shape[1] * (k+1)])
            ll = ll + distribution.log_prob(inputs[:, thetas.shape[1] * k: thetas.shape[1] * (k + 1)])
        return ll.detach()

if __name__ == '__main__':
    observation = torch.Tensor([-186., -186.])
    device = 'cpu'
    num = 1001
    lin = -10. + 20. * np.linspace(0,1,num)
    xx, yy = np.meshgrid(lin, lin)
    thetas = []
    for i in range(lin.shape[0]):
        for j in range(lin.shape[0]):
            thetas.append([xx[i][j],yy[i][j]])
    thetas = torch.Tensor(thetas)
    term1 = torch.zeros((5, thetas.shape[0]))
    term2 = torch.zeros((5, thetas.shape[0]))
    for k in range(1,6):
        term1[int(k-1)] = k * torch.cos((k + 1) * thetas[:, 0] + k)
        term2[int(k - 1)] = k * torch.cos((k + 1) * thetas[:, 1] + k)
        #term2[int(k-1)] = k * torch.cos((k + 1) * (0.5 * thetas[:, 1] + thetas[:, 1] ** 2) + k)
    val = torch.sum(term1, 0) * torch.sum(term2, 0)
    print(torch.sum(term1, 0))
    #samples = []
    print(thetas[torch.argmax(val)], torch.min(val))
    #for k in range(val.shape[0]):
    #    if val[k] < -186:
    #        samples.append(thetas[k].cpu().detach().numpy().tolist())
    #print(samples)
    mean = val.reshape(-1,1).repeat(1,2)#.reshape(-1, 1).repeat(1, 2)
    cov = 49 * torch.eye((2)).repeat(thetas.shape[0],1).reshape(thetas.shape[0],2,2)
    distribution = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)
    ll = torch.exp(distribution.log_prob(observation))


    #print(res)
    #for k in range(res.shape[0]):
    #    if res[k].item() > 55 or res[k].item() < 45:
    #        res[k] = 0.
    #plt.contourf(xx, yy, res.reshape(num,num), 100)
    plt.figure(figsize=(6, 6))
    plt.contourf(xx, yy, ll.reshape(num, num), 100)
    #plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('D:\Research\논문\AISTATS21/temp/Shubert_true_posterior.pdf')
    plt.close()
    res = val
    plt.contourf(xx, yy, res.reshape(num, num), 100)
    plt.show()

    thetas = -10.0 + 20. * torch.linspace(0,1,1000).to(device)



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    thetas = -10.0 + 20. * torch.rand((1000,2)).to(device)
    model = MultiModeModel(thetas)
    xDim = 2
    thetaDim = 2
    output = model.executeSimulation(device, xDim, thetaDim).cpu().detach().numpy()
    observation = np.array([0.5]*20)
    rejection = []
    for i in range(output.shape[0]):
        if np.linalg.norm((output[i] - observation)) < 0.3:
            rejection.append(thetas[i].cpu())
    for i in range(len(rejection)):
        plt.scatter(rejection[i][0], rejection[i][1])
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()