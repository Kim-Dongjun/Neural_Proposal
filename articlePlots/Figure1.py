import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import scipy.stats as stats
import torch
import math
from typing import Callable
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

#X, Y, Z = axes3d.get_test_data(0.2)
#print(Z)

x = np.linspace(0,1,101)

xx,yy = np.meshgrid(x,x)

theta = []
for i in range(x.shape[0]):
    for j in range(x.shape[0]):
        theta.append([xx[i][j],yy[i][j]])

numModes = 3

mus = [[1./4,2./4],[3./4,1./4],[3./4,3./4]]
sigmas = [[[0.01,0],[0,0.01]],[[0.01,0],[0,0.01]],[[0.01,0],[0,0.01]]]



#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------


import Uniform as uniform
class simulator_():
    def __init__(self):
        self.max = torch.Tensor([1,1])
        self.min = torch.Tensor([0,0])
simulator = simulator_()
prior = uniform.Prior(simulator)

num_chains = 100
thin = 10
n_sample = 100
proposal_std = 0.1
burnInMCMC = 100

gaussian1 = torch.distributions.multivariate_normal.MultivariateNormal(torch.Tensor(mus[0]), torch.Tensor(sigmas[0]))
gaussian2 = torch.distributions.multivariate_normal.MultivariateNormal(torch.Tensor(mus[1]), torch.Tensor(sigmas[1]))
gaussian3 = torch.distributions.multivariate_normal.MultivariateNormal(torch.Tensor(mus[2]), torch.Tensor(sigmas[2]))



class netLikelihood_:

    def __call__(
            self, gaussian1, gaussian2, gaussian3) -> Callable:
        self.gaussian1 = gaussian1
        self.gaussian2 = gaussian2
        self.gaussian3 = gaussian3

        return self.log_prob

    def log_prob(self, inputs='', context=''):
        return torch.exp(self.gaussian1.log_prob(context)) + torch.exp(self.gaussian2.log_prob(context)) + torch.exp(self.gaussian3.log_prob(context))

netLikelihood = netLikelihood_()(gaussian1, gaussian2, gaussian3)

thetas = torch.rand((num_chains, 2))
mcmc_samples = torch.Tensor([])
proposal_std = torch.Tensor([[proposal_std]]).repeat(num_chains, 2)
for itr in range(burnInMCMC + thin * math.ceil(n_sample / num_chains)):
    thetas_intermediate = thetas + proposal_std * torch.randn((num_chains, 2))
    rand = torch.rand(num_chains).reshape(-1)
    mask = (torch.exp(
        torch.min(netLikelihood(
            context=thetas_intermediate).reshape(-1) + torch.Tensor(
                        prior.eval(thetas_intermediate).reshape(-1))
                  - netLikelihood(context=thetas).reshape(-1) - torch.Tensor(
                        prior.eval(thetas).reshape(-1)),
                  torch.Tensor([0.] * num_chains))) > rand).float().reshape(-1, 1)
    if itr == 0:
        masks = mask.reshape(1, -1, 1)
    else:
        masks = torch.cat((masks, mask.reshape(1, -1, 1)))
    if itr % thin == 0:
        bool = (torch.sum(masks[-100:, :, :], 0) / 100 > 0.234).float()
        proposal_std = (1.1 * bool + 0.9 * (1 - bool)).repeat(1, 2) * proposal_std
    # print("proposal std : ", proposal_std)
    thetas = thetas_intermediate * mask + thetas * (1 - mask)
    if max(itr - burnInMCMC, 0) % thin == thin - 1:
        mcmc_samples = torch.cat((mcmc_samples, thetas))

print(mcmc_samples)


#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------








surrogate_samples = []


mcmc_samples = mcmc_samples.numpy()



gaussian1 = stats.multivariate_normal(mus[0], sigmas[0]).pdf(theta)
gaussian2 = stats.multivariate_normal(mus[1], sigmas[1]).pdf(theta)
gaussian3 = stats.multivariate_normal(mus[2], sigmas[2]).pdf(theta)

X = xx
Y = yy
Z = (gaussian1 + gaussian2 + gaussian3).reshape(x.shape[0], x.shape[0])

# Normalize to [0,1]
'''norm = plt.Normalize(Z.min(), Z.max())
#norm = plt.Normalize(-0.3, Z.max())
colors = cm.Greys(norm(Z))
rcount, ccount, _ = colors.shape

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(X, Y, np.abs(Z), rcount=rcount, ccount=ccount,
                       facecolors=colors, shade=False, alpha=0.2)
surf.set_facecolor((0,0,0,0))

#ax.scatter(mcmc_samples[:,0], mcmc_samples[:,1], [0] * mcmc_samples.shape[0], color='r', s=2)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

#ax.plot([mean_x,v[0]], [mean_y,v[1]], [mean_z,v[2]], color='red', alpha=0.8, lw=3)
#I will replace this line with:
colors = ['r'] * mcmc_samples.shape[0]
ax.scatter(mcmc_samples[0][0], mcmc_samples[0][1], [0], color=colors[0], s=4)
for i in range(1,mcmc_samples.shape[0]):
    ax.scatter(mcmc_samples[i][0], mcmc_samples[i][1], [0], color=colors[i], s=4)
    a = Arrow3D([mcmc_samples[i-1][0], mcmc_samples[i][0]], [mcmc_samples[i-1][1], mcmc_samples[i][1]],
                [0, 0], mutation_scale=10,
                lw=1, arrowstyle="-|>", color=colors[i])
    ax.add_artist(a)

plt.show()'''

plt.contourf(X, Y, Z, 100)

for i in range(mcmc_samples.shape[0]-1):
    plt.scatter(mcmc_samples[i][0], mcmc_samples[i][1], s=3, color='black')
    #plt.arrow(mcmc_samples[i][0], mcmc_samples[i][1], mcmc_samples[i+1][0] - mcmc_samples[i][0], mcmc_samples[i+1][1] - mcmc_samples[i][1], color='r')
plt.scatter(mcmc_samples[-1][0], mcmc_samples[-1][1], s=3, color='black')

plt.show()