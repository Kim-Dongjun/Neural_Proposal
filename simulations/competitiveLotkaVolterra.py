# Written 26/5/17 by dh4gan
# Module containing functions for Lotka-Volterra systems
# (Predator-Prey models)

import numpy as np
import torch
import matplotlib.pyplot as plt

predlabel = 'Predator Count (Thousands)'
preylabel = 'Prey Count (Thousands)'
timelabel = 'Time'

predcolor = 'red'
preycolor = 'blue'


class Competitive_Lotka_Volterra(object):
    '''Sets up a simple Lotka_Volterra system'''

    def __init__(self, args, parameters, numTime, timestep, prey_capacity=100.0, predator_capacity=100.0):
        '''Create Lotka-Volterra system'''
        self.args = args
        self.numTime = numTime
        self.dt = timestep
        self.time = torch.zeros(self.numTime)
        #self.numSpecies = int(2 * np.sqrt(parameters.shape[1]//2))
        self.numSpecies = 4
        self.population = torch.zeros((parameters.shape[0], self.numSpecies, self.numTime)).to(self.args.device)
        self.reproduction = torch.Tensor([1.5] * self.numSpecies).repeat(parameters.shape[0],1).to(self.args.device)
        '''comp = torch.ones((parameters.shape[0], self.numSpecies // 2))
        competition = torch.diag_embed(comp)
        for k in range(parameters.shape[0]):
            competition[k][0][1] = 1.0
            competition[k][1][0] = 0.
        competition1 = torch.cat((competition, parameters[:,:(self.numSpecies//2) ** 2].reshape(-1,self.numSpecies//2,self.numSpecies//2)),2)
        for k in range(parameters.shape[0]):
            competition[k][0][1] = 0.35
            competition[k][1][0] = 0.35
        competition2 = torch.cat((parameters[:,(self.numSpecies//2)**2:].reshape(-1,self.numSpecies//2,self.numSpecies//2), competition),2)
        self.competition = torch.cat((competition1, competition2), 1)'''

        self.competition = torch.ones(parameters.shape[0], self.numSpecies).to(self.args.device)
        self.competition = torch.diag_embed(self.competition).to(self.args.device)
        for k in range(parameters.shape[0]):
            self.competition[k][0][1] = 1.09
            self.competition[k][0][2] = parameters[k][0].item()
            self.competition[k][0][3] = parameters[k][1].item()
            self.competition[k][1][0] = 0.0
            if parameters.shape[1] == 8:
                self.competition[k][1][2] = parameters[k][2].item()
                self.competition[k][1][3] = parameters[k][3].item()
                self.competition[k][2][0] = parameters[k][4].item()
                self.competition[k][2][1] = parameters[k][5].item()
                self.competition[k][2][3] = 0.35
                self.competition[k][3][2] = 0.35
                self.competition[k][3][0] = parameters[k][6].item()
                self.competition[k][3][1] = parameters[k][7].item()
            elif parameters.shape[1] == 2:
                self.competition[k][1][2] = 0.44
                self.competition[k][1][3] = 1.36
                self.competition[k][2][0] = 2.33
                self.competition[k][2][1] = 0.
                self.competition[k][2][3] = 0.35
                self.competition[k][3][2] = 0.35
                self.competition[k][3][0] = 1.21
                self.competition[k][3][1] = 0.51
            elif parameters.shape[1] == 3:
                self.competition[k][3][1] = parameters[k][2].item()
                self.competition[k][1][2] = 0.44
                self.competition[k][1][3] = 1.36
                self.competition[k][2][0] = 2.33
                self.competition[k][2][1] = 0.
                self.competition[k][2][3] = 0.35
                self.competition[k][3][2] = 0.35
                self.competition[k][3][0] = 1.21
            elif parameters.shape[1] == 4:
                self.competition[k][3][0] = parameters[k][2].item()
                self.competition[k][3][1] = parameters[k][3].item()
                self.competition[k][1][2] = 0.44
                self.competition[k][1][3] = 1.36
                self.competition[k][2][0] = 2.33
                self.competition[k][2][1] = 0.
                self.competition[k][2][3] = 0.35
                self.competition[k][3][2] = 0.35


        self.prey_capacity = prey_capacity
        self.predator_capacity = predator_capacity
        self.idx = ((self.numTime - 50) // self.args.xDim) * (np.arange(self.args.xDim) + 1) + 49

    def set_initial_conditions(self, population, tzero=0.0):
        '''set initial conditions'''
        self.population = population
        self.increment = torch.zeros_like(self.population).to(self.args.device)
        self.time[0] = tzero

    def integrate(self):
        '''integrate vanilla Lotka-Volterra system (simple Euler method)'''
        for t in range(self.numTime - 1):
            self.time[t + 1] = self.time[t] + self.dt
            #print("population : ", self.population)
            for species in range(self.population.shape[1]):
                self.increment[:,species,t] = self.reproduction[:, species] * self.dt * self.population[:, species, t] \
                                 * (1. - torch.sum(self.competition[:, species, :] * self.population[:, :, t], 1))
                self.population[:,species,t+1] = self.population[:,species,t] + self.increment[:,species,t]
        self.sum = torch.sum(self.population, 1)
        self.mean = self.sum[:,self.idx]
        self.Sigma = 1 * torch.eye((self.args.xDim)).repeat(self.population.shape[0], 1).reshape(self.population.shape[0], self.args.xDim, self.args.xDim).to(self.args.device)
        self.distribution = torch.distributions.multivariate_normal.MultivariateNormal(self.mean, self.Sigma)
        return self.mean
        return self.distribution.sample([1]).transpose(1,0).reshape(self.population.shape[0],-1)

    def log_prob(self, context='', inputs=''):
        numSpecies = 4
        initial_population = torch.Tensor([.5, .5, .5, .5]).repeat(self.args.numTime, 1).t().reshape(1, numSpecies,
                                                                                                     self.args.numTime).repeat(
            context.shape[0], 1, 1).to(self.args.device)
        self.set_initial_conditions(population=initial_population)

        for t in range(self.numTime - 1):
            self.time[t + 1] = self.time[t] + self.dt
            for species in range(self.population.shape[1]):
                self.increment[:,species,t] = self.reproduction[:, species] * self.dt * self.population[:, species, t] \
                                 * (1. - torch.sum(self.competition[:, species, :] * self.population[:, :, t], 1))
                self.population[:,species,t+1] = self.population[:,species,t] + self.increment[:,species,t]
        self.sum = torch.sum(self.population, 1)
        self.mean = self.sum[:, self.idx]
        self.Sigma = 0.001 * torch.eye((self.args.xDim)).repeat(self.population.shape[0], 1).reshape(
            self.population.shape[0], self.args.xDim, self.args.xDim).to(self.args.device)
        self.distribution = torch.distributions.multivariate_normal.MultivariateNormal(self.mean, self.Sigma)

        ll = torch.zeros((context.shape[0])).to(self.args.device)
        for k in range(int(inputs.shape[1] / 10)):
            ll = ll + self.distribution.log_prob(inputs[:, 10 * k: 10 * (k + 1)])
        return ll.detach()

if __name__ == '__main__':
    import scipy
    #parameters = torch.Tensor([[1.52, 0., 0.44, 1.36, 2.33, 0., 1.21, 0.51], [0., 1.52, 1.36, 0.44, 1.21, 0.51, 2.33, 0.]])
    #parameters = torch.Tensor(
    #    [[1.52, 0., 0.44, 1.36, 2.33, 0., 1.21, 0.51], [1.62, 0.1, 0.44, 1.36, 2.33, 0., 1.21, 0.51]])
    parameters = torch.Tensor([[1.52,0.,0.51],[1.1,0.4,0.]])
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--device", default='cpu', help="Device to update parameters")
    args = parser.parse_args()

    xDim = 10

    parameters = []
    lin = np.linspace(-0.2,2,101)
    xx, yy = np.meshgrid(lin, lin)
    for i in range(lin.shape[0]):
        for j in range(lin.shape[0]):
            parameters.append([xx[i][j], yy[i][j]])
    parameters = torch.Tensor(parameters)
    parameters = torch.Tensor([[1.52, 0., 0.51], [1.1, 0.4, 0.]])
    #parameters = torch.Tensor([[1.21, 0.51], [1.3, 0.5], [1.2, 0.6], [1.9, 0.2]])
    numTime = 1000
    timestep = 0.1
    #numSpecies = int(2 * np.sqrt(parameters.shape[1] // 2))
    numSpecies = 4

    true_parameter = torch.Tensor([[1.52, 0.]])
    clv = Competitive_Lotka_Volterra(args, true_parameter, numTime, timestep)
    b = torch.Tensor([.5, .5, .5, .5]).repeat(numTime, 1).t().reshape(1, numSpecies, numTime).repeat(
        true_parameter.shape[0], 1, 1)
    print(b)
    population = b * torch.ones((true_parameter.shape[0], numSpecies, numTime))
    clv.set_initial_conditions(population)
    population = clv.integrate()[:,:,50:]
    observation = torch.sum(population[:, :, :], 1).reshape(-1, population.shape[2])
    #observation = torch.Tensor(np.abs(scipy.fft(observation.cpu().detach().numpy()))[:, :10])
    idx = (observation.shape[1] // xDim) * (np.arange(xDim) + 1) - 1
    res = observation[:, idx]
    print(res)

    for p in range(true_parameter.shape[0]):
        for k in range(numSpecies):
            plt.plot(np.arange(observation.shape[1]), population[p][k][:], label='Species '+str(k))
            #plt.plot(np.arange(numTime), clv.increment[0][k][:], label='Species '+str(k))
        #plt.plot(np.arange(numTime), torch.sum(population[p,:2,:],0), label='Sum of Species 1')
        #plt.plot(np.arange(numTime), torch.sum(population[p,2:,:],0), label='Sum of Species 2')
        plt.plot(np.arange(observation.shape[1]), torch.sum(population[p,:,:],0), label='Sums')
        plt.legend()
        plt.show()

    clv = Competitive_Lotka_Volterra(args, parameters, numTime, timestep)
    #ratio = (clv.competition[0][0][0] - clv.competition[0][1][0]) / (clv.competition[0][0][1] - clv.competition[0][1][1])
    #a = clv.competition[0][0][0] - clv.competition[0][0][1] * ratio + 0.05
    b = torch.Tensor([.5, .5, .5, .5]).repeat(numTime, 1).t().reshape(1, numSpecies, numTime).repeat(parameters.shape[0],1,1)
    print(b)
    population = b * torch.ones((parameters.shape[0], numSpecies, numTime))
    clv.set_initial_conditions(population)
    population = clv.integrate()
    aggregated = torch.sum(population[:,:,:],1).reshape(-1,population.shape[2])
    import summaryStatistics as ss
    res = ss.autoregressiveSummaryStatistics(aggregated.reshape(-1, 1, aggregated.shape[1]), lag=xDim - 2,
                                             lagStart=0)
    print("res : ", res)

    for p in range(aggregated.shape[0]):
        #plt.close()
        #for k in range(numSpecies):
        #    plt.plot(np.arange(aggregated.shape[1]), population[p][k][:], label='Species '+str(k))
        plt.plot(np.arange(aggregated.shape[1]), torch.sum(population[p,:,:],0), label='Sums')
    plt.legend()
    plt.show()


    #res = torch.Tensor(np.abs(scipy.fft(aggregated.cpu().detach().numpy()))[:, :10])



    '''norms = torch.norm((res - observation) / observation, dim=1)
    print(res.shape, observation.shape, (res - observation).shape)
    print(norms)
    plt.contourf(xx, yy, torch.clamp(norms, max=torch.max(norms) * 0.5).reshape(lin.shape[0], lin.shape[0]), 100, cmap=plt.get_cmap('gist_ncar'))
    plt.colorbar()
    plt.show()'''


    #print(aggregated.cpu().detach().numpy().tolist())

    '''increment = clv.increment.reshape(parameters.shape[0], -1)
    xDim = 15
    result = torch.zeros((parameters.shape[0], xDim))
    for k in range(parameters.shape[0]):
        sign = torch.sign(increment[k])
        index = sign + torch.cat((torch.Tensor([0]), sign[:-1]))
        index_ = ((index == 0.).float().nonzero() % numTime).reshape(-1)
        # print("index : ", index_)
        temp = population.reshape(parameters.shape[0],-1)[k][index == 0.][index_.sort().indices]
        print(temp.shape)
        if temp.shape[0] >= xDim:
            result[k] = temp[:xDim]
        else:
            result[k] = torch.cat((temp, torch.zeros(xDim - temp.shape[0])))
    print("result : ", result)

    import summaryStatistics as ss
    summary = torch.Tensor(ss.autoregressiveSummaryStatistics(aggregated))#, self.observation_mean, self.observation_std))
    mean = torch.mean(summary)
    std = torch.std(summary)
    print("!! : ", summary)
    print((summary - mean) / std)
    group1 = population[:, :int(numSpecies / 2), :].reshape(-1,int(numSpecies / 2) * numTime)
    group2 = population[:, int(numSpecies / 2):, :].reshape(-1,int(numSpecies / 2) * numTime)
    import summaryStatistics as ss
    group1_ss = torch.Tensor(ss.plainSummaryStatistics(group1, 5))
    group2_ss = torch.Tensor(ss.plainSummaryStatistics(group2, 5))
    result = torch.cat((group1_ss, group2_ss), 1)
    print(result.cpu().detach().numpy())
    #print("increment : ", clv.increment)
    for p in range(parameters.shape[0]):
        for k in range(numSpecies):
            plt.plot(np.arange(numTime), population[p][k][:], label='Species '+str(k))
            #plt.plot(np.arange(numTime), clv.increment[0][k][:], label='Species '+str(k))
        #plt.plot(np.arange(numTime), torch.sum(population[p,:2,:],0), label='Sum of Species 1')
        #plt.plot(np.arange(numTime), torch.sum(population[p,2:,:],0), label='Sum of Species 2')
        plt.plot(np.arange(numTime-20), torch.sum(population[p,:,:],0)[20:], label='Sums')
        plt.legend()
        plt.show()

'''