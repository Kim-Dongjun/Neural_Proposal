import numpy as np
from torch.multiprocessing import Process, Pool
import time
import torch
import torch.distributions
import os
import scipy

import simulations.summaryStatistics as ss

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0  # initial time
    tf = time.time()  # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf - ti  # returns the time difference


TicToc = TicTocGenerator()  # create an instance of the TicTocGen generator


# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print("Elapsed time: %f seconds.\n" % tempTimeInterval)


def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

class parallel_simulator():
    def __init__(self, args):
        self.args = args
        self.burn_in = 0
        self.num = 0
        self.simulation = self.args.simulation
        if self.simulation == 'SLCP-16':
            self.min = torch.Tensor([-3., -3., -3., -3., -3.]).to(self.args.device)
            self.max = torch.Tensor([3., 3., 3., 3., 3.]).to(self.args.device)
        elif self.simulation == 'SLCP-256':
            self.min = torch.Tensor([-3., -3., -3., -3., -3., -3., -3., -3.]).to(self.args.device)
            self.max = torch.Tensor([3., 3., 3., 3., 3., 3., 3., 3.]).to(self.args.device)
        elif self.simulation == 'shubert':
            self.min = torch.Tensor([-10., -10.]).to(self.args.device)
            self.max = torch.Tensor([10., 10.]).to(self.args.device)
        elif self.simulation == 'mg1':
            self.min = torch.Tensor([0., 0., 0.]).to(self.args.device)
            self.max = torch.Tensor([10., 10., 1./3.]).to(self.args.device)
        elif self.simulation == 'CLV':
            if self.args.thetaDim == 2:
                self.min = torch.Tensor([-0.1, -0.1]).to(self.args.device)
                self.max = torch.Tensor([2., 2.]).to(self.args.device)
            elif self.args.thetaDim == 3:
                self.min = torch.Tensor([-0.1, -0.1, -0.1]).to(self.args.device)
                self.max = torch.Tensor([2., 2., 1.]).to(self.args.device)
            elif self.args.thetaDim == 4:
                self.min = torch.Tensor([-0.1, -0.1, 0., -0.1]).to(self.args.device)
                self.max = torch.Tensor([2., 2., 3., 1.]).to(self.args.device)
            elif self.args.thetaDim == 8:
                self.min = torch.Tensor([-0.1,-0.1,0.,0.,0.,-0.1,0.,-0.1]).to(self.args.device)
                self.max = torch.Tensor([2.,2.,2.,2.,3.,1.,3.,1.]).to(self.args.device)


    def parallel_simulator(self, thetas, validation=False):

        self.ss = ss.summaryStatisticsFunctions(self.args.device)

        if self.simulation == 'SLCP-16':
            import simulations.SLCP_16 as toy
            #self.thetas = self.paramMin + thetas * (self.paramMax - self.paramMin)
            self.simulator = toy.ToyModel(thetas)
            return self.simulator.executeSimulation(self.args.device, self.args.xDim, self.args.thetaDim)

        elif self.simulation == 'SLCP-256':
            import simulations.SLCP_256 as toy
            self.simulator = toy.ToyModel(thetas)
            return self.simulator.executeSimulation(self.args.device, self.args.xDim, self.args.thetaDim)

        elif self.simulation == 'shubert':
            if not validation:
                import simulations.ShubertModel as SM
                self.simulator = SM.Model(thetas)
                return self.simulator.executeSimulation(self.args.device, self.args.xDim, self.args.thetaDim)# / 300.
            elif validation:
                #return torch.Tensor([50.] * self.args.xDim) / 300.
                return torch.Tensor([-186.7309] * self.args.xDim)# / 300.

        elif self.simulation == 'mg1':
            import simulations.MG1Model as mg1
            if validation:
                thetas = thetas.repeat(100, 1)
            self.simulator = mg1.MG1Model(thetas, self.args.device, self.args.numTime)
            if not validation:
                return self.simulator.executeSimulation().to(self.args.device).detach()  # / self.observation
            elif validation:
                self.observation = torch.mean(self.simulator.executeSimulation().to(self.args.device).detach(), 0).reshape(1, -1)
                return self.observation  # / self.observation

        elif self.simulation == 'CLV':
            import simulations.competitiveLotkaVolterra as CLV
            self.simulator = CLV.Competitive_Lotka_Volterra(args=self.args, parameters=thetas, numTime=self.args.numTime, timestep=self.args.dt)
            numSpecies = 4
            initial_population = torch.Tensor([.5, .5, .5, .5]).repeat(self.args.numTime, 1).t().reshape(1, numSpecies, self.args.numTime).repeat(thetas.shape[0],1,1).to(self.args.device)
            self.simulator.set_initial_conditions(population=initial_population)
            return self.simulator.integrate()
            population = self.simulator.integrate()[:,:,50:]
            aggregated = torch.sum(population[:,:,:],1).reshape(-1,population.shape[2])
            if self.args.summaryStatistics == 'raw':
                idx = (aggregated.shape[1] // self.args.xDim) * (np.arange(self.args.xDim) + 1) - 1
                res = aggregated[:,idx]
            elif self.args.summaryStatistics == 'quantiles':
                res = torch.Tensor(np.transpose(np.quantile(aggregated.cpu().detach().numpy(), (1. / (self.args.xDim - 1)) * np.arange(self.args.xDim), axis=1)))
            elif self.args.summaryStatistics == 'fourier':
                res = torch.Tensor(np.abs(scipy.fft(aggregated.cpu().detach().numpy()))[:,1:self.args.xDim+1]).to(self.args.device)
            elif self.args.summaryStatistics == 'timeSeries':
                res = ss.autoregressiveSummaryStatistics(aggregated.reshape(-1,1,aggregated.shape[1]), lag=self.args.xDim - 2, lagStart=0)
            elif self.args.summaryStatistics == 'raw+fourier':
                idx = (aggregated.shape[1] // (self.args.xDim/2)) * (np.arange((self.args.xDim/2)) + 1) - 1
                res = torch.cat((aggregated[:, idx], torch.Tensor(np.abs(scipy.fft(aggregated.cpu().detach().numpy()))[:,1:self.args.xDim//2+1]).to(self.args.device)), 1)
            elif self.args.summaryStatistics == 'raw+timeSeries':
                idx = (aggregated.shape[1] // (self.args.xDim / 2)) * (np.arange((self.args.xDim / 2)) + 1) - 1
                res = torch.cat((aggregated[:, idx], ss.autoregressiveSummaryStatistics(aggregated.reshape(-1,1,aggregated.shape[1]), lag=self.args.xDim//2 - 2, lagStart=0)), 1)
            elif self.args.summaryStatistics == 'fourier+timeSeries':
                res = torch.cat((torch.Tensor(np.abs(scipy.fft(aggregated.cpu().detach().numpy()))[:,1:self.args.xDim//2+1]).to(self.args.device),
                                 ss.autoregressiveSummaryStatistics(aggregated.reshape(-1,1,aggregated.shape[1]), lag=self.args.xDim//2 - 2, lagStart=0)), 1)
            if validation:
                self.observation_mean = torch.mean(res)
                self.observation_std = torch.std(res)
            return (res - self.observation_mean) / self.observation_std