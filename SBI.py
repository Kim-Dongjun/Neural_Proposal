import torch
import sbi.utils as utils
from sbi.inference.base import infer
from sbi.inference import SNPE, SNLE, SNRE_B, SNRE_A, SMCABC, prepare_for_sbi
from sbi.inference.snpe import snpe_b
from sbi.utils.get_nn_models import posterior_nn, likelihood_nn, classifier_nn
import numpy as np
import argparse
import os
from pyro.distributions.empirical import Empirical

import sys
#import simulators_SBI as simulators
import simulations.simulators as simulators
import trueParameter as trueParameter
import diagnostics.logs_SBI as logs
import diagnostics.plots_SBI as plots

class SBI():
    def __init__(self, args):
        self.args = args
        if not os.path.exists(self.args.dir + '/'+str(self.args.simulation)):
            os.makedirs(self.args.dir + '/'+str(self.args.simulation))
        self.dir = self.args.dir + '/'+str(self.args.simulation)+'/' + str(self.args.algorithm) + '_' + self.args.simulation
        if self.args.logCount:
            num = np.random.randint(0, 100000000)
            self.dir = self.dir + '_replication_' + str(num)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        if self.args.algorithm == 'SNLE':
            algorithm = SNLE
        elif self.args.algorithm == 'SNPE':
            algorithm = SNPE
        elif self.args.algorithm == 'SNPE_B':
            algorithm = snpe_b
        elif self.args.algorithm == 'SNRE_A':
            algorithm = SNRE_A
        elif self.args.algorithm == 'SNRE_B':
            algorithm = SNRE_B
        elif self.args.algorithm == 'SMC':
            algorithm = SMCABC

        #torch.manual_seed(0)
        #np.random.seed(0)

        self.simulation()
        self.prior_()
        self.simulator, self.prior = prepare_for_sbi(simulator=self.sim.parallel_simulator, prior=self.prior)
        if self.args.algorithm == 'SNLE':
            self.density_estimator_build_fun = likelihood_nn(model=self.args.likelihoodFlowType,
                                                            hidden_features=self.args.likelihoodHiddenDim,
                                                            num_transforms=self.args.likelihoodNumBlocks,
                                                            device=self.args.device,
                                                            num_bins=self.args.likelihoodNumBin,
                                                            tail=self.args.nsfTailBound)
        elif self.args.algorithm[:4] == 'SNPE':
            self.density_estimator_build_fun = posterior_nn(model=self.args.likelihoodFlowType,
                                                            hidden_features=self.args.likelihoodHiddenDim,
                                                            num_transforms=self.args.likelihoodNumBlocks,
                                                            device=self.args.device,
                                                            num_bins=self.args.likelihoodNumBin,
                                                            tail=self.args.nsfTailBound)
        elif self.args.algorithm[:4] == 'SNRE':
            self.density_estimator_build_fun = classifier_nn(model='resnet',
                                                            hidden_features=self.args.likelihoodHiddenDim,
                                                            device=self.args.device)
        self.posteriors = []
        self.proposal = None
        self.mcmc_parameters = {'num_chains': self.args.numChains, 'warmup_steps': self.args.burnInMCMC}

        if self.args.algorithm[:4] == 'SNLE' or self.args.algorithm[:4] == 'SNPE':
            self.inference = algorithm(self.simulator, self.prior, density_estimator=self.density_estimator_build_fun, mcmc_method='slice', mcmc_parameters=self.mcmc_parameters,
                             show_progress_bars=False, device=self.args.device)
        elif self.args.algorithm[:4] == 'SNRE':
            self.inference = algorithm(self.simulator, self.prior, classifier=self.density_estimator_build_fun, mcmc_method='slice', mcmc_parameters=self.mcmc_parameters,
                             show_progress_bars=False, device=self.args.device)
        elif self.args.algorithm == 'SMC':
            self.inference = algorithm(self.simulator, self.prior, simulation_batch_size=self.args.simulation_budget_per_round)
        self.plot_ = plots.plotClass(args, self.dir, self.sim, self.mcmc_parameters)
        self.log_ = logs.logClass(args, self.dir, self.observation)

        if self.args.algorithm == 'SMC':
            self.posteriorLearning(0)
        else:
            for round in range(self.args.numRound):
                print("Learning Start")
                self.posteriorLearning(round)
                print("Plotting Start")
                self.plot(round)
                print("Logging Start")
                self.log(round)

    def posteriorLearning(self, round):
        if self.args.algorithm != 'SMC':
            self.posterior = self.inference(num_simulations=self.args.simulation_budget_per_round, proposal=self.proposal, validation_fraction=self.args.validationRatio, device=self.args.device)
            self.posteriors.append(self.posterior)
            self.proposal = self.posterior.set_default_x(self.observation)
        elif self.args.algorithm == 'SMC':
            if round == 0:
                self.posterior, self.summary = self.inference(x_o = self.observation, num_particles = self.args.simulation_budget_per_round, num_initial_pop=self.args.simulation_budget_per_round,
                                                num_simulations=self.args.simulation_budget_per_round * self.args.numRound, epsilon_decay=0.9, return_summary=True)
                #self.posterior = Empirical(self.summary['particles'][self.get_idx(round * self.args.simulation_budget_per_round, self.summary['budgets'])],
                #                           log_weights=self.summary['weights'][self.get_idx(round * self.args.simulation_budget_per_round, self.summary['budgets'])])
                self.posterior = Empirical(self.summary['particles'][-1],
                                           log_weights=self.summary['weights'][-1])
                self.posteriors.append(self.posterior)
                print("Plotting Start")
                self.plot(round)
                print("Logging Start")
                self.log(round)

    def prior_(self):
        self.prior = utils.BoxUniform(low=self.sim.min, high=self.sim.max)

    def simulation(self):
        self.sim = simulators.parallel_simulator(self.args)
        self.true_theta = trueParameter.true_parameter(self.args)
        print("true theta : ", self.true_theta)
        self.observation = self.sim.parallel_simulator(self.true_theta, True).detach().reshape(1,-1).to(self.args.device)
        print("observation : ", self.observation)

    def plot(self, round):
        self.plot_.drawing(round, self.posterior, self.true_theta, self.observation,
                           self.sim, self.prior)

    def log(self, round):
        self.log_.log(round, self.posterior, self.plot_)

    def get_idx(self, idx, list):
        for k in range(len(list)-1):
            if idx >= list[k] and idx < list[k+1]:
                return k

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--simulation", default='multiModals', help="Simulation model to calibrate")
    parser.add_argument("--thetaDim", type=int, default=2, help="Simulation Input dimension")
    parser.add_argument("--xDim", type=int, default=20, help="Simulation output dimension")
    parser.add_argument("--numModes", type=int, default=9, help="Number of posterior modes")
    parser.add_argument("--numTime", type=int, default=1000, help="Simulation execution timestep")
    parser.add_argument("--dt", type=float, default=0.1, help="Simulation discrete time interval")
    parser.add_argument("--summaryStatistics", type=str, default='raw+timeSeries', help='Types of summary statistics')

    parser.add_argument("--algorithm", type=str, default='SNLE')

    parser.add_argument("--validationRatio", type=float, default=0.1, help="Validation dataset ratio")
    parser.add_argument("--posteriorInferenceMethod", default='no',
                        help="Posterior estimation inference method among rkl/ipm/no")
    parser.add_argument("--posteriorFlowType", default='nsf', help="Flow model to use in posterior estimation")
    parser.add_argument("--posteriorParameterInitialize", default=True,
                        help="Whether to initialize posterior parameters")
    parser.add_argument("--posteriorHiddenDim", type=int, default=50, help="Posterior hidden layer dimension")
    parser.add_argument("--posteriorNumBlocks", type=int, default=3,
                        help="Number of blocks of flow model used in posterior estimation")
    parser.add_argument("--posteriorInputDim", type=int, default=100,
                        help="Input dimension of GAN model for posterior estimation")
    parser.add_argument("--posteriorNumBin", type=int, default=16, help="Number of bins for posterior network")
    parser.add_argument("--lrPosterior", type=float, default=5e-4, help="Learning rate of posterior estimation")
    parser.add_argument("--posteriorLearningDecay", default=False, help="Cosine Learning Rate Decay")
    parser.add_argument("--likelihoodFlowType", default='nsf', help="Flow model to use in likelihood estimation")
    parser.add_argument("--likelihoodHiddenDim", type=int, default=50, help="Likelihood hidden layer dimension")
    parser.add_argument("--likelihoodNumBlocks", type=int, default=3,
                        help="Number of blocks of flow model used in likelihood estimation")
    parser.add_argument("--likelihoodNumBin", type=int, default=16, help="Number of bins for likelihood network")
    parser.add_argument("--nsfTailBound", type=float, default=1.0, help="Neural Spline Flow tail bound")

    parser.add_argument("--samplerType", default='sbiSliceSampler', help="Simulation input sampler type")
    parser.add_argument("--numChains", type=int, default=25, help="Number of chains")

    parser.add_argument("--dir", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/ThirdArticleExperimentalResults',
                        help="Base directory")
    parser.add_argument("--logCount", default=False, help="Log directory")
    parser.add_argument("--log", default=True, help="Whether to log or not")
    parser.add_argument("--plotLikelihood", default=True, help="Whether to plot or not")
    parser.add_argument("--plotConditionalLikelihood", default=True, help="Whether to plot or not")
    parser.add_argument("--plotMMD", default=True, help="Whether to plot or not")
    parser.add_argument("--plotMIS", default=False, help="Whether to plot or not")
    parser.add_argument("--plotHistogram", default=False, help="Whether to plot histogram or not")
    parser.add_argument("--plotPerformance", default=False, help="Whether to plot histogram or not")
    parser.add_argument("--device", default='cpu', help="Device to update parameters")

    parser.add_argument("--samplerExploitationRatio", type=float, default=1., help="Ratio of how sampler exploits")
    parser.add_argument("--burnInMCMC", type=int, default=200, help="Number of burn-in periods for MCMC algorithm")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument("--numRound", type=int, default=50, help="Number of rounds")
    parser.add_argument("--simulation_budget_per_round", type=int, default=100, help="Number of simulations per round")
    parser.add_argument("--num_training", type=int, default=5000,
                        help="Number of training data for posterior estimation")

    args = parser.parse_args()
    args.logCount = args.logCount in ['True', True]
    args.log = args.log in ['True', True]
    args.plotLikelihood = args.plotLikelihood in ['True', True]
    args.plotConditionalLikelihood = args.plotConditionalLikelihood in ['True', True]
    args.plotMIS = args.plotMIS in ['True', True]
    args.plotMMD = args.plotMMD in ['True', True]
    args.plotHistogram = args.plotHistogram in ['True', True]
    args.plotPerformance = args.plotPerformance in ['True', True]

    for _ in range(10):
        sbi = SBI(args)