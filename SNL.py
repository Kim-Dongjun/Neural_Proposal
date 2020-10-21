import torch
import os
import argparse
import sys
import json
import numpy as np

import train.LikelihoodLearning as LikelihoodLearning
import train.PosteriorLearning as PosteriorLearning
import simulations.simulators as sim
import sample.Uniform as uniform
import trueParameter
import diagnostics.plots as plots
import diagnostics.logs as logs
import train.networks as networks
import train.train as train
import sample.sample as sample

class SNL_with_ISP():
    def __init__(self, args):
        self.args = args
        self.dir = self.args.dir + '/'+str(self.args.simulation)+'/SNL_with_ISP_' + '_' + self.args.likelihoodFlowType \
                    + '_' + self.args.samplerType + '_' + str(self.args.numChains)
        if self.args.logCount:
            num = np.random.randint(0,100000000)
            self.dir = self.dir + '_replication_' + str(num)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        #torch.manual_seed(0)
        #np.random.seed(0)

        # Data
        self.training_theta = None
        self.training_x = None
        self.validation_theta = None
        self.validation_x = None
        self.teacher_theta = None

        # Simulation Engine
        self.sim = sim.parallel_simulator(self.args)
        self.true_theta = trueParameter.true_parameter(self.args)
        self.observation = self.sim.parallel_simulator(self.true_theta, True).detach().to(self.args.device)

        # Networks and Optimizers
        self.netLikelihood, self.optLikelihood = networks.likelihoodNetwork(args)
        self.netPosterior, self.optPosterior = networks.posteriorNetwork(args, self.sim)

        # Prior
        self.prior = uniform.Prior(self.sim)

        # Plot and Logger
        self.plot_ = plots.plotClass(args, self.dir, self.sim, self.observation)
        self.log_ = logs.logClass(args, self.dir, self.netLikelihood, self.observation)

        # Main Loop
        for round in range(self.args.numRound):
            print("train start")
            self.train(round)
            print("plot start")
            self.plot(round)
            print("log start")
            self.log(round)

    def train(self, round):

        # Simulation Input Sampling
        self.thetas = sample.sample(self.args, self.args.simulation_budget_per_round, self.netLikelihood,
                                    self.netPosterior, self.observation, self.prior,
                                    self.sim, round == 0, self.args.posteriorInferenceMethod == 'no', self.args.numChains).detach().to(self.args.device)

        # Simulation Execution
        simulated_output = self.sim.parallel_simulator(self.thetas)
        print("simulated output : ", simulated_output.shape, self.thetas.shape)

        # Likelihood Learning
        self.training_theta, self.training_x, self.validation_theta, self.validation_x, self.netLikelihood = \
            LikelihoodLearning.LikelihoodLearning(args, round, self.thetas, simulated_output, self.training_theta,
                                              self.training_x, self.validation_theta, self.validation_x,
                                                  self.netLikelihood, self.optLikelihood)

        # Get Training Teacher Data for Implicit Surrogate Proposal (ISP) Learning
        if self.args.posteriorInferenceMethod != 'no':
            self.teacher_theta = sample.sample(self.args, self.args.num_training, self.netLikelihood,
                                               self.netPosterior, self.observation, self.prior, self.sim,
                                               round == -1, True, self.args.num_training)

        # Implicit Surrogate Proposal (ISP) Distribution Learning
        self.netPosterior = PosteriorLearning.PosteriorLearning(args, self.sim, self.teacher_theta)

    def plot(self, round):
        self.plot_.drawing(round, self.netLikelihood, self.netPosterior, self.true_theta,
                                             self.training_theta, self.teacher_theta, self.thetas, self.sim, self.prior)

    def log(self, round):
        self.log_.log(round, self.netLikelihood, self.plot_.order, self.plot_, self.netPosterior)

if __name__ == '__main__':
    # Choose simulations among
    # Shubert : simulation with 2-dimensional input / 2-dimensional output / 18-modes
    # SLCP-16 : simulation with 5-dimensional input / 50-dimensional output / 16-modes
    # SLCP-256 : simulation with 8-dimensional input / 40-dimensional output / 256-modes
    # mg1 : simulation with 3-dimensional input / 5-dimensional output / 1-modes
    # CLV : simulation with 8-dimensional input / 10-dimensional output / 2-modes

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--simulation", default='multiModals', help="Simulation model to calibrate")
    parser.add_argument("--thetaDim", type=int, default=5, help="Simulation Input dimension")
    parser.add_argument("--xDim", type=int, default=50, help="Simulation output dimension")
    parser.add_argument("--numModes", type=int, default=16, help="Number of posterior modes")
    parser.add_argument("--numTime", type=int, default=1000, help="Simulation execution timestep")
    parser.add_argument("--dt", type=float, default=0.1, help="Simulation discrete time interval")
    parser.add_argument("--summaryStatistics", type=str, default='quantiles', help='Types of summary statistics')

    parser.add_argument("--lrLikelihood", type=float, default=1e-3, help="Learning rate of likelihood estimation")
    parser.add_argument("--likelihoodLearningDecay", default=False, help="Cosine Learning Rate Decay")
    parser.add_argument("--lrPosterior", type=float, default=5e-4, help="Learning rate of posterior estimation")
    parser.add_argument("--posteriorLearningDecay", default=False, help="Cosine Learning Rate Decay")
    parser.add_argument("--maxValidationTolerance", type=int, default=20, help="Maximum epochs that validation loss does not minimized anymore")

    parser.add_argument("--dir", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/ISPExperimentalResults', help="Base directory")
    parser.add_argument("--logCount", default=False, help="Log directory")
    parser.add_argument("--log", default=True, help="Whether to log or not")
    parser.add_argument("--plotLikelihood", default=False, help="Whether to plot or not")
    parser.add_argument("--plotConditionalLikelihood", default=False, help="Whether to plot or not")
    parser.add_argument("--plotMIS", default=False, help="Whether to plot or not")
    parser.add_argument("--plotHistogram", default=False, help="Whether to plot histogram or not")
    parser.add_argument("--plotPerformance", default=False, help="Whether to plot histogram or not")
    parser.add_argument("--device", default='cuda:0', help="Device to update parameters")
    parser.add_argument("--validationRatio", type=float, default=0.1, help="Validation dataset ratio")

    parser.add_argument("--posteriorInferenceMethod", default='no', help="Posterior estimation inference method among rkl/no")
    parser.add_argument("--likelihoodFlowType", default='nsf', help="Flow model to use in likelihood estimation")
    parser.add_argument("--nsfTailBound", type=float, default=1.0, help="Neural Spline Flow tail bound")
    parser.add_argument("--samplerType", default='sbiSliceSampler', help="Simulation input sampler type")
    parser.add_argument("--posteriorFlowType", default='nsf', help="Flow model to use in posterior estimation")
    parser.add_argument("--likelihoodParameterInitialize", default=False, help="Whether to initialize likelihood parameters")
    parser.add_argument("--posteriorParameterInitialize", default=True, help="Whether to initialize posterior parameters")
    parser.add_argument("--likelihoodLambda", type=float, default=0., help="Regularizer hyperparameter")

    parser.add_argument("--likelihoodHiddenDim", type=int, default=50, help="Likelihood hidden layer dimension")
    parser.add_argument("--posteriorHiddenDim", type=int, default=50, help="Posterior hidden layer dimension")
    parser.add_argument("--likelihoodNumBlocks", type=int, default=3, help="Number of blocks of flow model used in likelihood estimation")
    parser.add_argument("--posteriorNumBlocks", type=int, default=3, help="Number of blocks of flow model used in posterior estimation")
    parser.add_argument("--likelihoodInputDim", type=int, default=100, help="Input dimension of GAN model for likelihood estimation")
    parser.add_argument("--posteriorInputDim", type=int, default=100, help="Input dimension of GAN model for posterior estimation")
    parser.add_argument("--likelihoodNumBin", type=int, default=16, help="Number of bins for likelihood network")
    parser.add_argument("--posteriorNumBin", type=int, default=16, help="Number of bins for posterior network")

    parser.add_argument("--iterLikelihoodBool", type=bool, default=True, help="Wheter to learn likelihood for certain iterations")
    parser.add_argument("--iterLikelihood", type=int, default=10000, help="Number of iterations for likelihood estimation learning")
    parser.add_argument("--iterPosterior", type=int, default=2000, help="Number of iterations for posterior estimation learning")
    parser.add_argument("--burnInMCMC", type=int, default=200, help="Number of burn-in periods for MCMC algorithm")
    parser.add_argument("--numChains", type=int, default=25, help="Number of chains")
    parser.add_argument("--samplerExploitationRatio", type=float, default=1., help="Ratio of how sampler exploits")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument("--numRound", type=int, default=50, help="Number of rounds")
    parser.add_argument("--simulation_budget_per_round", type=int, default=100, help="Number of simulations per round")
    parser.add_argument("--num_training", type=int, default=5000, help="Number of training data for posterior estimation")
    parser.add_argument("--thinning", type=int, default=10, help="thinning interval")

    args = parser.parse_args()
    args.logCount = args.logCount in ['True', True]
    args.log = args.log in ['True', True]
    args.plotLikelihood = args.plotLikelihood in ['True', True]
    args.plotConditionalLikelihood = args.plotConditionalLikelihood in ['True', True]
    args.plotMIS = args.plotMIS in ['True', True]
    args.plotHistogram = args.plotHistogram in ['True', True]
    args.plotPerformance = args.plotPerformance in ['True', True]
    args.posteriorParameterInitialize = args.posteriorParameterInitialize in ['True', True]
    args.iterLikelihoodBool = args.iterLikelihoodBool in ['True', True]
    args.likelihoodParameterInitialize = args.likelihoodParameterInitialize in ['True', True]
    args.likelihoodLearningDecay = args.likelihoodLearningDecay in ['True', True]

    for i in range(10):
        snl_with_isp = SNL_with_ISP(args)