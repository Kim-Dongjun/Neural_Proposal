import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import itertools
import os
import sys
import cv2
import sbi.utils.metrics as metrics
from sklearn.datasets import make_spd_matrix
from sklearn import mixture
import diagnostics.performanceCalculator as performanceCalculator
from scipy.stats import entropy

import sample.sample_SBI as sample


class plotClass():
    def __init__(self, args, dir, simulation, mcmc_parameters):
        self.args = args
        self.sim = simulation
        self.mcmc_parameters = mcmc_parameters

        self.thetaDomain = torch.Tensor([[simulation.min[0].item(), simulation.max[0].item()]])
        for i in range(1, self.args.thetaDim):
            self.thetaDomain = torch.cat(
                (self.thetaDomain, torch.Tensor([[simulation.min[i].item(), simulation.max[i].item()]])))
        self.thetaDomain = self.thetaDomain.cpu().detach().numpy()


        test_thetas = []
        self.num = 501
        self.lin = np.linspace(0, 1, self.num)
        self.base = np.linspace(0,1,self.num).reshape(-1,1)
        for j in range(self.num):
            for i in range(self.num):
                test_thetas.append([self.thetaDomain[0][0] + (self.thetaDomain[0][1] - self.thetaDomain[0][0]) * self.lin[i],
                                    self.thetaDomain[1][0] + (self.thetaDomain[1][1] - self.thetaDomain[1][0]) * self.lin[j]])
        self.test_thetas = torch.Tensor(test_thetas).to(args.device)
        self.xx, self.yy = np.meshgrid(self.thetaDomain[0][0] + (self.thetaDomain[0][1] - self.thetaDomain[0][0]) * self.lin,
                                       self.thetaDomain[1][0] + (self.thetaDomain[1][1] - self.thetaDomain[1][0]) * self.lin)
        self.x_for_fixed_theta = []
        for i in range(self.num):
            for j in range(self.num):
                self.x_for_fixed_theta.append([self.xx[i][j], self.yy[i][j]])
        self.x_for_fixed_theta = torch.Tensor(self.x_for_fixed_theta).to(args.device)
        self.x_for_fixed_theta = - 1.0 + 2.0 * self.x_for_fixed_theta
        self.fixed_theta = torch.Tensor([0.4, 0.4]).repeat(self.x_for_fixed_theta.shape[0], 1).to(args.device)
        self.dic = {'no': 'Multi-Chain MH Sampler', 'ipm': 'Implicit Sampler', 'rkl': 'Flow Sampler'}
        self.dir = dir

        self.mmd2 = 0
        self.wassersteinDistance = 0
        self.mis = 0
        self.inception_score = 0
        self.mis_posterior = 0
        self.tv = 0
        self.logPosterior = 0
        self.logPosterior_kde = 0
        self.logPosterior_sbi = 0
        self.mmdHistory = []
        self.wassersteinHistory = []
        self.isHistory = []
        self.misHistory = []
        self.misHistory_posterior = []
        self.tvHistory = []
        self.logPosteriors = []
        self.logPosteriors_kde = []
        self.logPosteriors_sbi = []

        if self.args.simulation == 'multiModals':
            self.t = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
            self.true_thetas = []
            for i in range(self.t.shape[0]):
                for j in range(self.t.shape[0]):
                    self.true_thetas.append([self.t[i], self.t[j]])
            self.true_thetas = np.array(self.true_thetas)
        elif self.args.simulation == 'competitiveLotkaVolterra_ver2':
            if self.args.thetaDim == 8:
                self.true_thetas = torch.Tensor(
                    [[1.52, 0., 0.44, 1.36, 2.33, 0., 1.21, 0.51], [0., 1.52, 1.36, 0.44, 1.21, 0.51, 2.33, 0.]]).to(
                    self.args.device)
            elif self.args.thetaDim == 3:
                self.true_thetas = torch.Tensor(
                    [[1.52, 0., 0.51]]).to(self.args.device)
            elif self.args.thetaDim == 4:
                self.true_thetas = torch.Tensor(
                    [[1.52, 0., 1.21, 0.51]]).to(self.args.device)

        elif self.args.simulation == 'shubert':
            if self.args.numModes == 16:
                self.true_thetas = torch.Tensor(
                    [[-7.090000152587891, -7.710000038146973], [-7.710000038146973, -7.090000152587891],
                     [-6.470000267028809, -7.090000152587891], [-7.090000152587891, -6.470000267028809],
                     [-0.8100000023841858, -7.710000038146973], [-1.4299999475479126, -7.090000152587891],
                     [-0.19000005722045898, -7.090000152587891], [-0.8100000023841858, -6.470000267028809],
                     [-0.8100000023841858, -1.440000057220459], [-1.440000057220459, -0.8100000023841858],
                     [-0.19000005722045898, -0.8100000023841858], [-0.8100000023841858, -0.19000005722045898],
                     [-7.090000152587891, -1.440000057220459], [-7.710000038146973, -0.8100000023841858],
                     [-6.470000267028809, -0.8100000023841858], [-7.090000152587891, -0.19000005722045898]]).to(
                    self.args.device)
            elif self.args.numModes == 18:
                self.true_thetas = torch.Tensor(
                    [[-7.090000152587891, -7.710000038146973], [-7.710000038146973, -7.090000152587891],
                     [-0.8100000023841858, -7.710000038146973], [-1.4299999475479126, -7.090000152587891],
                     [-0.8100000023841858, -1.440000057220459], [-1.440000057220459, -0.8100000023841858],
                     [-7.090000152587891, -1.440000057220459], [-7.710000038146973, -0.8100000023841858],
                     [4.849999904632568, -0.8100000023841858], [-7.090000152587891, 4.849999904632568],
                     [-0.8100000023841858, 4.849999904632568], [5.46999979019165, 4.849999904632568],
                     [-7.71999979019165, 5.46999979019165], [-1.4299999475479126, 5.46999979019165],
                     [4.849999904632568, 5.46999979019165], [5.46999979019165, -7.71999979019165],
                     [4.860000133514404, -7.099999904632568], [5.480000019073486, -1.440000057220459]]).to(
                    self.args.device)
        elif self.args.simulation == 'multiModals_ver2':
            pass
        elif self.args.simulation == 'crossRoad':
            self.true_thetas = torch.Tensor([[25., 47., 34., 29.], [34., 29., 25., 47.]]).to(self.args.device)
        elif self.args.simulation == 'complexPosterior':
            self.true_thetas = torch.Tensor(
                [[1.5,-2.0,-1.0,-0.9,0.6],[1.5,-2.0,-1.0,0.9,0.6],[1.5,-2.0,1.0,-0.9,0.6],[1.5,-2.0,1.0,0.9,0.6],
                                             [-1.5,-2.0,-1.0,-0.9,0.6],[-1.5,-2.0,-1.0,0.9,0.6],[-1.5,-2.0,1.0,-0.9,0.6],[-1.5,-2.0,1.0,0.9,0.6],
                                             [1.5,2.0,-1.0,-0.9,0.6],[1.5,2.0,-1.0,0.9,0.6],[1.5,2.0,1.0,-0.9,0.6],[1.5,2.0,1.0,0.9,0.6],
                                             [-1.5,2.0,-1.0,-0.9,0.6],[-1.5,2.0,-1.0,0.9,0.6],[-1.5,2.0,1.0,-0.9,0.6],[-1.5,2.0,1.0,0.9,0.6]]).to(self.args.device)

        elif self.args.simulation == 'complexPosterior_ver2':
            true_thetas = np.array([1.5,2.0,1.3,1.2,1.8,2.5,1.6,1.1])
            from itertools import chain, combinations

            def powerset(set):
                "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
                return chain.from_iterable(combinations(set, r) for r in range(len(set) + 1))
            fullList = np.arange(true_thetas.shape[0]).tolist()
            subsets = list(powerset(np.arange(true_thetas.shape[0])))
            for subset in subsets:
                subset = list(subset)
                theta = []
                for k in range(true_thetas.shape[0]):
                    if k in subset:
                        theta.append(true_thetas[k])
                    else:
                        theta.append(-true_thetas[k])
                if len(subset) == 0:
                    self.true_thetas = torch.Tensor(theta).reshape(1,-1)
                else:
                    self.true_thetas = torch.cat((self.true_thetas, torch.Tensor(theta).reshape(1,-1)))
            self.true_thetas = self.true_thetas.to(self.args.device)
            print("true thetas shape : ", self.true_thetas.shape)

        elif self.args.simulation == 'mg1':
            self.true_thetas = torch.Tensor([[1, 4, 0.15]]).to(self.args.device)

        self.thetaDomain = torch.Tensor([[simulation.min[0].item(), simulation.max[0].item()]])
        for i in range(1, self.args.thetaDim):
            self.thetaDomain = torch.cat((self.thetaDomain, torch.Tensor([[simulation.min[i].item(), simulation.max[i].item()]])))
        self.thetaDomain = self.thetaDomain.cpu().detach().numpy()
        self.bases = []
        for i in range(self.args.thetaDim):
            self.bases.append(np.linspace(simulation.min[i].item(), simulation.max[i].item(), self.num).reshape(-1,1))

    def plotLikelihood(self, round, netLikelihood, observation, test_thetas, algorithm):
        plt.close()
        if observation.get_device() == -1:
            device = 'cpu'
        else:
            device = self.args.device
        #estimatedLikelihood = torch.exp(
        #    netLikelihood.log_prob(self.test_thetas.to(device), observation)).detach()
        try:
            self.estimatedLikelihood = torch.exp(
                netLikelihood.log_prob(observation.to(device), test_thetas.to(device))).detach()
        except:
            self.estimatedLikelihood = torch.exp(
                netLikelihood.log_prob(test_thetas.to(device), observation.to(device))).detach()
        plt.figure(figsize=(6, 6))

        plt.contourf(self.xx, self.yy, self.estimatedLikelihood.cpu().numpy().reshape(self.num, self.num),
                     100, cmap='gray')
        # cmap=plt.cm.bone)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(self.dir + '/' + str(algorithm) + '_' + str(round) + '_gray.pdf')
        plt.close()
        plt.cla()
        plt.clf()

        plt.figure(figsize=(6, 6))

        plt.contourf(self.xx, self.yy, self.estimatedLikelihood.cpu().numpy().reshape(self.num, self.num),
                     100, cmap='binary')
        # cmap=plt.cm.bone)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(self.dir + '/' + str(algorithm) + '_' + str(round) + '_binary.pdf')
        plt.close()
        plt.cla()
        plt.clf()

    def plotHighDimensional(self, round, netLikelihood, observation, true_theta, upper=True):
        #fake = self.thetaDomain[:,0] + (self.thetaDomain[:,1] - self.thetaDomain[:,0]) * self.fake[:1000].cpu().detach().numpy()
        #if self.args.algorithm[:4] == 'SNLE':
        #    self.fake = sample.MHMultiChainsSampler(self.args, 1000, netLikelihood, observation.cpu(), simulator=self.sim).to(
        #        self.args.device)  # .cpu().detach().numpy()
        #elif self.args.algorithm[:4] == 'SNPE':
        #    self.fake = netLikelihood.sample((1000,), x=observation)
        #fake = self.fake[:1000].cpu().detach().numpy()
        plt.close()
        fig = plt.figure(figsize=(16,16))
        #print("true theta : ", true_theta)
        #true_theta = self.thetaDomain[:,0] + (self.thetaDomain[:,1] - self.thetaDomain[:,0]) * true_theta
        for i in range(self.args.thetaDim):
            for j in range(i, self.args.thetaDim) if upper else range(i + 1):

                ax = fig.add_subplot(self.args.thetaDim, self.args.thetaDim, i * self.args.thetaDim + j + 1)

                if i == j:
                    bandwidths = 10 ** np.linspace(-4, 1, 30)
                    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                                        {'bandwidth': bandwidths},
                                        cv=10)
                    grid.fit(self.teacher_theta[:1000, j].reshape(-1, 1))
                    kde = grid.best_estimator_
                    likelihood = np.exp(kde.score_samples(self.bases[j])).reshape(-1)
                    ax.plot(self.bases[j], likelihood)
                    ax.set_xlim(self.thetaDomain[j])
                    ax.set_ylim([0.0, ax.get_ylim()[1]])
                    ax.set_ylim([0.0, ax.get_ylim()[1]])
                    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
                    if i < self.args.thetaDim - 1 and not upper:
                        ax.tick_params(axis='x', which='both', labelbottom=False)
                    if self.true_thetas.shape[0] > 1:
                        for k in range(self.true_thetas.shape[0]):
                            ax.vlines(self.true_thetas[k][j].item(), 0, ax.get_ylim()[1], color='r')
                    else:
                        if true_theta is not None: ax.vlines(true_theta[0][j].item(), 0, ax.get_ylim()[1],
                                                             color='r')

                else:
                    ax.scatter(self.teacher_theta[:1000, j], self.teacher_theta[:1000, i], s=1, color='black')
                    ax.set_xlim(self.thetaDomain[j])
                    ax.set_ylim(self.thetaDomain[i])
                    if i < self.args.thetaDim - 1: ax.tick_params(axis='x', which='both', labelbottom=False)
                    if j > 0: ax.tick_params(axis='y', which='both', labelleft=False)
                    if j == self.args.thetaDim - 1: ax.tick_params(axis='y', which='both', labelright=True)
                    if self.true_thetas.shape[0] > 1:
                        for k in range(self.true_thetas.shape[0]):
                            ax.plot(self.true_thetas[k][j].item(), self.true_thetas[k][i].item(), 'r.', ms=8)
                    else:
                        if true_theta is not None: ax.plot(true_theta[0][j].item(), true_theta[0][i].item(), 'r.',
                                                           ms=8)
        plt.tight_layout()
        plt.savefig(self.dir + '/total_plot_iter_' + str(round) + '.pdf')

    def plotMMD(self, sim):
        """
        Finite sample estimate of square maximum mean discrepancy. Uses a gaussian kernel.
        :param xs: first sample
        :param ys: second sample
        :param wxs: weights for first sample, optional
        :param wys: weights for second sample, optional
        :param scale: kernel scale. If None, calculate it from data
        :return: squared mmd, scale if not given
        """

        self.fake = torch.Tensor(self.teacher_theta).cpu().detach()

        self.real = sample.sample(self.args, self.fake.shape[0], sim.simulator, None, self.observation, self.prior, self.sim, False, True, self.args.num_training).cpu().detach()

        self.mmd2 = metrics.unbiased_mmd_squared(self.real, self.fake)

        self.mmdHistory.append(self.mmd2.item())
        plt.close()
        plt.plot(np.arange(len(self.mmdHistory)), self.mmdHistory)
        plt.xlabel('Simulation Round')
        plt.ylabel('MMD')
        plt.savefig(self.dir + '/MMD.png')

        '''self.wassersteinDistance = cv2.EMD(self.real.cpu().detach().numpy(), self.fake.cpu().detach().numpy(), cv2.DIST_L2)[
            0]
        self.wassersteinHistory.append(self.wassersteinDistance)
        plt.close()
        plt.plot(np.arange(len(self.wassersteinHistory)), self.wassersteinHistory)
        plt.xlabel('Simulation Round')
        plt.ylabel('Wasserstein')
        plt.savefig(self.dir + '/Wasserstein.png')'''

    def get_hist_quantile(self, prob, n_trials, n_bins):
        """
        Calculates a given quantile of the height of a bin of a uniform histogram.
        :param prob: quantile probability
        :param n_trials: number of datapoints in the histogram
        :param n_bins: number of bins in the histogram
        :return: quantile
        """

        assert 0.0 <= prob <= 1.0

        k = 0
        while scipy.stats.binom.cdf(k, n_trials, 1.0 / n_bins) < prob:
            k += 1

        return k / float(n_trials)

    def pltHistogram(self, round, simulation, netLikelihood, prior, true_theta):
        if torch.sum(true_theta) != 0:
            n_trials = 300
            n_bins = 10
            self.order = torch.zeros((n_trials,self.args.thetaDim))
            l_quant = self.get_hist_quantile(0.005, n_trials, n_bins)
            u_quant = self.get_hist_quantile(0.995, n_trials, n_bins)
            centre = 1.0 / n_bins
            samples_from_prior = torch.Tensor(prior.gen(n_trials)).to(self.args.device).detach()
            simulated = simulation.parallel_simulator(samples_from_prior, False).detach()
            if self.args.algorithm[:4] == 'SNLE':
                samples = sample.MHMultiChainsSampler(self.args, n_trials * (n_bins - 1), netLikelihood, simulated, simulator=self.sim).detach()
            elif self.args.algorithm[:4] == 'SNPE':
                samples = netLikelihood.sample((n_trials * (n_bins - 1),), x=simulated)
            for k in range(n_trials):
                self.order[k] = sum(samples_from_prior[k] > samples)
            self.order = self.order.cpu().detach().numpy()
            for j in range(self.args.thetaDim):
                plt.close()
                plt.hist(self.order[:, j], bins=np.arange(n_bins + 1) - 0.5, density=True, color='r')
                plt.axhspan(l_quant, u_quant, facecolor='0.5', alpha=0.5)
                plt.axhline(centre, color='k', lw=2)
                plt.xlim([-0.5, n_bins - 0.5])
                plt.savefig(self.dir + '/histogram_round_'+str(round)+'_dim_'+str(j)+'.png')

            return self.order

    def plotMIS(self, netLikelihood, prior, observation):
        self.gmm = mixture.GaussianMixture(
            n_components=self.args.numModes, covariance_type='diag')
        self.gmm.fit(self.teacher_theta + 0.001 * np.random.randn(self.teacher_theta.shape[0], self.teacher_theta.shape[1]))
        self.pred_probs = self.gmm.predict_proba(self.teacher_theta)
        self.preds = self.gmm.predict(self.teacher_theta)

        scores = []
        # Calculating the inception score
        for i in range(self.true_thetas.shape[0]):
            part = self.pred_probs[self.preds == i] + 1e-6
            logp = np.log(part)
            selfi = np.sum(part * logp, axis=1)
            cross = np.mean(np.dot(part, np.transpose(logp)), axis=1)
            diff = selfi - cross
            kl = np.mean(diff)
            scores.append(kl)
        self.mis = np.exp(np.mean(scores))
        self.misHistory.append(self.mis)
        plt.close()
        plt.plot(np.arange(len(self.misHistory)), self.misHistory, label='Modified Inception Score')
        plt.legend()
        plt.savefig(self.dir + '/Modified_Inception_Score.png')

        scores = []
        # Calculating the inception score
        part = self.pred_probs + 1e-6
        py = np.mean(part, axis=0)
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        self.inception_score = np.exp(np.mean(scores))
        self.isHistory.append(self.inception_score)
        plt.close()
        plt.plot(np.arange(len(self.isHistory)), self.isHistory, label='Inception Score')
        plt.legend()
        plt.savefig(self.dir + '/Inception_Score.png')

    def plotTotalVariation(self, round, observation):
        if self.args.algorithm == 'SNLE':
            if round == 0:
                try:
                    self.true_likelihood = torch.exp(self.sim.simulator.log_prob(observation.reshape(1, -1), self.test_thetas)).cpu().detach().numpy()
                except:
                    self.true_likelihood = torch.exp(self.sim.simulator.log_prob(self.test_thetas, observation.reshape(1, -1))).cpu().detach().numpy()
            self.tv = np.sum(np.abs(self.estimatedLikelihood.cpu().detach().numpy() - self.true_likelihood)) / self.true_likelihood.shape[0]
            self.tvHistory.append(self.tv)
            plt.close()
            plt.plot(np.arange(len(self.tvHistory)), self.tvHistory)
            plt.savefig(self.dir + '/Total_Variation.png')

    def plotLogPosterior_SNPE(self, netPosterior, observation, title):
        self.logPosterior_sbi = - torch.sum(
            netPosterior.log_prob(self.true_thetas, observation).detach()).item()

        self.logPosteriors_sbi.append(self.logPosterior_sbi)
        plt.close()
        plt.plot(np.arange(len(self.logPosteriors_sbi)), self.logPosteriors_sbi)
        plt.xlabel('Number of Rounds')
        plt.ylabel('- log '+str(title)+' probability of true parameters ')
        plt.savefig(self.dir + '/log_probability_'+str(title)+'.png')

    def plotLogPosterior_SNLE(self):
        std = self.teacher_theta.shape[0] ** (-1.0 / (self.true_thetas.shape[1] + 4))
        self.logPosterior_sbi = -np.sum(performanceCalculator.gaussian_kde(self.teacher_theta, None, std).eval(self.true_thetas.cpu().detach().numpy()))
        self.logPosteriors_sbi.append(self.logPosterior_sbi)
        plt.close()
        plt.plot(np.arange(len(self.logPosteriors_sbi)), self.logPosteriors_sbi)
        plt.xlabel('Number of Rounds')
        plt.ylabel('- log snle probability of true parameters ')
        plt.savefig(self.dir + '/log_probability_snle.png')

    def plotPosteriorPerformance(self, netLikelihood, netPosterior):
        import PosteriorLearning
        netPosterior = PosteriorLearning.PosteriorLearning(self.args, self.sim, torch.Tensor(self.teacher_theta).to(self.args.device), netPosterior=None)
        self.logPosterior = - torch.sum(netPosterior.log_prob(self.true_thetas)).item()
        self.logPosteriors.append(self.logPosterior)
        plt.close()
        plt.plot(np.arange(len(self.logPosteriors)), self.logPosteriors)
        plt.xlabel('Number of Rounds')
        plt.ylabel('- log probability of true parameters')
        plt.savefig(self.dir + '/log_probability.png')

    def plotKernelDensityPosteriorPerformance(self):
        bandwidths = 10 ** np.linspace(-4, 0, 30)
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': bandwidths},
                            cv=10)
        grid.fit(self.teacher_theta)
        kde = grid.best_estimator_
        self.logPosterior_kde = - np.sum(kde.score_samples(self.true_thetas.cpu().detach().numpy()))
        self.logPosteriors_kde.append(self.logPosterior_kde)
        plt.close()
        plt.plot(np.arange(len(self.logPosteriors_kde)), self.logPosteriors_kde)
        plt.xlabel('Number of Rounds')
        plt.ylabel('- log kde probability of true parameters')
        plt.savefig(self.dir + '/log_probability_kde.png')

    def drawing(self, round, netLikelihood, true_theta, observation, simulation, prior):
        self.observation = observation
        self.prior = prior
        self.simulation = simulation
        self.round = round

        if self.args.algorithm[:4] == 'SNLE':
            self.teacher_theta = sample.sample(self.args, self.args.num_training, netLikelihood,
                                               None, self.observation, self.prior, self.simulation, self.round == -1, True, numChains=self.args.num_training, mcmc_parameters=self.mcmc_parameters).cpu().detach().numpy()
        elif self.args.algorithm[:4] == 'SNPE':
            self.teacher_theta = sample.sample(self.args, self.args.num_training, None, netLikelihood,
                                               self.observation, self.prior, self.simulation, self.round == -1, False,
                                               mcmc_parameters=self.mcmc_parameters).cpu().detach().numpy()
        elif self.args.algorithm[:4] == 'SNRE':
            self.teacher_theta = sample.sample(self.args, self.args.num_training, netLikelihood, netLikelihood,
                                               self.observation, self.prior, self.simulation, self.round == -1, True, mcmc_parameters=self.mcmc_parameters).cpu().detach().numpy()
        elif self.args.algorithm == 'SMC':
            self.teacher_theta = sample.sample(self.args, self.args.num_training, None, netLikelihood,
                                               self.observation, self.prior, self.simulation, self.round == -1, False).cpu().detach().numpy()

        if self.args.thetaDim >= 2:
            print("Marginal Distribution Plotting")
            self.plotHighDimensional(round, netLikelihood, observation, true_theta)

        if self.args.plotPerformance:
            print("Performance Plotting")
            if self.args.algorithm[:4] == 'SNLE':
                self.plotPosteriorPerformance(netLikelihood, None)
                self.plotKernelDensityPosteriorPerformance()
                self.plotLogPosterior_SNLE()
            elif self.args.algorithm[:4] == 'SNPE':
                self.plotPosteriorPerformance(None, netLikelihood)
                self.plotKernelDensityPosteriorPerformance()
                self.plotLogPosterior_SNPE(netLikelihood, observation, 'snpe')
            elif self.args.algorithm[:4] == 'SNRE':
                self.plotPosteriorPerformance(None, None)
                self.plotKernelDensityPosteriorPerformance()
                self.plotLogPosterior_SNPE(netLikelihood, observation, 'snre')
            elif self.args.algorithm == 'SMC':
                self.plotPosteriorPerformance(None, netLikelihood)
                self.plotKernelDensityPosteriorPerformance()
                self.plotLogPosterior_SNLE()

        if self.args.plotMIS:
            print("Inception Score Performance Plotting")
            self.plotMIS(netLikelihood, prior, observation)
            print("Maximum Mean Discrepancy Plotting")
            self.plotMMD(simulation)
            #self.plotTotalVariation(round)


        #self.plotHighDimensional(round, true_theta)
        if self.args.thetaDim == 2:
            if not self.args.algorithm == 'SMC':
                self.plotLikelihood(round, netLikelihood, observation, self.test_thetas, self.args.algorithm)
