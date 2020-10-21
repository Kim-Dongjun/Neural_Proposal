import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import itertools
import cv2
import sbi.utils.metrics as metrics
from sklearn.datasets import make_spd_matrix
from sklearn import mixture
from scipy.stats import entropy
import diagnostics.performanceCalculator as performanceCalculator
import train.converterNetLikelihoodToNetLikelihoodToEvidenceRatio as converter

import sample.sample as sample


class plotClass():
    def __init__(self, args, dir, simulation, observation):
        self.args = args
        self.sim = simulation
        self.observation = observation

        self.dic = {'no': 'Multi-Chain MH Sampler', 'rkl': 'Implicit Surrogate Proposal'}
        self.dir = dir

        self.mmd2 = 0
        self.wassersteinDistance = 0
        self.mis = 0
        self.inception_score = 0
        self.mis_posterior = 0
        self.tv = 0
        self.logPosterior = 0
        self.logPosterior_kde = 0
        self.logPosterior_snle = 0
        self.mmdHistory = []
        self.wassersteinHistory = []
        self.isHistory = []
        self.misHistory = []
        self.misHistory_posterior = []
        self.tvHistory = []
        self.logPosteriors = []
        self.logPosteriors_kde = []
        self.logPosteriors_snle =[]

        self.thetaDomain = torch.Tensor([[simulation.min[0].item(), simulation.max[0].item()]])
        for i in range(1, self.args.thetaDim):
            self.thetaDomain = torch.cat((self.thetaDomain, torch.Tensor([[simulation.min[i].item(), simulation.max[i].item()]])))
        self.thetaDomain = self.thetaDomain.cpu().detach().numpy()
        self.real = None
        self.fake = None
        self.true_likelihood = None

        test_thetas = []
        self.num = 501
        self.lin = np.linspace(0, 1, self.num)
        self.base = np.linspace(0,1,self.num).reshape(-1,1)
        for j in range(self.num):
            for i in range(self.num):
                test_thetas.append([self.thetaDomain[0][0] + (self.thetaDomain[0][1] - self.thetaDomain[0][0]) * self.lin[i],
                                    self.thetaDomain[1][0] + (self.thetaDomain[1][1] - self.thetaDomain[1][0]) * self.lin[j]])
        self.test_thetas = torch.Tensor(test_thetas).to(args.device).to(self.args.device)

        self.xx, self.yy = np.meshgrid(self.thetaDomain[0][0] + (self.thetaDomain[0][1] - self.thetaDomain[0][0]) * self.lin,
                                       self.thetaDomain[1][0] + (self.thetaDomain[1][1] - self.thetaDomain[1][0]) * self.lin)

        if self.args.simulation == 'SLCP-16':
            self.true_thetas = torch.Tensor(
                [[1.5, -2.0, -1.0, -0.9, 0.6], [1.5, -2.0, -1.0, 0.9, 0.6], [1.5, -2.0, 1.0, -0.9, 0.6],
                 [1.5, -2.0, 1.0, 0.9, 0.6],
                 [-1.5, -2.0, -1.0, -0.9, 0.6], [-1.5, -2.0, -1.0, 0.9, 0.6], [-1.5, -2.0, 1.0, -0.9, 0.6],
                 [-1.5, -2.0, 1.0, 0.9, 0.6],
                 [1.5, 2.0, -1.0, -0.9, 0.6], [1.5, 2.0, -1.0, 0.9, 0.6], [1.5, 2.0, 1.0, -0.9, 0.6],
                 [1.5, 2.0, 1.0, 0.9, 0.6],
                 [-1.5, 2.0, -1.0, -0.9, 0.6], [-1.5, 2.0, -1.0, 0.9, 0.6], [-1.5, 2.0, 1.0, -0.9, 0.6],
                 [-1.5, 2.0, 1.0, 0.9, 0.6]]).to(self.args.device)

        elif self.args.simulation == 'SLCP-256':
            true_thetas = np.array([1.5, 2.0, 1.3, 1.2, 1.8, 2.5, 1.6, 1.1])
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
                    self.true_thetas = torch.Tensor(theta).reshape(1, -1)
                else:
                    self.true_thetas = torch.cat((self.true_thetas, torch.Tensor(theta).reshape(1, -1)))
            self.true_thetas = self.true_thetas.to(self.args.device)
            print("true thetas shape : ", self.true_thetas.shape)

        elif self.args.simulation == 'shubert':
            if self.args.numModes == 16:
                self.true_thetas = torch.Tensor([[-7.090000152587891, -7.710000038146973], [-7.710000038146973, -7.090000152587891],
                                                 [-6.470000267028809, -7.090000152587891], [-7.090000152587891, -6.470000267028809],
                                                 [-0.8100000023841858, -7.710000038146973], [-1.4299999475479126, -7.090000152587891],
                                                 [-0.19000005722045898, -7.090000152587891],[-0.8100000023841858, -6.470000267028809],
                                                 [-0.8100000023841858, -1.440000057220459], [-1.440000057220459, -0.8100000023841858],
                                                 [-0.19000005722045898, -0.8100000023841858], [-0.8100000023841858, -0.19000005722045898],
                                                 [-7.090000152587891, -1.440000057220459], [-7.710000038146973, -0.8100000023841858],
                                                 [-6.470000267028809, -0.8100000023841858], [-7.090000152587891, -0.19000005722045898]]).to(self.args.device)
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
                     [4.860000133514404, -7.099999904632568], [5.480000019073486, -1.440000057220459]]).to(self.args.device)

        elif self.args.simulation == 'mg1':
            self.true_thetas = torch.Tensor([[1, 4, 0.2]]).to(self.args.device)

        elif self.args.simulation == 'CLV':
            if self.args.thetaDim == 8:
                self.true_thetas = torch.Tensor([[1.52, 0., 0.44, 1.36, 2.33, 0., 1.21, 0.51], [0., 1.52, 1.36, 0.44, 1.21, 0.51, 2.33, 0.]]).to(self.args.device)
            elif self.args.thetaDim == 3:
                self.true_thetas = torch.Tensor(
                    [[1.52, 0., 0.51]]).to(self.args.device)
            elif self.args.thetaDim == 4:
                self.true_thetas = torch.Tensor(
                    [[1.52, 0., 1.21, 0.51]]).to(self.args.device)

        self.bases = []
        for i in range(self.args.thetaDim):
            self.bases.append(np.linspace(simulation.min[i].item(), simulation.max[i].item(), self.num).reshape(-1,1))



    def plotLikelihood(self, round, netLikelihood, training_theta):
        plt.close()
        if self.observation.get_device() == -1:
            device = 'cpu'
        else:
            device = self.args.device
        if not self.SNLE:
            netLikelihood = converter.netRatio()(netLikelihood)
        try:
            self.estimatedLikelihood = torch.exp(
                netLikelihood.log_prob(x=self.observation[:self.args.xDim].repeat(self.test_thetas.shape[0], 1), theta=self.test_thetas)).detach()
        except:
            try:
                self.estimatedLikelihood = torch.exp(
                    netLikelihood.log_prob(context=self.test_thetas.to(device), inputs=self.observation[:self.args.xDim].repeat(self.test_thetas.shape[0], 1))).detach()
            except:
                self.estimatedLikelihood = torch.exp(
                    netLikelihood(context=self.test_thetas.to(device),
                    inputs=self.observation[:self.args.xDim].repeat(self.test_thetas.shape[0], 1))).detach()
        plt.figure(figsize=(6, 6))
        plt.contourf(self.xx, self.yy, self.estimatedLikelihood.cpu().numpy().reshape(self.num, self.num),
                     100, cmap='binary')

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(self.dir + '/likelihood_iter_binary_' + str(round) + '.pdf')
        plt.close()
        plt.figure(figsize=(6, 6))
        plt.contourf(self.xx, self.yy, self.estimatedLikelihood.cpu().numpy().reshape(self.num, self.num),
                     100, cmap='gray')

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(self.dir + '/likelihood_iter_gray_' + str(round) + '.pdf')
        plt.scatter(training_theta[-self.args.simulation_budget_per_round + int(
            self.args.validationRatio * self.args.simulation_budget_per_round)][0].item(),
                    training_theta[-self.args.simulation_budget_per_round + int(
                        self.args.validationRatio * self.args.simulation_budget_per_round)][1].item(), s=2, color='r',
                    label=self.dic[self.args.posteriorInferenceMethod])
        if self.args.simulation_budget_per_round < 300:
            plt.scatter(training_theta[-self.args.simulation_budget_per_round + int(self.args.validationRatio * self.args.simulation_budget_per_round):-1,0].cpu().detach(),
                        training_theta[-self.args.simulation_budget_per_round + int(self.args.validationRatio * self.args.simulation_budget_per_round):-1,1].cpu().detach(), s=2, color='r')
        else:
            plt.scatter(training_theta[-200:-1, 0].cpu().detach(),
                        training_theta[-200:-1, 1].cpu().detach(), s=2, color='r')
            plt.legend()
        plt.savefig(self.dir + '/sample_iter_' + str(round) + '.png')


    def plotTotalVariation(self, round):
        if round == 0:
            self.true_likelihood = torch.exp(self.sim.simulator.log_prob(self.observation.reshape(1, -1), self.test_thetas)).cpu().detach().numpy()
        self.tv = np.sum(np.abs(self.estimatedLikelihood.cpu().detach().numpy() - self.true_likelihood)) / self.true_likelihood.shape[0]
        self.tvHistory.append(self.tv)
        plt.close()
        plt.plot(np.arange(len(self.tvHistory)), self.tvHistory)
        plt.savefig(self.dir + '/Total_Variation.png')

    def plotSamples(self, round, netPosterior, teacher_theta):
        plt.close()
        teacher = teacher_theta[torch.randint(teacher_theta.shape[0], [1000])].cpu().detach()
        plt.scatter(teacher[0][0], teacher[0][1], s=2, color='r', label=r'Teacher $\theta$')
        plt.scatter(teacher[1:,0], teacher[1:,1], s=2, color='r')
        try:
            student = netPosterior.sampling(torch.randn([1000, self.args.thetaDim]).to(self.args.device)).cpu().detach()
        except:
            student = netPosterior.sample(1000).cpu().detach()
        plt.scatter(student[0][0], student[0][1], s=2, color='b', label=r'Student $\theta$')
        plt.scatter(student[1:,0], student[1:,1], s=2, color='b')
        plt.legend()
        plt.xlim(self.thetaDomain[0][0], self.thetaDomain[0][1])
        plt.ylim(self.thetaDomain[1][0], self.thetaDomain[1][1])
        plt.savefig(self.dir + '/samples_iter_' + str(round) + '.png')

    def plotPosterior(self, round, netPosterior):
        estimated_posterior = torch.exp(netPosterior.log_prob(self.test_thetas)).cpu().detach().numpy()
        plt.close()
        plt.contourf(self.xx, self.yy, estimated_posterior.reshape(self.num, self.num), 100)
        plt.colorbar()
        plt.savefig(self.dir + '/posterior'+str(round)+'.png')

    def plotHighDimensional(self, round, netLikelihood, prior, sim, true_theta, input_theta, title, upper=True):
        plt.close()
        fig = plt.figure(figsize=(16,16))
        for i in range(self.args.thetaDim):
            for j in range(i, self.args.thetaDim) if upper else range(i + 1):

                ax = fig.add_subplot(self.args.thetaDim, self.args.thetaDim, i * self.args.thetaDim + j + 1)

                if i == j:
                    bandwidths = 10 ** np.linspace(-4, 0, 30)
                    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                                        {'bandwidth': bandwidths},
                                        cv=10)
                    grid.fit(input_theta[:,j].reshape(-1,1))
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
                        if true_theta is not None: ax.vlines(true_theta[0][j].item(), 0, ax.get_ylim()[1], color='r')

                else:
                    ax.scatter(input_theta[:,j], input_theta[:,i], s=1, color='black')
                    ax.set_xlim(self.thetaDomain[j])
                    ax.set_ylim(self.thetaDomain[i])
                    if i < self.args.thetaDim - 1: ax.tick_params(axis='x', which='both', labelbottom=False)
                    if j > 0: ax.tick_params(axis='y', which='both', labelleft=False)
                    if j == self.args.thetaDim - 1: ax.tick_params(axis='y', which='both', labelright=True)
                    if self.true_thetas.shape[0] > 1:
                        for k in range(self.true_thetas.shape[0]):
                            ax.plot(self.true_thetas[k][j].item(), self.true_thetas[k][i].item(), 'r.', ms=8)
                    else:
                        if true_theta is not None: ax.plot(true_theta[0][j].item(), true_theta[0][i].item(), 'r.', ms=8)
        plt.tight_layout()
        plt.savefig(self.dir + '/samples_from_' + str(title) + '_round_' + str(round) + '.pdf')

    def pltHistogram_Yeto(self, round, simulation, netLikelihood, prior, true_theta):
        if torch.sum(true_theta) != 0:
            n_trials = 1000
            n_bins = 10
            order = torch.zeros((n_trials, self.args.thetaDim))
            l_quant = self.get_hist_quantile(0.005, n_trials, n_bins)
            u_quant = self.get_hist_quantile(0.995, n_trials, n_bins)
            centre = 1.0 / n_bins
            samples_from_prior = torch.Tensor(prior.gen(n_trials)).to(self.args.device).detach()
            for k in range(n_trials):
                simulated = simulation.parallel_simulator(samples_from_prior[k].reshape(1, -1), False).detach()
                samples = sample.MHMultiChainsSampler(self.args, n_bins - 1, netLikelihood, simulated.cpu(), prior, simulation).detach()
                for j in range(self.args.thetaDim):
                    order[k][j] = sum(samples_from_prior[k][j] > samples[:, j])
                print(str(k) + "-th trial over")
            order = order.cpu().detach().numpy()
            for j in range(self.args.thetaDim):
                plt.close()
                plt.hist(order[:, j], bins=np.arange(n_bins + 1) - 0.5, density=True, color='r')
                plt.axhspan(l_quant, u_quant, facecolor='0.5', alpha=0.5)
                plt.axhline(centre, color='k', lw=2)
                plt.xlim([-0.5, n_bins - 0.5])
                pmf = np.zeros(n_bins)
                for k in range(order.shape[0]):
                    pmf[int(order[k][j])] += 1.0 / n_trials
                print("order, pmf : ", order, pmf)
                plt.title('TV metric ' + str(np.sum(np.abs(pmf - centre))))
                plt.savefig(self.dir + '/histogram_round_' + str(round) + '_dim_' + str(j) + '.png')

            return order

    def f(self, a, b, diag=False):
        if diag:
            return torch.sum((a[None, ...] - b[:, None, :]) ** 2, dim=-1).reshape(-1)
        else:
            m, n = a.shape[0], b.shape[0]
            ix = torch.tril_indices(m, n, offset=-1)
            return torch.sum(
                (a[None, ...] - b[:, None, :]) ** 2, dim=-1, keepdim=False
            )[ix[0, :], ix[1, :]].reshape(-1)

    def plotMMD(self, round, netLikelihood, prior, sim, wxs=None, wys=None, scale=None, return_scale=False):
        """
        Finite sample estimate of square maximum mean discrepancy. Uses a gaussian kernel.
        :param xs: first sample
        :param ys: second sample
        :param wxs: weights for first sample, optional
        :param wys: weights for second sample, optional
        :param scale: kernel scale. If None, calculate it from data
        :return: squared mmd, scale if not given
        """
        if not sim.simulator.log_prob(context=torch.Tensor(self.teacher_theta[0]).to(self.args.device).reshape(1,-1), inputs=self.observation.reshape(1,-1)) == None:
            real = sample.sample(self.args, self.teacher_theta.shape[0], sim.simulator, None, self.observation, prior,
                                      sim, False, True, self.teacher_theta.shape[0], parallel=True)
            #real = sample.MHGaussianMultiChainsSampler(self.args, self.teacher_theta.shape[0], sim.simulator, self.observation,
            #                                           prior=prior, simulator=sim, num_chains=self.teacher_theta.shape[0])

            self.mmd2 = metrics.unbiased_mmd_squared(real.cpu().detach(), self.fake.cpu().detach())

            self.mmdHistory.append(self.mmd2.item())
            plt.close()
            plt.plot(np.arange(len(self.mmdHistory)), self.mmdHistory)
            plt.xlabel('Simulation Round')
            plt.ylabel('MMD')
            plt.savefig(self.dir + '/MMD.png')
            if round == 0:
                self.plotHighDimensional('True', netLikelihood, prior, sim, self.true_thetas, real.cpu().detach().numpy(), 'true_posterior')

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
            simulated = simulation.parallel_simulator(samples_from_prior, False).cpu().detach()
            samples = sample.MHMultiChainsSampler(self.args, n_trials * (n_bins - 1), netLikelihood, simulated, prior, simulation).detach()
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

    def plotMIS(self, round, netPosterior, netLikelihood, prior, simulation):
        self.gmm = mixture.GaussianMixture(
            n_components=self.args.numModes, covariance_type='diag')
        self.gmm.fit(self.teacher_theta)
        self.pred_probs = self.gmm.predict_proba(self.teacher_theta)
        self.preds = self.gmm.predict(self.teacher_theta)

        scores = []
        # Calculating the inception score
        for i in range(self.args.numModes):
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

    def plotLogPosterior_SNLE(self):
        std = self.teacher_theta.shape[0] ** (-1.0 / (self.true_thetas.shape[1] + 4))
        self.logPosterior_snle = -np.sum(performanceCalculator.gaussian_kde(self.teacher_theta, None, std).eval(self.true_thetas.cpu().detach().numpy()))
        self.logPosteriors_snle.append(self.logPosterior_snle)
        plt.close()
        plt.plot(np.arange(len(self.logPosteriors_snle)), self.logPosteriors_snle)
        plt.xlabel('Number of Rounds')
        plt.ylabel('- log snle probability of true parameters ')
        plt.savefig(self.dir + '/log_probability_snle.png')

    def plotPosteriorPerformance(self, round, netLikelihood, netPosterior):
        if self.args.posteriorInferenceMethod != 'no':
            self.logPosterior = - torch.sum(netPosterior.log_prob(self.true_thetas)).item()
        else:
            import train.PosteriorLearning as PosteriorLearning
            netPosterior = PosteriorLearning.PosteriorLearning(self.args, self.sim, torch.Tensor(self.teacher_theta).to(self.args.device), netPosterior=None)
            samples = netPosterior.sample(int(self.fake.shape[0])).cpu().detach().numpy()
            self.plotHighDimensional(round, netLikelihood, self.prior, self.simulation, self.true_thetas, samples, 'posterior')
            self.logPosterior = - torch.sum(netPosterior.log_prob(self.true_thetas)).item()
        self.logPosteriors.append(self.logPosterior)
        plt.close()
        plt.plot(np.arange(len(self.logPosteriors)), self.logPosteriors)
        plt.xlabel('Number of Rounds')
        plt.ylabel('- log probability of true parameters')
        plt.savefig(self.dir + '/log_probability.png')

    def plotKernelDensityPosteriorPerformance(self):
        bandwidths = 10 ** np.linspace(-4, 1, 30)
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

    def drawing(self, round, netLikelihood, netPosterior, true_theta, training_theta, teacher_theta, fake, simulation, prior, SNLE=True):
        self.fake = fake
        self.teacher_theta = teacher_theta
        self.prior = prior
        self.simulation = simulation
        self.round = round
        self.SNLE = SNLE
        self.order = []

        if self.args.thetaDim == 2:
            if self.args.plotConditionalLikelihood:
                self.plotLearning(round, netLikelihood)
            if self.args.posteriorInferenceMethod != 'no':
                self.plotSamples(round, netPosterior, teacher_theta)
                self.plotPosterior(round, netPosterior)
            if self.args.plotLikelihood:
                if self.args.likelihoodFlowType == 'nflow_maf':
                    self.observation = self.observation.cpu()
                print("Likelihood Plotting")
                self.plotLikelihood(round, netLikelihood, training_theta)

        if self.args.plotPerformance:
            if self.teacher_theta == None:
                self.teacher_theta = sample.sample(self.args, self.args.num_training, netLikelihood, netPosterior,
                                                   self.observation, self.prior, self.simulation, self.round == -1,
                                                   True, numChains=self.args.num_training, parallel=True,
                                                   SNLE=SNLE).cpu().detach().numpy()
            else:
                self.teacher_theta = self.teacher_theta.cpu().detach().numpy()

            print("Marginal Distribution Plotting")
            self.plotHighDimensional(round, netLikelihood, prior, simulation, true_theta, self.teacher_theta[:1000], 'parallelMCMC')
            print("Marginal Distribution Plotting")
            self.plotHighDimensional(round, netLikelihood, prior, simulation, true_theta, self.fake.cpu().detach().numpy(), 'MCMC')
            print("Performance Plotting")
            self.plotPosteriorPerformance(round, netLikelihood, netPosterior)
            self.plotKernelDensityPosteriorPerformance()
            self.plotLogPosterior_SNLE()

        if self.args.plotMIS:
            print("Inception Score Performance Plotting")
            self.plotMMD(round, netLikelihood, prior, simulation)
            self.plotMIS(round, netPosterior, netLikelihood, prior, simulation)
            #self.plotTotalVariation(round)