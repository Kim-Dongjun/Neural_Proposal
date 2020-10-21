import os
import csv
import sample
import torch
import numpy as np

class logClass():
    def __init__(self, args, dir, observation):
        self.args = args
        self.dir = dir
        self.observation = observation
        self.samples = []
        file = open(self.dir + '/experimental_setting.csv', 'w', newline='')
        writer = csv.writer(file)
        args_ = self.args.__dict__
        for key in list(args_.keys()):
            list_ = [key, args_[key]]
            writer.writerow(list_)
        file.close()
        self.true_parameter = []

        if args.simulation == 'twoMoons':
            pass
        elif args.simulation == 'fourModes':
            self.true_parameter_1 = np.array([0.2, 0.8])
            self.true_parameter = []
            for i in range(self.true_parameter_1.shape[0]):
                for j in range(self.true_parameter_1.shape[0]):
                    self.true_parameter.append([self.true_parameter_1[i], self.true_parameter_1[j]])
            self.true_parameter = torch.Tensor(self.true_parameter).to(args.device)
        elif args.simulation == 'multiModals':
            self.true_parameter_1 = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
            self.true_parameter = []
            for i in range(self.true_parameter_1.shape[0]):
                for j in range(self.true_parameter_1.shape[0]):
                    self.true_parameter.append([self.true_parameter_1[i], self.true_parameter_1[j]])
            self.true_parameter = torch.Tensor(self.true_parameter).to(args.device)

    def savePerformance(self, round, netLikelihood):
        if self.true_parameter != []:
            likelihood = netLikelihood.log_prob(self.true_parameter, self.observation).detach()
            performance = -likelihood.sum().item()
            if round == 0:
                file = open(self.dir + '/Performance.csv', 'w', newline='')
            elif round > 0:
                file = open(self.dir + '/Performance.csv', 'a', newline='')
            writer = csv.writer(file)
            writer.writerow([performance])
            file.close()

    def saveLikelihoodModel(self, round, netLikelihood):
        if self.args.algorithm[:4] == 'SNLE':
            torch.save(netLikelihood.net.state_dict(), self.dir + '/likelihood_'+str(round)+'.pth')
        elif self.args.algorithm[:4] == 'SNPE':
            torch.save(netLikelihood.net.state_dict(), self.dir + '/posterior_' + str(round) + '.pth')

    def saveSamplesFromEstimatedLikelihood(self, round, netLikelihood):
        file = open(self.dir + '/SamplesFromFinalEstimatedLikelihood_' + str(round) + '.csv' ,'w' ,newline='')
        writer = csv.writer(file)
        samples = sample.MHMultiChainsSampler(self.args, 1000, netLikelihood, self.observation, None).cpu().detach().numpy()
        writer.writerow(samples.reshape(-1))
        file.close()
        return samples

    def saveOrder(self, round, order):
        if not order == []:
            file = open(self.dir + '/Order_' + str(round) + '.csv', 'w', newline='')
            writer = csv.writer(file)
            order_ = np.transpose(order)
            for i in range(order_.shape[0]):
                writer.writerow(order_[i])
            file.close()

    def saveMMD(self, plot_):
        if os.path.exists(self.dir + '/MMD_performance.csv'):
            w = 'a'
        else:
            w = 'w'
        file = open(self.dir + '/MMD_performance.csv', w, newline='')
        writer = csv.writer(file)
        writer.writerow([plot_.mmd2])
        file.close()
        if os.path.exists(self.dir + '/Wassetstein_performance.csv'):
            w = 'a'
        else:
            w = 'w'
        file = open(self.dir + '/Wassetstein_performance.csv', w, newline='')
        writer = csv.writer(file)
        writer.writerow([plot_.wassersteinDistance])
        file.close()
        if os.path.exists(self.dir + '/Inception_Score_performance.csv'):
            w = 'a'
        else:
            w = 'w'
        file = open(self.dir + '/Inception_Score_performance.csv', w, newline='')
        writer = csv.writer(file)
        writer.writerow([plot_.inception_score])
        file.close()
        if os.path.exists(self.dir + '/Modified_Inception_Score_performance.csv'):
            w = 'a'
        else:
            w = 'w'
        file = open(self.dir + '/Modified_Inception_Score_performance.csv', w, newline='')
        writer = csv.writer(file)
        writer.writerow([plot_.mis])
        file.close()
        if os.path.exists(self.dir + '/Total_Variation_performance.csv'):
            w = 'a'
        else:
            w = 'w'
        file = open(self.dir + '/Total_Variation_performance.csv', w, newline='')
        writer = csv.writer(file)
        writer.writerow([plot_.tv])
        file.close()
        if os.path.exists(self.dir + '/Log_posterior_performance.csv'):
            w = 'a'
        else:
            w = 'w'
        file = open(self.dir + '/Log_posterior_performance.csv', w, newline='')
        writer = csv.writer(file)
        writer.writerow([plot_.logPosterior])
        file.close()
        if os.path.exists(self.dir + '/Log_kde_posterior_performance.csv'):
            w = 'a'
        else:
            w = 'w'
        file = open(self.dir + '/Log_kde_posterior_performance.csv', w, newline='')
        writer = csv.writer(file)
        writer.writerow([plot_.logPosterior_kde])
        file.close()
        if os.path.exists(self.dir + '/Log_sbi_posterior_performance.csv'):
            w = 'a'
        else:
            w = 'w'
        file = open(self.dir + '/Log_sbi_posterior_performance.csv', w, newline='')
        writer = csv.writer(file)
        writer.writerow([plot_.logPosterior_sbi])
        file.close()

    def log(self, round, netLikelihood, plot_):
        if self.args.log:
            self.saveLikelihoodModel(round, netLikelihood)
            #self.saveOrder(round, order)
            self.saveMMD(plot_)