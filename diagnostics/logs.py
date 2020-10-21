import os
import csv
import sample
import torch
import numpy as np

class logClass():
    def __init__(self, args, dir, netLikelihood, observation):
        self.args = args
        self.dir = dir
        self.netLikelihood = netLikelihood
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


    def saveLikelihoodModel(self, round, netLikelihood, netPosterior):
        torch.save(netLikelihood.state_dict(), self.dir + '/likelihood_'+str(round)+'.pth')
        if self.args.posteriorInferenceMethod != 'no':
            torch.save(netPosterior.state_dict(), self.dir + '/posterior_'+str(round)+'.pth')

    def saveOrder(self, round, order):
        if not order == []:
            file = open(self.dir + '/Order_' + str(round) + '.csv', 'w', newline='')
            writer = csv.writer(file)
            order_ = np.transpose(order)
            for i in range(order_.shape[0]):
                writer.writerow(order_[i])
            file.close()

    def saveMMD(self, plot):
        if os.path.exists(self.dir + '/MMD_performance.csv'):
            w = 'a'
        else:
            w = 'w'
        file = open(self.dir + '/MMD_performance.csv', w, newline='')
        writer = csv.writer(file)
        writer.writerow([plot.mmd2])
        file.close()
        if os.path.exists(self.dir + '/Wasserstein_performance.csv'):
            w = 'a'
        else:
            w = 'w'
        file = open(self.dir + '/Wasserstein_performance.csv', w, newline='')
        writer = csv.writer(file)
        writer.writerow([plot.wassersteinDistance])
        file.close()
        if os.path.exists(self.dir + '/Modified_Inception_Score_performance.csv'):
            w = 'a'
        else:
            w = 'w'
        file = open(self.dir + '/Modified_Inception_Score_performance.csv', w, newline='')
        writer = csv.writer(file)
        writer.writerow([plot.mis])
        file.close()
        if os.path.exists(self.dir + '/Inception_Score_performance.csv'):
            w = 'a'
        else:
            w = 'w'
        file = open(self.dir + '/Inception_Score_performance.csv', w, newline='')
        writer = csv.writer(file)
        writer.writerow([plot.inception_score])
        file.close()
        if os.path.exists(self.dir + '/Total_Variation_performance.csv'):
            w = 'a'
        else:
            w = 'w'
        file = open(self.dir + '/Total_Variation_performance.csv', w, newline='')
        writer = csv.writer(file)
        writer.writerow([plot.tv])
        file.close()
        if os.path.exists(self.dir + '/Log_posterior_performance.csv'):
            w = 'a'
        else:
            w = 'w'
        file = open(self.dir + '/Log_posterior_performance.csv', w, newline='')
        writer = csv.writer(file)
        writer.writerow([plot.logPosterior])
        file.close()
        if os.path.exists(self.dir + '/Log_kde_posterior_performance.csv'):
            w = 'a'
        else:
            w = 'w'
        file = open(self.dir + '/Log_kde_posterior_performance.csv', w, newline='')
        writer = csv.writer(file)
        writer.writerow([plot.logPosterior_kde])
        file.close()
        if os.path.exists(self.dir + '/Log_snle_posterior_performance.csv'):
            w = 'a'
        else:
            w = 'w'
        file = open(self.dir + '/Log_snle_posterior_performance.csv', w, newline='')
        writer = csv.writer(file)
        writer.writerow([plot.logPosterior_snle])
        file.close()

    def log(self, round, netLikelihood, order, plot, netPosterior):
        if self.args.log:
            self.saveLikelihoodModel(round, netLikelihood, netPosterior)
            self.saveOrder(round, order)
            self.saveMMD(plot)