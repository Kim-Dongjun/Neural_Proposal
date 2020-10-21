import torch
import numpy as np
from statsmodels.tsa.stattools import acf, pacf, acovf

class summaryStatisticsFunctions():
    def __init__(self, device):
        self.device = device

    def SummaryStatistics(self, sim_output):
        sumstats = torch.Tensor([])
        for sim in range(sim_output.shape[0]):
            temp = torch.Tensor([])
            corr = acf(sim_output[sim].cpu().detach().numpy(), nlags=2, fft=False)[1:]
            pcorr = pacf(sim_output[sim].cpu().detach().numpy(), nlags=2)[1:]
            if sim == 0:
                temp = torch.cat((temp, torch.Tensor(np.concatenate((corr ,pcorr))).reshape(1 ,-1)) ,dim=1)
            else:
                temp = torch.cat((temp, torch.Tensor(np.concatenate((corr, pcorr))).reshape(1, -1)), dim=1)
            if sim == 0:
                sumstats = torch.cat((sumstats ,temp.reshape(1 ,-1)) ,dim=1)
            else:
                sumstats = torch.cat((sumstats, temp.reshape(1, -1)), dim=0)
        return sumstats

    def GeneralSummaryStatistics(self, sim_output):
        sumstats = torch.Tensor([[]]).to(self.device)
        sumstats = torch.cat((sumstats, torch.mean(sim_output,axis=1).reshape(1,-1)), dim=1)
        var_ = torch.mean(sim_output[:, :] * sim_output[:, :], axis=1)

        def autocov(x, lag=1):
            C = torch.mean(x[:, lag:] * x[:, :-lag], axis=1) / var_
            return C.reshape(1,-1)

        for k in range(5):
            sumstats = torch.cat((sumstats, autocov(sim_output, lag=k+1)))

        return sumstats

def autoregressiveSummaryStatistics(data, lag=2, lagStart=0, observation_mean=None, observation_std=None):
    if type(data).__module__ == 'numpy':
        return autoregressiveSummaryStatistics_np(data, lag, lagStart, observation_mean, observation_std)
    elif type(data).__module__ == 'torch':
        return autoregressiveSummaryStatistics_torch(data, lag, lagStart, observation_mean, observation_std)

def autoregressiveSummaryStatistics_np(data, lag=2, lagStart=0, observation_mean = None, observation_std = None):
    means = np.mean(data, 2)
    var = np.std(data, 2) ** 2

    data = (data - np.transpose(np.tile(means, (1, data.shape[2])).reshape(data.shape[0], -1, data.shape[1]), (0,2,1))) \
           / np.sqrt(np.transpose(np.tile(var, (1, data.shape[2])).reshape(data.shape[0], -1, data.shape[1]), (0,2,1)))

    # autucorrelations
    autocovariance = (np.sum((data[:, :, :-1-lagStart] * data[:, :, 1+lagStart:]), 2) / (data.shape[2] - 1) - np.mean(data[:,:,:-1-lagStart],2) * np.mean(data[:,:,1+lagStart:],2)) /\
                     (np.std(data[:,:,:-1-lagStart]) * np.std(data[:,:,1+lagStart:]))
    for k in range(2, lag + 1):
        autocovariance = np.concatenate((autocovariance, (np.sum((data[:, :, :-k-lagStart] * data[:, :, k+lagStart:]), 2) / (data.shape[2] - 1)\
                          - np.mean(data[:,:,:-1-lagStart],2) * np.mean(data[:,:,1+lagStart:],2)) / (np.std(data[:,:,:-k-lagStart]) * np.std(data[:,:,k+lagStart:]))), 1)
    #for k in range(data.shape[0]):
    #    autocovariance = acovf(data[k][])

    res = np.concatenate((means, np.log(var + 1.), autocovariance), 1)

    # cross correlation coefficient
    if data.shape[1] == 2:
        cross = np.sum(data[:, 0, :] * data[:, 1, :], 1).reshape(-1, 1) / (data.shape[2] - 1)
        res = np.concatenate((res, cross), 1)

    if observation_mean != None:
        res -= observation_mean
        res /= observation_std

    return res

def autoregressiveSummaryStatistics_torch(data, lag=2, lagStart=0, observation_mean = None, observation_std = None):
    means = torch.mean(data, 2)
    var = torch.var(data, 2)
    data = (data - torch.transpose(means.repeat(1,data.shape[2]).reshape(data.shape[0],-1,data.shape[1]), 1, 2))\
           / torch.sqrt(torch.transpose(var.repeat(1,data.shape[2]).reshape(data.shape[0],-1,data.shape[1]), 1, 2))

    # autucorrelations
    autocovariance = torch.sum((data[:,:,:-1-lagStart] * data[:,:,1+lagStart:]), 2) / (data.shape[2] - 1)
    for k in range(2, lag + 1):
        autocovariance = torch.cat((autocovariance, torch.sum((data[:,:,:-k-lagStart] * data[:,:,k+lagStart:]), 2) / (data.shape[2] - 1)), 1)

    res = torch.cat((means, torch.log(var + 1.), autocovariance), 1)

    # cross correlation coefficient
    if data.shape[1] == 2:
        cross = torch.sum(data[:,0,:] * data[:,1,:], 1).reshape(-1,1) / (data.shape[2] - 1)
        res = torch.cat((res, cross), 1)

    if observation_mean != None:
        res -= observation_mean
        res /= observation_std
    return res

def plainSummaryStatistics(data, num):
    res = np.zeros((data.shape[0], num))
    for dim in range(data.shape[0]):
         res[dim] = np.quantile(data[dim], 1./(num-1) * np.arange(num), axis=0)
    return res

