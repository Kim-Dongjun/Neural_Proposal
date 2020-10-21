from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import pandas as pd
import sys

dir = 'D:\Research\ThirdArticleExperimentalResults\simulationDiscrepancy_complexPosterior/SNLE_B_rkl_nsf_no_nsf_MHGaussianMultiChainsSampler_1_replication_44677802/'
dir = 'D:\Research\ThirdArticleExperimentalResults\simulationDiscrepancy_complexPosterior/SNLE_B_rkl_nsf_no_nsf_MHGaussianMultiChainsSampler_1000_replication_16148403/'

base_dir = 'D:\Research\ThirdArticleExperimentalResults\simulationDiscrepancy_complexPosterior/'
dirlist = os.listdir(base_dir)
print(dirlist[0])
#dirlist = ['SNLE_B_rkl_nsf_no_nsf_MHGaussianMultiChainsSampler_1000_replication_16148403', 'SNLE_B_rkl_nsf_no_nsf_MHGaussianMultiChainsSampler_1_replication_48335303']

xDim = 50
cmap = plt.get_cmap('gist_ncar')
colors = [cmap(i) for i in np.linspace(0, 1, 16)]

true_thetas = torch.Tensor([[1.5,-2.0,-1.0,-0.9,0.6],[1.5,-2.0,-1.0,0.9,0.6],[1.5,-2.0,1.0,-0.9,0.6],[1.5,-2.0,1.0,0.9,0.6],
                 [-1.5,-2.0,-1.0,-0.9,0.6],[-1.5,-2.0,-1.0,0.9,0.6],[-1.5,-2.0,1.0,-0.9,0.6],[-1.5,-2.0,1.0,0.9,0.6],
                 [1.5,2.0,-1.0,-0.9,0.6],[1.5,2.0,-1.0,0.9,0.6],[1.5,2.0,1.0,-0.9,0.6],[1.5,2.0,1.0,0.9,0.6],
                 [-1.5,2.0,-1.0,-0.9,0.6],[-1.5,2.0,-1.0,0.9,0.6],[-1.5,2.0,1.0,-0.9,0.6],[-1.5,2.0,1.0,0.9,0.6]])

true_theta = torch.Tensor([[1.5,-2.0,-1.0,-0.9,0.6]])

norms_0 = {'APT': [], 'AALR': [], 'SNL': [], 'AALR with ISP': [], 'SNL with ISP': []}
norms_4 = {'APT': [], 'AALR': [], 'SNL': [], 'AALR with ISP': [], 'SNL with ISP': []}
norms_9 = {'APT': [], 'AALR': [], 'SNL': [], 'AALR with ISP': [], 'SNL with ISP': []}

norms = {'APT': [], 'AALR': [], 'SNL': [], 'AALR with ISP': [], 'SNL with ISP': []}

itr_1 = 0
itr_2 = 0

for kk in range(11):
    norms = {'APT': [], 'AALR': [], 'SNL': [], 'AALR with ISP': [], 'SNL with ISP': []}
    for round in range(10):
        for dir in dirlist:
            file = open(base_dir + dir + '/samples_'+str(round)+'.csv', 'r')
            samples = file.readlines()

            mean = true_theta[:, :2] ** 2
            diag1 = true_theta[:, 2].reshape(-1, 1) ** 2
            diag2 = true_theta[:, 3].reshape(-1, 1) ** 2
            corr = torch.tanh(true_theta[:, 4]).reshape(-1, 1)
            cov = torch.cat((diag1 ** 2, corr * diag1 * diag2, corr * diag1 * diag2, diag2 ** 2), 1).reshape(-1, 2, 2)
            distribution = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov + 1e-6 * torch.eye(2))
            observation = distribution.sample([int(xDim / mean.shape[1])]).transpose(1, 0).reshape(-1, xDim)

            thetas = []
            for k in range(len(samples)):
                thetas.append([float(x) for x in samples[k].split(',')])
            thetas = torch.Tensor(thetas)
            #print(dir, round)

            mean = thetas[:,:2] ** 2
            diag1 = thetas[:,2].reshape(-1,1) ** 2
            diag2 = thetas[:,3].reshape(-1,1) ** 2
            corr = torch.tanh(thetas[:,4]).reshape(-1,1)
            cov = torch.cat((diag1 ** 2, corr * diag1 * diag2, corr * diag1 * diag2, diag2 ** 2), 1).reshape(-1,2,2)
            distribution = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov + 1e-6 * torch.eye(2))
            simulated_output = distribution.sample([int(xDim/mean.shape[1])]).transpose(1,0).reshape(-1,xDim)

            norm = torch.norm(simulated_output - observation, dim=1).cpu().detach().numpy()

            dirS = dir.split('_')
            if dirS[0] == 'SNPE':
                norms['APT'].append(np.quantile(norm, 1./10 * np.arange(11))[kk])
                #norms['APT'].append(np.median(norm))
            else:
                if dirS[0] == 'SNLE':
                    if dir.split('_')[7] == '1':
                        norms['SNL'].append(np.quantile(norm, 1./10 * np.arange(11))[kk])
                        #norms['SNL'].append(np.median(norm))
                    elif dir.split('_')[7] == '1000':
                        norms['SNL with ISP'].append(np.quantile(norm, 1./10 * np.arange(11))[kk])
                        #norms['SNL with ISP'].append(np.median(norm))
                elif dirS[0] == 'SNRE':
                    if dir.split('_')[7] == '1':
                        norms['AALR'].append(np.quantile(norm, 1./10 * np.arange(11))[kk])
                        #norms['AALR'].append(np.median(norm))
                    elif dir.split('_')[7] == '1000':
                        norms['AALR with ISP'].append(np.quantile(norm, 1./10 * np.arange(11))[kk])
                        #norms['AALR with ISP'].append(np.median(norm))


            '''if dir.split('_')[7] == '1' and round == 0:
                norms_0['SNL'].append(norm.cpu().detach().numpy())
                #norms_0['SNL'].append(np.max(norm.cpu().detach().numpy().tolist()))
            elif dir.split('_')[7] == '1' and round == 4:
                norms_4['SNL'].append(norm.cpu().detach().numpy())
                #norms_4['SNL'].append(np.max(norm.cpu().detach().numpy().tolist()))
            elif dir.split('_')[7] == '1' and round == 9:
                norms_9['SNL'].append(norm.cpu().detach().numpy())
                #norms_9['SNL'].append(np.max(norm.cpu().detach().numpy().tolist()))
    
            elif dir.split('_')[7] == '1000' and round == 0:
                norms_0['SNL with ISP'].append(norm.cpu().detach().numpy())
                #norms_0['SNL with ISP'].append(np.max(norm.cpu().detach().numpy().tolist()))
            elif dir.split('_')[7] == '1000' and round == 4:
                norms_4['SNL with ISP'].append(norm.cpu().detach().numpy())
                #norms_4['SNL with ISP'].append(np.max(norm.cpu().detach().numpy().tolist()))
            elif dir.split('_')[7] == '1000' and round == 9:
                norms_9['SNL with ISP'].append(norm.cpu().detach().numpy())
                #norms_9['SNL with ISP'].append(np.max(norm.cpu().detach().numpy().tolist()))
    
            if dir.split('_')[7] == '1':
                itr_1 += 1
            elif dir.split('_')[7] == '1000':
                itr_2 += 1'''

    '''print(itr_1, itr_2)
    print(len(norms_0['SNL']), len(norms_0['SNL'][0]))
    #dictionary = {'Rounds':['0']*20+['4']*20+['9']*20,
    #              'SNL':(norms_0['SNL'][:20]+norms_4['SNL'][:20]+norms_9['SNL'][:20]),
    #              'SNL with ISP':(norms_0['SNL with ISP'][:20]+norms_4['SNL with ISP'][:20]+norms_9['SNL with ISP'][:20])}
    dictionary = {'Rounds':['0']*20000+['4']*20000+['9']*20000,
                  'SNL':(np.array(norms_0['SNL'])[:20].reshape(-1).tolist()+np.array(norms_4['SNL'])[:20].reshape(-1).tolist()+np.array(norms_9['SNL'])[:20].reshape(-1).tolist()),
                  'SNL with ISP':(np.array(norms_0['SNL with ISP'])[:20].reshape(-1).tolist()+np.array(norms_4['SNL with ISP'])[:20].reshape(-1).tolist()+np.array(norms_9['SNL with ISP'])[:20].reshape(-1).tolist())}
    df = pd.DataFrame(dictionary)
    
    df = pd.melt(df,id_vars=['Rounds'],value_vars=['SNL','SNL with ISP'],
                 var_name='zone', value_name='amount')
    
    import seaborn as sns
    plt.figure(figsize=(9,9))
    sns.boxplot(x='Rounds', y='amount', hue='zone', data=df, palette='Set1')
    plt.show()'''

    #print(norms['SNL'])
    SNL = np.array(norms['SNL']).reshape(-1,len(norms['SNL'])//10)
    SNL_with_ISP = np.array(norms['SNL with ISP']).reshape(-1,len(norms['SNL with ISP'])//10)
    AALR = np.array(norms['AALR']).reshape(-1,len(norms['AALR'])//10)
    AALR_with_ISP = np.array(norms['AALR with ISP']).reshape(-1,len(norms['AALR with ISP'])//10)
    APT = np.array(norms['APT']).reshape(-1,len(norms['APT'])//10)

    SNL_mean = np.mean(SNL, 1)
    SNL_std = np.std(SNL, 1)
    SNL_with_ISP_mean = np.mean(SNL_with_ISP, 1)
    SNL_with_ISP_std = np.std(SNL_with_ISP, 1)
    AALR_mean = np.mean(AALR, 1)
    AALR_std = np.std(AALR, 1)
    AALR_with_ISP_mean = np.mean(AALR_with_ISP, 1)
    AALR_with_ISP_std = np.std(AALR_with_ISP, 1)
    APT_mean = np.mean(APT, 1)
    APT_std = np.std(APT, 1)

    scale = 1
    times = 1000 * np.arange(SNL_mean.shape[0]) + 1000
    markers = ['o', 'v', '>', 's', 'd']

    plt.close()

    plt.figure(figsize=(6, 3))

    plt.semilogx(times, APT_mean, markers[0]+'-', label='APT', markeredgecolor='black',
         markeredgewidth=0.3)
    plt.fill_between(times, np.subtract(APT_mean, scale * APT_std), np.add(APT_mean, scale * APT_std), alpha=0.15, edgecolor='black')

    plt.semilogx(times, AALR_mean, markers[1]+'-', label='AALR', markeredgecolor='black',
         markeredgewidth=0.3)
    plt.fill_between(times, np.subtract(AALR_mean, scale * AALR_std), np.add(AALR_mean, scale * AALR_std), alpha=0.15, edgecolor='black')

    plt.semilogx(times, SNL_mean, markers[2]+'-', label='SNL', markeredgecolor='black',
         markeredgewidth=0.3)
    plt.fill_between(times, np.subtract(SNL_mean, scale * SNL_std), np.add(SNL_mean, scale * SNL_std), alpha=0.15, edgecolor='black')

    plt.semilogx(times, AALR_with_ISP_mean, markers[3]+'-', label='AALR with ISP', markeredgecolor='black',
         markeredgewidth=0.3)
    plt.fill_between(times, np.subtract(AALR_with_ISP_mean, scale * AALR_with_ISP_std), np.add(AALR_with_ISP_mean, scale * AALR_with_ISP_std), alpha=0.15, edgecolor='black')

    plt.semilogx(times, SNL_with_ISP_mean, markers[4]+'-', label='SNL with ISP', markeredgecolor='black',
         markeredgewidth=0.3)
    plt.fill_between(times, np.subtract(SNL_with_ISP_mean, scale * SNL_with_ISP_std), np.add(SNL_with_ISP_mean, scale * SNL_with_ISP_std), alpha=0.15, edgecolor='black')





    plt.xlabel('Number of Simulations (log scale)')
    plt.ylabel(str(int(kk * 10)) + '-th Percentile of Distance')

    #plt.plot(np.arange(SNL_mean.shape[0]), SNL_mean, label='SNL')
    #plt.fill_between(np.arange(SNL_mean.shape[0]), np.subtract(SNL_mean, scale * SNL_std), np.add(SNL_mean, scale * SNL_std), alpha=0.15)
    #plt.plot(np.arange(SNL_with_ISP_mean.shape[0]), SNL_with_ISP_mean, label='SNL with ISP')
    #plt.fill_between(np.arange(SNL_with_ISP_mean.shape[0]), np.subtract(SNL_with_ISP_mean, scale * SNL_with_ISP_std), np.add(SNL_with_ISP_mean, scale * SNL_with_ISP_std), alpha=0.15)
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig('D:\Research\논문\Sequential_Likelihood_Free_Inference_with_Surrogate_Proposal\AISTATS21\imgs/complexPosterior16_quantile_'+str(kk)+'_resize.pdf')