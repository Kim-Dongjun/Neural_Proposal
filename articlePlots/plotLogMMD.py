import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.lines import Line2D
import copy

leng = 10

base_dir = 'D:\Research\ThirdArticleExperimentalResults/200929_complexPosterior/'
#base_dir = 'D:\Research\ThirdArticleExperimentalResults/200929_shubert/'
base_dir = 'D:\Research\ThirdArticleExperimentalResults/200929_complexPosterior_ver2/'
dirlist = os.listdir(base_dir)

filenames = ['MMD_performance']

what = ['SNLE', 'SNRE']

for filename in filenames:
    performance = {'APT':[], 'AALR':[], 'SNL':[], 'AALR with ISP':[], 'SNL with ISP':[]}
    #performance = {}
    for dir in dirlist:
        dir_ = base_dir + dir
        file = open(dir_ + '/' + filename + '.csv', 'r')
        lines = file.readlines()
        try:
            performance_temp = [float(x) for x in lines]
        except:
            performance_temp = [float(x[7:13]) for x in lines]
        if len(performance_temp) >= leng:
            #print("!! : ", dir.split('_'))
            if dir.split('_')[0] in what:
                if dir.split('_')[4] == 'rkl':
                    if dir.split('_')[0] == 'SNLE':
                        if not 'SNL with ISP' in list(performance.keys()):
                            performance['SNL with ISP'] = [performance_temp[:leng]]
                        else:
                            performance['SNL with ISP'].append(performance_temp[:leng])
                    elif dir.split('_')[0] == 'SNRE':
                        if not 'AALR with ISP' in list(performance.keys()):
                            performance['AALR with ISP'] = [performance_temp[:leng]]
                        else:
                            performance['AALR with ISP'].append(performance_temp[:leng])
                else:

                    numChains = dir.split('_')[7]
                    if dir.split('_')[0] == 'SNLE':
                        if numChains == '1' and dir.split('_')[6] == 'sbiSliceSampler':
                            if not 'SNL' in list(performance.keys()):
                                performance['SNL'] = [performance_temp[:leng]]
                            else:
                                performance['SNL'].append(performance_temp[:leng])
                    elif dir.split('_')[0] == 'SNRE':
                        if numChains == '1' and dir.split('_')[6] == 'MHGaussianMultiChainsSampler':
                            if not 'AALR' in list(performance.keys()):
                                performance['AALR'] = [performance_temp[:leng]]
                            else:
                                performance['AALR'].append(performance_temp[:leng])
            elif dir.split('_')[0] == 'SNPE':
                #print(dir, performance_temp)
                if not 'APT' in list(performance.keys()):
                    performance['APT'] = [performance_temp]
                else:
                    performance['APT'].append(performance_temp)


plt.close()
plt.figure(figsize=(6, 3))
markers = []
#for m in Line2D.markers:
#    markers.append(m)
markers = ['o', 'v', '>', 's', 'd']
itr = 0
for key in list(performance.keys()):
    perf = np.log(np.array(performance[key]))
    #perf = np.array(performance[key])
    mean = np.mean(perf, 0)
    std = np.std(perf, 0)
    times = 1000 * np.arange(mean.shape[0]) + 1000
    plt.semilogx(times, mean, markers[itr]+'-', label=key, markeredgecolor='black',
     markeredgewidth=0.3)
    plt.fill_between(times, np.subtract(mean, std), np.add(mean, std), alpha=0.15, edgecolor='black')
    print(key, filename, mean[-1], std[-1])
    itr += 1
plt.xlabel('Number of Simulations (log scale)')
name = ''
for n in filename.split('_'):
    name += n+' '
plt.ylabel('log Maximum Mean Discrepancy')
#if filename == 'Modified_Inception_Score_performance':
#plt.ylim(0,0.5)

plt.legend()
plt.tight_layout()
plt.show()
#plt.savefig('D:\Research\논문\Sequential_Likelihood_Free_Inference_with_Surrogate_Proposal\AISTATS21\imgs/complexPosterior256_MMD.pdf')
