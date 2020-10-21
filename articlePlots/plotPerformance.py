import numpy as np
import matplotlib.pyplot as plt
import sys
import os

simulator = 'shubert'
leng = 10
#simulator = 'complexPosterior'
#leng = 13

base_dir = 'D:\Research\ThirdArticleExperimentalResults/200926_1_complexPosterior/'
base_dir = 'D:\Research\ThirdArticleExperimentalResults/200926_1_shubert/'
base_dir = 'D:\Research\ThirdArticleExperimentalResults/200926_1_complexPosterior_ver2/'
base_dir = 'D:\Research\ThirdArticleExperimentalResults/200929_complexPosterior/'
base_dir = 'D:\Research\ThirdArticleExperimentalResults/200929_shubert/'
#base_dir = 'D:\Research\ThirdArticleExperimentalResults/200929_complexPosterior_ver2/'
dirlist = os.listdir(base_dir)

filenames = ['Log_kde_posterior_performance',
             'Log_posterior_performance', 'Log_snle_posterior_performance', 'MMD_performance',
            'Inception_Score_performance']
#filenames = ['Modified_Inception_Score_performance', 'Inception_Score_performance']

what = ['SNLE', 'SNRE']

for filename in filenames:
    #performance = {'SNPE_C':[], 'SNLE_A_1':[], 'SNLE_A_5':[], 'SNLE_A_10':[], 'SNLE_A_100':[], 'SNLE_A_1000':[], 'SNLE_C':[]}
    performance = {}
    for dir in dirlist:
        dir_ = base_dir + dir
        try:
            file = open(dir_ + '/' + filename + '.csv', 'r')
        except:
            #pass
            file = open(dir_ + '/' + 'Log_sbi_posterior_performance' + '.csv', 'r')
        lines = file.readlines()
        try:
            performance_temp = [float(x) for x in lines]
        except:
            performance_temp = [float(x[7:13]) for x in lines]
        if len(performance_temp) >= leng:
            #print("!! : ", dir.split('_'))
            if dir.split('_')[0] in what:
                if dir.split('_')[4] == 'rkl':
                    #print(dir, performance_temp)
                    if not dir.split('_')[0] + '_C_' +str(dir.split('_')[6]) in list(performance.keys()):
                        performance[dir.split('_')[0] + '_C_'+str(dir.split('_')[6])] = [performance_temp[:leng]]
                    else:
                        performance[dir.split('_')[0] + '_C_'+str(dir.split('_')[6])].append(performance_temp[:leng])
                else:
                    #print(dir, performance_temp)
                    #if performance_temp[:leng][-1] < 1e3:
                    numChains = dir.split('_')[7]
                    if not dir.split('_')[0] + '_A_'+str(dir.split('_')[6])+'_'+str(numChains) in list(performance.keys()):
                        performance[dir.split('_')[0] + '_A_'+str(dir.split('_')[6])+'_'+str(numChains)] = [performance_temp[:leng]]
                    else:
                        performance[dir.split('_')[0] + '_A_'+str(dir.split('_')[6])+'_'+str(numChains)].append(performance_temp[:leng])
            elif dir.split('_')[0] == 'SNPE':
                #print(dir, performance_temp)
                if not 'SNPE_C' in list(performance.keys()):
                    performance['SNPE_C'] = [performance_temp]
                else:
                    performance['SNPE_C'].append(performance_temp)
            #elif dir.split('_')[0] == 'SNRE':
            #        if not 'SNRE_' + str(dir.split('_')[1]) in list(performance.keys()):
            #            performance['SNRE_' + str(dir.split('_')[1])] = [performance_temp]
            #        else:
            #            performance['SNRE_' + str(dir.split('_')[1])].append(performance_temp)


    plt.close()
    for key in list(performance.keys()):
        if filename == 'MMD_performance':
            #print("performance : ", performance[key])
            perf = np.log(np.array(performance[key]))
            #print(key, perf)
            #perf = np.array(performance[key])
        elif filename == 'Modified_Inception_Score_performance':
            perf = np.log(np.array(performance[key])) * 100
            #perf = np.array(performance[key])
        else:
            perf = np.array(performance[key])
        perf_aulc = np.sum(np.array(perf), 1)
        #print("performance : ", performance[key], perf_aulc)
        mean = np.mean(perf, 0)
        std = np.std(perf, 0)

        mean_aulc = np.mean(perf_aulc, 0)
        std_aulc = np.std(perf_aulc, 0)
        times = 1000 * np.arange(mean.shape[0]) + 1000
        plt.plot(times, mean, label=key)
        plt.fill_between(times, np.subtract(mean, std), np.add(mean, std), alpha=0.15)
        #if key == 'SNLE_C':
        #    plt.ylim(min(mean)-1, max(mean)+1)
        print(key, filename, mean[-1], std[-1])
        #print(key, filename, mean_aulc, std_aulc)
    plt.xlabel('Number of Simulations')
    name = ''
    for n in filename.split('_'):
        name += n+' '
    plt.ylabel(name[:-1])
    #if filename == 'Modified_Inception_Score_performance':
    #    plt.ylim(0,2)

    plt.legend()
    plt.show()
