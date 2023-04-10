import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import itertools
import pandas as pd

def make_perf_profile(scores, max_tau=100, step=0.01, labels=None, metric_name='runtime'):
    '''plot performance profiles of scores (a list of metric values for each considered algorithm);
    max_tau and step denote the scale and the stepsizes on the x axis'''
    
    n_solv = len(scores)
    n_probs = len(scores[0])

    taus = [x * step for x in range(int(1/step), int(max_tau*(1/step)))]




    if not labels:
        labels = ['']*n_solv

    perfs = []
    for i in range(n_solv):
        perfs.append({})

        for tau in taus:
            perfs[i][tau] = 0

    counted = 0
    for j in range(n_probs):
        ref = min([scores[i][j] for i in range(n_solv)])

        ref = max(ref, 1e-09)
        for tau in taus:
            for i in range(n_solv):
                if(scores[i][j]/ref <= tau and scores[i][j] < 300):
                    perfs[i][tau] += 1

    for tau in taus:
        for i in range(n_solv):
            perfs[i][tau] /= n_probs

    for i in range(n_solv):
        print('{} wins {} times'.format(labels[i], perfs[i][taus[0]]))



    f, ax = plt.subplots(1,1)
    ax.set_xlim([1,max_tau])
    ax.set_ylim([-0.05,1.05])

    plt.xscale('log')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    linestyles = ['--', '-.', ':']
    markers = ['^', 'o',  's', '+', 'x', 'd', 'v', '<', '>']
    markers_cycler = itertools.cycle(markers)
    linestyle_cycler = itertools.cycle(linestyles)

    ax.set_xlabel('performance ratio - {}'.format(metric_name), fontsize=13)
    ax.set_ylabel('fraction of problems', fontsize=13)


    for i in range(n_solv):
        lists = sorted(perfs[i].items()) # sorted by key, return a list of tuples
        x, y = zip(*lists,)

        ax.plot(x,y,next(markers_cycler), markersize=8,
                    markevery=0.15, linestyle=next(linestyle_cycler),label=labels[i])


    plt.legend(loc=4)
    f.savefig('pp_{}.pdf'.format(metric_name))



def make_fval_profile(scores, max_tau=100, step=0.01, labels=None, metric_name='f_val'):
    '''plot cumulative distribution of absolute distance from optimum for group of solvers
    on problems benchmark; scores scores contains a list of metric values for each considered algorithm;
    max_tau and step denote the scale and the stepsizes on the x axis'''
    
    n_solv = len(scores)
    n_probs = len(scores[0])

    taus = [x * step for x in range(int(max_tau*(1/step)))]




    if not labels:
        labels = ['']*n_solv

    perfs = []
    for i in range(n_solv):
        perfs.append({})

        for tau in taus:
            perfs[i][tau] = 0

    counted = 0
    for j in range(n_probs):
        ref = min([scores[i][j] for i in range(n_solv)])
        for tau in taus:
            for i in range(n_solv):
                if(np.abs(scores[i][j]-ref) <= tau):
                    perfs[i][tau] += 1

    for tau in taus:
        for i in range(n_solv):
            perfs[i][tau] /= n_probs

    for i in range(n_solv):
        print('{} wins {} times'.format(labels[i], perfs[i][taus[0]]))



    f, ax = plt.subplots(1,1)
    ax.set_xlim([0.05,max_tau])
    ax.set_ylim([-0.05,1.05])

    #plt.xscale('log')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(0, 10, 2))
    #ax.set_xticks([0.05, 0.1, 0.5, 1, 5, 10])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    linestyles = ['--', '-.', ':']
    markers = ['^', 'o',  's', '+', 'x', 'd', 'v', '<', '>']
    markers_cycler = itertools.cycle(markers)
    linestyle_cycler = itertools.cycle(linestyles)

    ax.set_xlabel('t', fontsize=13)
    ax.set_ylabel('p(abs_gap < t)', fontsize=13)


    for i in range(n_solv):
        if labels[i] == 'MIP':
            continue
        else:
            lists = sorted(perfs[i].items()) # sorted by key, return a list of tuples
            x, y = zip(*lists,)

            ax.plot(x,y,next(markers_cycler), markersize=8,
                    markevery=0.15, linestyle=next(linestyle_cycler),label=labels[i])


    plt.legend(loc=4)
    f.savefig('pp_{}.pdf'.format(metric_name))


metric_column_index = 4
data = pd.read_csv('tests/log_logistic_v0_trasp.csv', header=None)
runtimes_logistic_0 = data.to_numpy()[:, metric_column_index]

data = pd.read_csv('tests/log_logistic_trasp_ref_balanced_acc.csv', header=None)
runtimes_logistic_0_ref = data.to_numpy()[:, metric_column_index]

data = pd.read_csv('tests/log_oct.csv', header=None)
runtimes_oct = data.to_numpy()[:, metric_column_index]

#data = pd.read_csv('last_tests/log_cart.csv', header=None)
#runtimes_cart = data.to_numpy()[:, metric_column_index]

data = pd.read_csv('tests/log_logistic_v1_ref_new.csv', header=None)
runtimes_logistic_1_ref = data.to_numpy()[:, metric_column_index]

data = pd.read_csv('tests/log_margot_hfs.csv', header=None)
runtimes_l2 = data.to_numpy()[:, metric_column_index]

data = pd.read_csv('tests/log_margot_sfs.csv', header=None)
runtimes_sfs = data.to_numpy()[:, metric_column_index]


#runtimes = np.row_stack((runtimes_logistic_0, runtimes_logistic_0_ref, runtimes_logistic_1, runtimes_logistic_1_ref,   runtimes_l2, runtimes_sfs))
#print(np.count_nonzero(1 - runtimes_l2 <= 1- runtimes_logistic_0_ref) / 50)
runtimes = np.row_stack((runtimes_oct, runtimes_logistic_0, runtimes_l2))
#runtimes = np.row_stack((runtimes_logistic_0, runtimes_l2, runtimes_sfs))
print(runtimes)

#make_perf_profile(runtimes, labels=['OLCT', 'MARGOT', 'SFS-MARGOT'], metric_name='Sparsity', max_tau=32)
#make_fval_profile(runtimes, labels=['OLCT_V0', 'OLCT_V0_R', 'OLCT_V1', 'OLCT_V1_R', 'MARGOT', 'SFS-MARGOT'], metric_name='1 - BAcc', max_tau=5)
#make_perf_profile(runtimes, labels=['OLCT_V0', 'OLCT_V0_R', 'OLCT_V1', 'OLCT_V1_R', 'MARGOT', 'SFS-MARGOT'], metric_name='time', max_tau=32)
#make_fval_profile(runtimes, labels=['OCT', 'T-OLCT', 'HFS-MARGOT'], metric_name='1-Bacc', max_tau=10)
make_perf_profile(runtimes, labels=['OCT', 'T-OLCT', 'HFS-MARGOT'], metric_name='Time', max_tau=32)
#make_perf_profile(runtimes, labels=['T-OLCT', 'HFS-MARGOT'], metric_name='Time', max_tau=32)
#make_perf_profile(runtimes, labels=['T-OLCT', 'HFS-MARGOT'], metric_name='Time', max_tau=32)

