import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

import os

import numpy as np 
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import torch
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.interpolate import splrep, BSpline
plt.matplotlib.style.use("https://gist.githubusercontent.com/JonnyCBB/c464d302fefce4722fe6cf5f461114ea/raw/64a78942d3f7b4b5054902f2cee84213eaff872f/matplotlibrc")
cool_colors = ['#00BEFF', '#D4CA3A', '#FF6DAE', '#67E1B5', '#EBACFA', '#9E9E9E', '#F1988E', '#5DB15A', '#E28544', '#52B8AA']
cool_colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

plt.rcParams.update({'font.size': 14})

search_to_color = {'BO': cool_colors[0], 'random': cool_colors[1], 'evolutionary': cool_colors[2], 'RF': cool_colors[3], 'RF (div)': cool_colors[5]}
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w
def plot_y_max(res,nb_iterations=100,axs=None,color=search_to_color['BO'],label='BO',operation= np.max,target_name='target',df_total=[],nb_initialisation=0):
    nb_iterations_range = np.arange(nb_iterations) + 1
    y_max_mu      = -10*np.ones(nb_iterations)
    y_max_sig_bot = -10*np.ones(nb_iterations)
    y_max_sig_top = -10*np.ones(nb_iterations)
    nb_runs = len(res)
    for i in range(1, nb_iterations+1):
        # max value acquired up to this point

        y_maxes = np.array([operation(res[r]['fitness_acquired'][nb_initialisation:nb_initialisation+i]) for r in range(nb_runs)]) # among runs
        assert np.size(y_maxes) == nb_runs
        y_max_mu[i-1]      = np.mean(y_maxes)
        y_max_sig_bot[i-1] = np.std(y_maxes[y_maxes < y_max_mu[i-1]])
        y_max_sig_top[i-1] = np.std(y_maxes[y_maxes > y_max_mu[i-1]])
    axs.plot(nb_iterations_range, y_max_mu, label=label, color=color)
    axs.fill_between(nb_iterations_range, y_max_mu  - y_max_sig_bot, 
                                            y_max_mu + y_max_sig_top, # 
                        alpha=0.2, ec="None", color=color)
    axs.set_xlabel('# evaluated oligomers')
    axs.set_ylabel('maximum fitness acquired up to iteration')
    axs.set_ylim([df_total[target_name].min(),df_total[target_name].max()])

    axs.axhline(y=np.max(df_total[target_name].values), color="k", linestyle="--", zorder=0)
    return y_max_mu, y_max_sig_bot, y_max_sig_top
def plot_y_mean(res,nb_iterations=100,axs=None,color=search_to_color['BO'],label='BO',target_name='target',df_total=[],nb_initialisation=0):
    nb_iterations_range = np.arange(nb_iterations) + 1
    y_mean_mu_BO      = -10*np.ones(nb_iterations)
    nb_runs = len(res)
    for i in range(1, nb_iterations+1):
        # max value acquired up to this point
        y_maxes = np.array([res[r]['fitness_acquired'][nb_initialisation+i-1] for r in range(nb_runs) if len(res[r]['fitness_acquired'])>nb_initialisation+i-1]) # among runs
        if len(y_maxes) == 0:
            break
        y_mean_mu_BO[i-1]      = np.mean(y_maxes)
    y_mean_mov_av = moving_average(np.array(y_mean_mu_BO), 5)
        

    axs.plot(nb_iterations_range, y_mean_mov_av, label=label, color=color)

    axs.set_xlabel('# evaluated oligomers')
    axs.set_ylabel('mean fitness acquired at iteration')
    axs.set_ylim([df_total[target_name].min(),df_total[target_name].max()])
    axs.axhline(y=np.max(df_total[target_name].values), color="k", linestyle="--", zorder=0)    
    return y_mean_mu_BO
def plot_element_above_min(res,min_target,nb_iterations=100, topKmol = 1000,axs=None,color=search_to_color['BO'],label='BO',df_total=[],nb_initialisation=0):
    nb_iterations_range = np.arange(nb_iterations) + 1
    y_elm      = -10*np.ones(nb_iterations)
    y_elm_sig_bot = -10*np.ones(nb_iterations)
    y_elm_sig_top = -10*np.ones(nb_iterations)
    nb_runs = len(res)
    for i in range(1, nb_iterations+1):
        # max value acquired up to this point

        y_maxes =np.array([np.array(res[r]['fitness_acquired'][nb_initialisation:nb_initialisation+i])>min_target for r in range(nb_runs) if len(res[r]['fitness_acquired'])>nb_initialisation+i-1]).sum(axis=1) #/topKmol*100 # among runs
        y_elm[i-1]      = np.mean(y_maxes) 
        y_elm_sig_bot[i-1] = np.std(y_maxes[y_maxes < y_elm[i-1]])
        y_elm_sig_top[i-1] = np.std(y_maxes[y_maxes > y_elm[i-1]])
    axs.plot(nb_iterations_range, y_elm, label=label, color=color)
    axs.fill_between(nb_iterations_range, y_elm  - y_elm_sig_bot, 
                                            y_elm + y_elm_sig_top, # 
                        alpha=0.2, ec="None", color=color)
    axs.set_xlabel('# evaluated oligomers')
    axs.set_ylabel(f'Top {topKmol/df_total.shape[0]*100:1.2f}% of oligomers ({topKmol} molecules) ')
    return y_elm, y_elm_sig_bot, y_elm_sig_top

def plot_hist_mol_found(search_results,target_name,df_total,num_elem_initialisation=100,axs=None,color=search_to_color['BO']):
    INchikeys_found = []
    for search_result in search_results:
        INchikeys_found.append(search_result['InchiKey_acquired'][num_elem_initialisation:])
    INchikeys_found = np.concatenate(INchikeys_found)
    df_total_found = df_total[df_total['InChIKey'].isin(INchikeys_found)]
    print('mol_found',df_total_found.shape[0])   
    df_total_found[target_name].hist(ax=axs, bins=20, orientation="horizontal", color=color, alpha=0.5,density=True)
    axs.set_ylim([df_total[target_name].min(),df_total[target_name].max()])
    #axs.set_xscale('log')
    #axs.set_xlim([0.9,1e4])
def plot_exploration_evolution(BOresults, df_total_org, nb_initialisation,nb_iteration=100,axs=None,color=search_to_color['BO'],label='BO',operation= np.max,target_name='target',aim=5.5,topKmol = 1000):
    
    df_total=df_total_org.copy()
    df_total[target_name] = df_total[target_name].apply(lambda x: -np.sqrt((x-aim)**2))
    min_target_out_of_database = -np.sort(-df_total[target_name].values)[topKmol]
    y_elm, y_elm_sig_bot, y_elm_sig_top = plot_element_above_min(BOresults,min_target_out_of_database,nb_iterations=nb_iteration, topKmol = topKmol,axs=axs[2],color=color,label=label,df_total=df_total,nb_initialisation=0)
    y_mean_mu_BO = plot_y_mean(BOresults,nb_iterations=nb_iteration,axs=axs[1],color=color,label=label,target_name=target_name,df_total=df_total,nb_initialisation=0)
    y_max_mu_BO, y_max_sig_bot_BO, y_max_sig_top_BO = plot_y_max(BOresults,nb_iterations=nb_iteration,axs=axs[0],color=color,label=label,df_total=df_total,operation=operation,target_name=target_name,nb_initialisation=0)
    #df_total[target_name].hist(ax=axs[3], bins=20, orientation="horizontal", color=search_to_color['BO'])
    #axs[3].set_ylim([df_total[target_name].min(),df_total[target_name].max()])
    min_target_out_of_database = -np.sort(-df_total[target_name].values)[100]
    y_elm, y_elm_sig_bot, y_elm_sig_top = plot_element_above_min(BOresults,min_target_out_of_database,nb_iterations=nb_iteration, topKmol = 100,axs=axs[3],color=color,label=label,df_total=df_total,nb_initialisation=0)

import glob
import pickle
def load_search_data(search_type, date,test_name,min_eval=100):
    files = glob.glob(f'data/output/search_experiment/{test_name}/'+search_type+'/'+date+'/*.pkl')
    BOresults = []
    max_num_eval = 0
    for file in files:

        with open(file, 'rb') as f:
            results = pickle.load(f)
            if len(results['fitness_acquired'])>min_eval:
                BOresults.append(results)
                max_num_eval = max(max_num_eval,len(results['fitness_acquired']))
    print(len(BOresults),max_num_eval)
    return BOresults