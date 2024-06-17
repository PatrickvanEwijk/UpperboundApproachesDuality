import numpy as np
np.set_printoptions(linewidth=200, edgeitems=8)
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.path import Path
import matplotlib.colors as colors
colors_list = list(colors._colors_full_map.values())

import pickle as pic
import pandas as pd
from utils import save_graph, smoothen_label
import re 
from scipy import stats

SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 18

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
width_triangles=2
height_triangles=4
triangle_up_ = lambda width_triangles,height_triangles : Path([[-1*width_triangles, 0], [width_triangles, 0], [0, height_triangles], [-1*width_triangles, 0]], closed=True)
triangle_down_ = lambda width_triangles,height_triangles : Path([[-1*width_triangles, 0], [width_triangles, 0], [0, -1*height_triangles], [-1*width_triangles, 0]], closed=True)

print('Addition name, press enter for no addition ')
name_addition = input().strip()
print('Methods removal, split by - ')
input_seq_to_remove=input().strip()
key_words_label_remove = [i.strip() for i in (input_seq_to_remove).split('-')]
key_words_label_remove=[i for i in key_words_label_remove if i!='']
file_='run20240417110433.pic' # MultipleStopping
#file_ ='run20240416210454.pic' #fBM.
file_='run20240418180419.pic'
file_='run20240531080505.pic'# Test Run fair computational effort analysis
file_= 'run20240601060629.pic' # Google cloud run
file_='run20240602090618.pic'# Google cloud run
file_='run20240606200655.pic'# Google cloud run FINAL MAX CALL
file_='run20240608230625.pic' # test run fbm4
file_= 'run20240609210633.pic' # Final Run fair computational effort fbm
file_= 'run20240610110654.pic' # Google cloud run final Final Run fair computational effort analysis (90,5,L)
# file_='run20240614080614.pic'# Final Run fair computational effort fbm
# std_=True


file_ ='run20240615230639.pic' # final run 16-6 for fbm N_T=9
file_ = 'run20240616130610.pic'# final run 16-6 for Bermudan Max Call N_T=9

file_basename=file_.split('.')[0]
with open(file_, 'rb') as file:
    information=pic.load(file)
df = pd.DataFrame.from_records(information)
df.iloc[:,0] = (df.iloc[:,0]).str.replace('Belom. et al 1', ' Belom. et al. (2009)')
type_fbm= df.iloc[:,2].max()>=0 and df.iloc[:,2].max()<=1
print(type_fbm)

drop_outly_schoenmakers2013_belom2009_fbm=True

if type_fbm:
    methods = df.iloc[:,0]
    ## Smoothen name of methods in figures.
    methods=methods.apply(lambda x: smoothen_label(x))
    df.iloc[:,0]=methods
    unique_methods= list(np.unique(methods))
    unique_methods = [m for m in unique_methods if not(any(keyword.lower().strip() in m.lower() for keyword in key_words_label_remove))]
    df=df.loc[df.iloc[:,0].isin(unique_methods)]
    methods = df.iloc[:,0]
    lower_bounds = df.iloc[:,7]
    lower_bounds_cv =  df.iloc[:,[2, 13, 14]]
#  lower_bounds_cv['95_perc_CI_upper_limit']=lower_bounds_cv.loc[:, 14].values-stats.norm.ppf(0.975)*lower_bounds_cv.loc[:,15].values
   # lower_bounds_cv.iloc[:, 2]= lower_bounds_cv.iloc[:, 1]+ stats.norm.ppf(0.05)*lower_bounds_cv.iloc[:,2]
    loc_LBest_small_se = lower_bounds_cv.groupby(2)[14].idxmin()
    lower_bounds_cv_sel = lower_bounds_cv.loc[loc_LBest_small_se, [2, 13]].rename(columns={13: 'best'}) # Select lower bound with lowest standard error for gap.
    lower_bounds_cv=pd.merge(left= lower_bounds_cv, right=lower_bounds_cv_sel, on=2)['best']
    lower_bounds_cv.index= df.index
    hurst = df.iloc[:, 2]

    for std_ in [True, False]:
        upper_bounds = df.iloc[:,9]
        lower_bounds_cv=df.iloc[:,[2,13, 14]]

        lower_bounds_cv =  df.iloc[:,[2, 13, 14]]
    #  lower_bounds_cv['95_perc_CI_upper_limit']=lower_bounds_cv.loc[:, 14].values-stats.norm.ppf(0.975)*lower_bounds_cv.loc[:,15].values
      #  lower_bounds_cv.iloc[:, 2]= lower_bounds_cv.iloc[:, 1]+ stats.norm.ppf(0.05)*lower_bounds_cv.iloc[:,2]
        loc_LBest_low_cv = lower_bounds_cv.groupby(2)[14].idxmin()
        lower_bounds_cv_sel = lower_bounds_cv.loc[loc_LBest_low_cv, [2, 13]].rename(columns={13: 'best'})
        lower_bounds_cv=pd.merge(left= lower_bounds_cv, right=lower_bounds_cv_sel, on=2)['best']
        lower_bounds_cv.index= df.index
        
        if std_ :
            upper_bounds = upper_bounds + stats.norm.ppf(0.95)* df.iloc[:, 10]
        gap = upper_bounds-lower_bounds_cv
        ### FIG GAP
        for num,method in enumerate(unique_methods): ## Style overlapping points a bit ad hoc
            mask_method = methods.index[(methods==method)]
            offset_=pd.DataFrame([0.0 for j in range(len(hurst.loc[mask_method]))], index=hurst.loc[mask_method])      
            offset_factor= 0.003          
            if 'glasserman' in method.lower():
                offset_.loc[0.2]=offset_factor
            elif 'andersen' in method.lower():
                offset_.loc[0.2]= -1*offset_factor              
            if '=0.5' in method.lower():
                offset_.loc[0.7]= -1*offset_factor  
            elif '=1' in method.lower():
                offset_.loc[0.7]= offset_factor    
            if '=1' in method.lower():
                offset_.loc[0.45]= 1*offset_factor    
            elif '=0.05' in method.lower():
                offset_.loc[0.45]= -1*offset_factor  
            if std_:
                if 'desai' in method.lower() and '[i]' in method.lower():
                    offset_.loc[0.45]= 1*offset_factor 
                elif '2023' in method.lower() and '[ii]' in method.lower():
                    offset_.loc[0.45]= -1*offset_factor 
                if 'desai' in method.lower() and '[ii]' in method.lower():
                    offset_.loc[0.45]= -1*offset_factor 
                elif '=0.5' in method.lower():
                    offset_.loc[0.45]= 1*offset_factor 
            else:
                if '2019' in method.lower():
                    offset_.loc[0.7]= 1*offset_factor 
                elif '=0.05' in method.lower():
                    offset_.loc[0.7]= -1*offset_factor 
            offset_.index=hurst.loc[mask_method].index
            color_= f'C{num}' if num<10 else colors_list[int(1.2*num+4)]
            plt.scatter((hurst.loc[mask_method].values+offset_.values.flatten()), gap.loc[mask_method].values,c=color_, label=method)

        plt.xlabel(r'$H$') #'Hurst')
        if not(std_):
            plt.ylabel(r'GAP')
        else:
            plt.ylabel(r'GAP$_{95}$')
        # plt.title('GAP vs Hurst')
        legend = plt.legend(bbox_to_anchor=(1.015, 1.05), loc='upper left', borderaxespad=.1, facecolor='white')
        plt.gca().add_artist(legend)
        if drop_outly_schoenmakers2013_belom2009_fbm:
            if std_ is True:
                plt.ylim(0, 0.014)
            else:
                plt.ylim(0, 0.0125)
        if std_:
            addition='upperQuantile'
        else:
            addition=''
        save_graph(f'GAP{file_basename}{name_addition}{addition}.pdf')

    upper_bounds = df.iloc[:,9]
    hurst = df.iloc[:, 2]
    dataframe_sel = df.iloc[:, [0, 2, 9, 7]] 

    ### FIG Bound
    triangle_down = triangle_down_( 1/110*(np.max(lower_bounds_cv)-np.min(lower_bounds_cv)), 1/40*(np.max(lower_bounds_cv)-np.min(lower_bounds_cv)))
    triangle_up = triangle_up_( 1/110*(np.max(lower_bounds_cv)-np.min(lower_bounds_cv)), 1/40*(np.max(lower_bounds_cv)-np.min(lower_bounds_cv)))
    fig, ax1 = plt.subplots(1, 1) 
    lb=dict()
    ub=dict()
    lower_bounds = df.iloc[:,8]
    upper_bounds = df.iloc[:,9]
    triangle_down = triangle_down_( 1/55*(np.max(lower_bounds_cv)-np.min(lower_bounds_cv)), 1/30*(np.max(lower_bounds_cv)-np.min(lower_bounds_cv)))
    triangle_up = triangle_up_( 1/55*(np.max(lower_bounds_cv)-np.min(lower_bounds_cv)), 1/30*(np.max(lower_bounds_cv)-np.min(lower_bounds_cv)))
    offset_= np.linspace(-.02, .02, len(unique_methods))
    for number_method, method in enumerate(unique_methods):
        mask_method = methods.index[(methods==method)]
        color_= f'C{number_method}' if number_method<10 else colors_list[int(1.2*number_method+4)]
        ub[method]=ax1.scatter(hurst.loc[mask_method]+offset_[number_method], upper_bounds.loc[mask_method], label=method, marker=triangle_up, c=color_, linewidths=0.5)
    number_method+=1
    col_ = colors_list[int(1.2*number_method)+4] if number_method>9 else f'C{number_method}'
    ub['Lower bound'] = ax1.scatter(hurst.loc[mask_method], lower_bounds_cv.loc[mask_method], label='Lower bound', marker=triangle_down, c=col_, linewidths=0.5)
    l = ax1.legend( list(ub.values()), list(ub.keys()), bbox_to_anchor=(1.015, 1.05),  loc='upper left', markerscale=1.5, scatterpoints=1,  handletextpad=0, borderaxespad=.1, borderpad=.8) # bbox_to_anchor=(1, 0.5),
    for i, t in enumerate(l.get_texts()):
        t.set_va('bottom') 
        if i == len(l.get_texts()) - 1:  # If it's the last text entry
            t.set_va('center_baseline') 

    legend_frame = l.get_frame()
    legend_frame.set_facecolor('white')  # Set face color to white
    plt.gca().add_artist(l)
    ax1.set_xlabel(r'$H$')#'Hurst')
    ax1.set_ylim(0, np.max(upper_bounds)*1.05)
    ax1.set_xlim(0, np.max(hurst)+.1)
    ax1.set_xticks(np.arange(0, np.max(hurst)+.05, 0.1, dtype=np.float64))
    # .xticks(np.arange(np.max(exercise_right), dtype=int))
    ax1.set_ylabel('Optimal expected stopping value')
    save_graph(f'BOUNDS{file_basename}{name_addition}.pdf')

else:
    methods = df.iloc[:,0]
    ## Smoothen name of methods in figures.
    methods=methods.apply(lambda x: smoothen_label(x))
    # pattern_remove_labels=r'((-\d+)?)\s(\d{1,5})$'
    # methods=methods.str.replace(pattern_remove_labels, '', regex=True)
    # pattern_remove_labels2=r'KU\d{1,5}'   
    # methods=methods.str.replace(pattern_remove_labels2, '', regex=True)
    # methods=methods.str.replace('lambda', r'$\lambda$')
    df.iloc[:,0]=methods
    unique_methods= list(np.unique(methods))
    unique_methods = [m for m in unique_methods if not(any(keyword.lower().strip() in m.lower() for keyword in key_words_label_remove))]
    df=df.loc[df.iloc[:,0].isin(unique_methods)]


    for std_ in [True, False]:
        lower_bounds = df.iloc[:,8]
        lower_bounds_cv=df.iloc[:,[3,14, 15]]
    #  lower_bounds_cv['95_perc_CI_upper_limit']=lower_bounds_cv.loc[:, 14].values-stats.norm.ppf(0.975)*lower_bounds_cv.loc[:,15].values
       # lower_bounds_cv.iloc[:, 2]= lower_bounds_cv.iloc[:, 1]+ stats.norm.ppf(0.05)*lower_bounds_cv.iloc[:,2]
        loc_LBest_low_se = lower_bounds_cv.groupby(3)[15].idxmin()
        lower_bounds_cv_sel = lower_bounds_cv.loc[loc_LBest_low_se, [3, 14]].rename(columns={14: 'best'})
        lower_bounds_cv=pd.merge(left= lower_bounds_cv, right=lower_bounds_cv_sel, on=3)['best']
        lower_bounds_cv.index= df.index
        methods = df.iloc[:,0]
        upper_bounds = df.iloc[:, 10]
        if std_ :
            upper_bounds = upper_bounds + stats.norm.ppf(0.95)* df.iloc[:, 11]
        exercise_right = df.iloc[:, 3]
        dataframe_sel = df.iloc[:, [0, 3, 8, 10]]
        gap = upper_bounds-lower_bounds_cv#upper_bounds-lower_bounds
        print(dataframe_sel)
        
        ### FIG GAP
        for num, method in enumerate(unique_methods):
            mask_method = methods.index[(methods==method)]
            color_= f'C{num}' if num<10 else colors_list[int(1.2*num+4)]
            # color_= f'C{num}' if num<10 else colors_list[num+3]
            plt.scatter(exercise_right.loc[mask_method], gap.loc[mask_method],c=color_, label=method)

        plt.xlabel('$L$') #'Exercise Rights')
        plt.xticks(np.arange(1, np.max(exercise_right)+.2, dtype=int))
        plt.xlim(0.8, np.max(exercise_right)+.2)
        if not(std_):
            plt.ylabel(r'GAP')
        else:
            plt.ylabel(r'GAP$_{95}$')
        # if not(std_):
        #     plt.title('GAP vs Exercise Right') 
        # else:
        #     plt.title('GAP (95\% upper quantile UB) vs Exercise Right') 
        legend = plt.legend(bbox_to_anchor=(1.015, 1.05), loc='upper left',  borderaxespad=.1, facecolor='white')
        plt.gca().add_artist(legend)
        if std_:
            addition='upperQuantile'
        else:
            addition=''
        save_graph(f'GAP{file_basename}{name_addition}{addition}.pdf')

    ### FIG Bound
    lower_bounds = df.iloc[:,8]
    lower_bounds_cv=df.iloc[:,[3,14, 15]]
    lower_bounds_cv=df.iloc[:,[3,14, 15]]
#  lower_bounds_cv['95_perc_CI_upper_limit']=lower_bounds_cv.loc[:, 14].values-stats.norm.ppf(0.975)*lower_bounds_cv.loc[:,15].values
  #  lower_bounds_cv.iloc[:, 2]= lower_bounds_cv.iloc[:, 1]+ stats.norm.ppf(0.05)*lower_bounds_cv.iloc[:,2]
    loc_LBest_low_se = lower_bounds_cv.groupby(3)[15].idxmin()
    lower_bounds_cv_sel = lower_bounds_cv.loc[loc_LBest_low_se, [3, 14]].rename(columns={14: 'best'})
    lower_bounds_cv=pd.merge(left= lower_bounds_cv, right=lower_bounds_cv_sel, on=3)['best']
    lower_bounds_cv.index= df.index
    exercise_right = df.iloc[:, 3]

    fig, ax1 = plt.subplots(1, 1) 
    lb=dict()
    ub=dict()
    lower_bounds = df.iloc[:,8]
    upper_bounds = df.iloc[:, 10]
    triangle_down = triangle_down_( 1/55*(np.max(lower_bounds_cv)-np.min(lower_bounds_cv)), 1/30*(np.max(lower_bounds_cv)-np.min(lower_bounds_cv)))
    triangle_up = triangle_up_( 1/55*(np.max(lower_bounds_cv)-np.min(lower_bounds_cv)), 1/30*(np.max(lower_bounds_cv)-np.min(lower_bounds_cv)))
    offset_= np.linspace(-.18, .18, len(unique_methods))
    for number_method, method in enumerate(unique_methods):
        mask_method = methods.index[(methods==method)]
       # lb[method]=ax1.scatter(exercise_right.loc[mask_method], lower_bounds.loc[mask_method], label=method, marker=triangle_down, c=f'C{number_method}', linewidths=0.5)
        color_= f'C{number_method}' if number_method<10 else colors_list[number_method+3]
        ub[method]=ax1.scatter(exercise_right.loc[mask_method]+offset_[number_method], upper_bounds.loc[mask_method], label=method, marker=triangle_up, c=color_, linewidths=0.5)
    number_method+=1
    col_ = colors_list[number_method+4] if number_method>9 else f'C{number_method}'
    ub['Lower bound'] = ax1.scatter(exercise_right.loc[mask_method], lower_bounds_cv.loc[mask_method], label='Lower bound', marker=triangle_down, c=col_, linewidths=0.5)
    #dict_bounds ={method: (lb[method], ub[method]) for method in lb.keys()}
    #l = ax1.legend( list(dict_bounds.values()), list(dict_bounds.keys()), handler_map={tuple: HandlerTuple(ndivide=None)},bbox_to_anchor=(1.05, 1.05),  loc='upper left') # bbox_to_anchor=(1, 0.5),
    
    # scatteryoffsets_ =[-0.3,4]
    #scatteryoffsets_[3]=4
    l = ax1.legend( list(ub.values()), list(ub.keys()), bbox_to_anchor=(1.015, 1.05),  loc='upper left', markerscale=1.5, scatterpoints=1,  handletextpad=0, borderaxespad=.1, borderpad=.8) # bbox_to_anchor=(1, 0.5),
    for i, t in enumerate(l.get_texts()):
        t.set_va('bottom') 
        if i == len(l.get_texts()) - 1:  # If it's the last text entry
            t.set_va('center_baseline') 

    legend_frame = l.get_frame()
    legend_frame.set_facecolor('white')  # Set face color to white
    plt.gca().add_artist(l)
    ax1.set_xlabel(r'$L$') #'Exercise Rights')
    ax1.set_ylim(0, np.max(upper_bounds)*1.05)
    ax1.set_xlim(0.3, np.max(exercise_right)+.7)
    ax1.set_xticks(np.arange(1, np.max(exercise_right)+.2, dtype=int))
    # .xticks(np.arange(np.max(exercise_right), dtype=int))
    ax1.set_ylabel('Optimal expected stopping value')
    # ax1.set_title('Lowerbound and Upperbound vs Exercise Right')
    save_graph(f'BOUNDS{file_basename}{name_addition}.pdf')
    