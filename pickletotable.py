import os
import sys
from datetime import datetime
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils
from tabulate import tabulate
import pickle as pic
from scipy import stats
import pandas as pd
from txt_to_pickle import txt_to_pickle

############### TEST RUNS ##################
file_ = 'run20240418180419.pic' #Ms
file_ = 'run20240502160503.pic'
#file_ ='run20240416210454.pic' #fBM.
file_ = 'run20240503200517.pic' # Independent set (pca3)
file_= 'run20240503220535.pic' # 200 no pca
file_= 'run20240503230502.pic' # pca 1

file_0 = 'run20240526170531.pic' # KAGGLE fBM Belomestny, Bender, Schoenmakers 2024
file_='run20240526200556.pic' # KAGGLE fBM  1.0, 1.2
file_2='run20240526220547.pic'# KAGGLE fBM  1.0, 1.2
file_='run20240527040556.pic' # Original M-C
file_2 = 'run20240526220508.pic'# Additional M-C
file_MC = 'run20240527110533.pic'# Extended M-C (A)
file_FBM='run20240527110508.pic'# Extended fbm (A)
file_ = 'run20240529120545.pic' # Test Run fair computational effort analysis
file_ = 'run20240529230505.pic' # Test Run fair computational effort analysis
file_ = 'run20240530150543.pic'# Test Run fair computational effort analysis
file_= 'run20240531080505.pic'# Test Run fair computational effort analysis
#file_ = 'run20240527040556.pic'
# file_='run20240511150517.pic'
# file_='run20240511180558.pic'
file_ = 'run20240601060629.pic' # Google Cloud Run (test)
file_='run20240602070653.pic' # Google Cloud Run (test)
file_='run20240602090618.pic'# Google Cloud Run (test)
file_='run20240602140619.pic'# Google Cloud Run (test)
file_='run20240604030642.pic'# Google Cloud Run (test [50,9,L])
file_='run20240606200655.pic'# Google Cloud Run (test [50,9,L])
# file_='run20240605010655.pic' # Test run high dimensions
file_='run20240606180647.pic'
file_='run20240607020621.pic'
file_='run20240608180610.pic' #fBM test run
file_ = 'run20240609070632.pic'# test Run fair computational effort analysis (90,5,L)
file_ = 'run20240608230625.pic'# test Run fair computational effort fbm
file_2= 'run20240609210633.pic' # Final Run fair computational effort fbm
file_= 'run20240610110654.pic' # Google cloud run final Final Run fair computational effort analysis (90,5,L)
file_= 'run20240610200648.pic'# Google cloud run final Final Run fair computational effort analysis (90,10,3)
file_='run20240611230633.pic'# test Run fair computational effort analysis (90,5,3) for N_T=49
file_='run20240612100609.pic'# Run fair computational effort analysis (90,5,3) for N_T=49
file_= 'run20240612140646.pic'# test Run fair computational effort analysis fBm for N_T=49
file_='run20240612220606.pic' # high dimensional test instance
file_ = 'run20240612220603.pic' # test instance fbm
file_='run20240613070643.pic' # final run fbm N_T =49
file_ = 'run20240614040605.pic'# Run fair computational effort fbm (final N_T=9). H=0.45
file_='run20240614080614.txt' # Run fair computational effort fbm (final N_T=9). All H
file_='run20240614070647.txt' # test Run fair computational effort fbm (N_T=49). H=0.45
file_='run20240614100613.txt'# final run fbm N_T =49 including H=0.45
file_='run20240615180639.txt'# test run final run fbm N_T =49
file_= 'run20240616170654.txt'# final run 16-6 for Bermudan Max Call N_T=49
file_= 'run20240617160643.txt'
file_='run20240618120619.txt'
file_='run20240618150618.txt'
file_='run20240619030632.txt'
file_='run20240619000608.txt' # fBM higher inner sim AB (too high)
file_='run20240619110659.txt' # fBM higher inner sim AB
file_='run20240619120658.txt'# fBM higher inner sim AB

############### FINAL RUNS ##################
file_ ='run20240625110611.txt'# final run 16-6 for fbm N_T=49.=final run 25-6
file_='run20240619120658.txt'# final run fBM (higher inner sim AB), N_T=9. 22-6
file_='run20240622210621.txt' #final run fBM (higher inner sim AB), N_T=9, hurst close to 0.5; 22-6

file_='run20240622120626.txt'# final run Bermudan Max Call N_T=9. 22-6. Div=.15, r=0.0
file_ ='run20240622200627.txt'# final run Bermudan Max Call N_T=9, d=10. Div=.15, r=0.0
file_='run20240623160644.txt'# final run Bermudan Max Call N_T=49. 22-6. Div=.15, r=0.0



def read_file(file_):
    try:
        file_pic=file_.replace('txt', 'pic')
        with open(file_pic, 'rb') as fh:  
            information=pic.load(fh)
            return information
    except Exception:
        txt_to_pickle(file_)
        base_name=file_.split('.')[0]
        file_=f'{base_name}.pic'
        with open(file_, 'rb') as fh:
            information=pic.load(fh)
            return information
information=read_file(file_)
# with open(file_2, 'rb') as fh:
#     information_original =pic.load(fh)
# information_all = [*information, *information_original]
# import github
# from github import Auth
# from github import Github
# auth = Auth.Token('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')
# g = github.Github(auth=auth)
# repo = g.get_repo('PatrickvanEwijk/UpperboundApproachesDuality')
# repo.create_file(f'resultfiles/run{datetime.now().strftime("%Y%m%d%H%m%S")}.txt', f'run{datetime.now().strftime("%Y%m%d%H%m%S")}.txt', str(information_all), branch='main')



i_2= [i[2] for i in information]
FBM= min(i_2)>=0 and max(i_2)<=1 # Auto check if underlying data file is fBm/max call type: 3rd column= price Max Call and H fBm.
BBS_A_analysis=False
fuji=False
plot=False
GAP_comparison= True # If True, all upper bounds compared to lower bound with control variate with lowest standard error.
if BBS_A_analysis:
    information_incl_95cb=[]
    for i in information:
        i[1]="%.1f" % i[1]+ '\hspace{0cm}' 
        i[2]="%.2f" % i[2]+ '\hspace{0cm}'
        i.append(i[10]+stats.norm.ppf(0.95)*i[11] ) #95\% upper confidence bound.
        information_incl_95cb.append(i)
    information=information_incl_95cb
    if FBM:
        table_ = tabulate(utils.information_format_fbm_BBS(information), headers=utils.header_bbs_fbm, tablefmt="latex_raw", floatfmt=".5f")
    else:
        table_ = tabulate(utils.information_format_maxcall_BBS(information), headers=utils.header_bbs, tablefmt="latex_raw", floatfmt=".4f")
    if plot and not FBM:
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        information=pd.DataFrame.from_records(information)
        fig = plt.figure()
        ## Empirical
        ax = fig.add_subplot(111, projection='3d')
        DATA_sel= np.where(information.iloc[:,-1]< np.quantile( information.iloc[:,-1], 0.8))[0]
        ax.scatter(information.iloc[DATA_sel, 1].str.replace('\\hspace{0cm}','').astype(float),information.iloc[DATA_sel, 2].str.replace('\\hspace{0cm}','').astype(float),information.iloc[DATA_sel,-1])
        # ax.set_title('95\% Upper Quantile')
        ax.set_xlabel(r'$\mu_A$')
        ax.set_ylabel(r'$\sigma_A$')
        ax.set_zlabel(r'95\% A.S. Upper Quantile')
        ax.set_box_aspect(aspect=None, zoom=0.95)
        ax.view_init(elev=17, azim=-53)
        plt.show()
else:
    if FBM:
        info_original=pd.DataFrame.from_records(information)
        if fuji is False:
            info_transformed=pd.DataFrame.from_records(utils.information_format_fbm_no_fu(information), columns=utils.header_fbm_no_fu)
        else:
            info_transformed=pd.DataFrame.from_records(utils.information_format_fbm(information), columns=utils.header_fbm)
        lower_bounds_cv=info_original.iloc[:,[2,13, 14]]
        upper_bounds = info_original.iloc[:, 9]
       # lower_bounds_cv.iloc[:, 2]= lower_bounds_cv.iloc[:, 1]+ stats.norm.ppf(0.05)*lower_bounds_cv.iloc[:,2]
        loc_lowest_se = lower_bounds_cv.groupby(2)[14].idxmin()
        lower_bounds_cv_sel = lower_bounds_cv.loc[loc_lowest_se, [2, 13]].rename(columns={13: 'best'})
        lower_bounds_cv=pd.merge(left= lower_bounds_cv, right=lower_bounds_cv_sel, on=2)['best']
        lower_bounds_cv.index= info_original.index
        info_transformed['GAP']= upper_bounds- lower_bounds_cv

    else:
        info_original=pd.DataFrame.from_records(information)
        info_transformed = pd.DataFrame.from_records(utils.information_format(information), columns=utils.header_)
        lower_bounds_cv=info_original.iloc[:,[3,14, 15]]
        upper_bounds = info_original.iloc[:, 10]
        #lower_bounds_cv.iloc[:, 2]= lower_bounds_cv.iloc[:, 1]+ stats.norm.ppf(0.05)*lower_bounds_cv.iloc[:,2]
        loc_lowest_se= lower_bounds_cv.groupby(3)[15].idxmin()
        lower_bounds_cv_sel = lower_bounds_cv.loc[loc_lowest_se, [3, 14]].rename(columns={14: 'best'})
        lower_bounds_cv=pd.merge(left= lower_bounds_cv, right=lower_bounds_cv_sel, on=3)['best']
        lower_bounds_cv.index= info_original.index
        info_transformed['GAP']= upper_bounds- lower_bounds_cv

    table_ = tabulate(info_transformed.values, headers=info_transformed.columns, tablefmt="latex_raw", floatfmt=".4f")

print(table_)
fh = open(f'logresults.txt', 'a')
fh.write(f'{datetime.now()}\n ')
fh.writelines(table_)
line="".join( ['*' for i in range(75)])
fh.write(f'\n {line} \n')
fh.close()


# information =  [[i[0]+' Cor.', *i[1:]] for i in information]

# file_2 ='run20240416100435.pic' #fBM.
# with open(file_2, 'rb') as fh:
#     information2=pic.load(fh)
# information_merged = [*information2, *information]


# with open(f'run{datetime.now().strftime("%Y%m%d%H%m%S")}.pic', 'wb') as fh:
#     pic.dump(information_merged, fh)
