# Comparative Analysis of Upper Bound Approaches for high dimensional Stopping Problems
This directory corresponds to a comparative analysis of various upper bound duality approaches for stopping problems, based on the dual representation by Rogers (2002) and Haugh and Kogan (2004),

![equation](http://www.sciweavers.org/tex2img.php?eq=Y_0%5E%2A%3D%20%20%5Cunderset%7BM%5Cin%20%5Cmathcal%7BM%7D%7D%7B%5Cinf%7D%20%5C%20%20%5Cmathbb%7BE%7D_0%20%5Cleft%5B%20%20%5Cunderset%7B%20j%5Cin%5C%7B0%2C1%2C%20%5Chdots%2C%20%5C%20%5Cmathcal%7BN%7D_T%5C%7D%7D%7B%5Cmax%7D%20%28Z_%7Bt_j%7D%20-%20M_%7Bt_j%7D%29%20%5Cright%5D.%20%5Chspace%7B4cm%7D%20%5Ctext%7BDual%20%20%20%281.2%29%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

The approaches are compared to the application of stopping a fractional Brownian motion and Bermudan max call (with multiple exercise rights, which should be used (if so) at different dates). Randomised Neural Networks (related to Extreme Learning Machines) are used as basis functions.
Generally, the primal LSMC is implemented using the Longstaff & Schwartz approach (2001)[^1].

## Approaches
### Primal-Dual Inner Simulation
  - Glasserman (2006), somewhat related to Haugh and Kogan (2004), who apply to supermartingales: fBMHK.py, multiplestoppingHK.py.
  - Andersen Broadie (2004): fBMABLS.py, multiplestoppingAB_fullyLS.py.
      -  Not used in the report:
           - multiplestoppingAB_fullyTvR.py: Tsitsiklis & Van Roy (2001) LSMC.
           - multiplestoppingAB_mainGAP: originally proposed by authors; involves the difference between lower and upper biased estimator which is estimated rather than upper biased estimator.
           - multiplestoppingAB_TvRpartLS.py: interpolation Tsitsiklis & Van Roy (2001) and Longstaff Schwartz LSMC across different exercise rights->continuation value Tsitsiklis & Van Roy (2001), stopping value current right-> Longstaff Schwartz.  
### Primal-Dual Nonnested
  - Belomestny et al. (2009): fBMBelomestny.py, multiplestoppingBelomestny.py.
  - Schoenmakers et al. (2013): fBMSchoenmakersSZH.py, multiplestoppingSchoenmakersSZH.py.
       - Not used in report: (fBMSchoenmakers.py, multiplestoppingSchoenmakers.py)-> involves not substracting martingale during LSMC. 
### Pure-Dual Linear Programming Formulations
   - Desai et al. (2012): fBMSAA.py, multiplestoppingSAA.py; both with mode_desai_BBS_BHS='desai'.
   - Belomestny et al. (2019): fBMSAA.py, multiplestoppingSAA.py; both with mode_desai_BBS_BHS='bhs'.
   - Belomestny et al. (2023): fBMSAA.py, multiplestoppingSAA.py; both with mode_desai_BBS_BHS='bbs'.
### Pure-Dual Non-Linear Formulations
   - Belomestny (2013): fBMSAABelomestny.py, multiplestoppingSAABelomestny.py.
       - Solved using BFGS as proposed by Dickmann (2014).

## Applications
1. Fractional Browian motion: /fractionalBrownianMotion/
2. Bermudan Max Call (with multiple exercise rights): /BermudanMaxCall_multiple_exercise_rights/

## Comparison
See /jupyter_notebooks_main_analysis/. 
Two Notebooks are present based on each application. Please note that github authentication link has been depreciated.
The result files have been stored in /resultfiles/.

## Additional analyses
1. Timing the pure dual (martingale minimisation approaches) and find a relationship between the empirical computation time in the relevant input parameters of the algorithms. See /timingMartingaleMinimisation/
2. Tuning the parameters in the choose of distribution of A in Belomestny et al. (2023). See /tuningAbelomestny2023/

## Other remarks
1. Jupyter Notebooks have been run in Google Cloud on a virtual machine (c3d-highmem-8, corresponding to a 4-core AMD EPYC (GENOA) 9B14 2.60 GHz CPU).
2. The Fujii et al. (2011) simple improvement upper bound has been implemented in the python files for the fractional Brownian motion, but has not been implemented in the report. 
In an earlier stage, this approach has shortly been considered. However, I did not invest significant time in this approach, so any use of this implementation should be undertaken with caution and at one's own risk.

 ## Additional python scripts
 1. txt_to_pickle.py: Convert raw data from .txt format to list format which is stored in a pickle file.
 2. pickletotable.py: For a specified pickle file (or txt file, which uses txt_to_pickle.py to convert to pickle), print the table corresponding to a raw data file.
 3. creatinfiguresfromrawdata.py: File to create figures corresponding to raw data files, and saving these in .pdf format. 
 4. utils.py: General file, containing column names for printed tables, function for setting seeds, function for saving figures in a certain resolution, smoothening the somewhat ill defined names of the approaches in the raw data files.
 5. modelRrobust2MS.py & modelRrobust2fBM.py: Object Oriented Programming classes for the approaches. Respectively for the Bermudan Max Call with multiple exercise rights and stopping a fractional Brownian motion.


[^1]: However, it is slightly adjusted as all trajectories are used rather than the ones in-the-money.

## Citation
 - Andersen, L., & Broadie, M. (2004). Primal-dual simulation algorithm for pricing multidimensional American options. Management Science, 50 (9), 1222–1234.
 - Belomestny, D. (2013). Solving optimal stopping problems via empirical dual optimization. The Annals of Applied Probability, 1988–2019.
 - Belomestny, D., Bender, C., & Schoenmakers, J. (2009). True upper bounds for bermudan products via non-nested Monte Carlo. Mathematical Finance: An International Journal of Mathematics, Statistics and Financial Economics, 19 (1), 53–71.
 - Belomestny, D., Bender, C., & Schoenmakers, J. (2023). Solving optimal stopping problems via randomization and empirical dual optimization. Mathematics of Operations Research, 48 (3), 1454–1480.
 - Belomestny, D., Hildebrand, R., & Schoenmakers, J. (2019). Optimal stopping via pathwise dual empirical maximisation. Applied Mathematics & Optimization, 79 (3), 715–741.
 - Dickmann, F. (2014). Multilevel approach for bermudan option pricing [Doctoral dissertation, Duisburg, Essen, 2014].
 - Fujii, M., Matsumoto, K., & Tsubota, K. (2011). Simple improvement method for upper bound of American option. Stochastics, 83(4-6), 449-466.
 - Glasserman, P. (2004). Monte carlo methods in financial engineering (Vol. 53). Springer.
 - Rogers, L. C. (2002). Monte carlo valuation of american options. Mathematical Finance, 12 (3), 271–286.
 - Schoenmakers, J., Zhang, J., & Huang, J. (2013). Optimal dual martingales, their analysis, and application to new algorithms for bermudan products. SIAM Journal on Financial Mathematics, 4 (1), 86–116.
 - Tsitsiklis, J. N., & Van Roy, B. (2001). Regression methods for pricing complex american-style options. IEEE Transactions on Neural Networks, 12 (4), 694–703.
 - Longstaff, F. A., & Schwartz, E. S. (2001). Valuing american options by simulation: A simple least-squares approach. The review of financial studies, 14 (1), 113–147.




