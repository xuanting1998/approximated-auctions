import csv
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

plt.style.use('ggplot')

data = pd.read_csv('quota_and_bidder_data_full.csv',header=None, names=('sim_id', 'quota_factor', 'num_bidders', 'err'))
data['num_bidders'] = data['num_bidders'].astype(int)
# sets any entries with 0 error to the machine epsilon to avoid taking log(0)
data.loc[data['err'] == 0,'err'] = np.finfo(float).eps
grouped_data = data.groupby(('quota_factor', 'num_bidders'), as_index=False)

mean_err_bidders = grouped_data.agg({'err': 'mean'})
# removes 33 outliers out of 317 entries whose group mean is close to 0
mean_err_bidders = mean_err_bidders[mean_err_bidders['err'] >= 5e-16]

mean_err_bidders_mod = smf.ols(formula='np.log(err) ~ quota_factor', data=mean_err_bidders)
mean_err_bidders_res = mean_err_bidders_mod.fit()
print mean_err_bidders_res.summary()

individual_regs = []

for n in mean_err_bidders.sort_values('num_bidders')['num_bidders'].unique():
    data_for_n = mean_err_bidders[mean_err_bidders['num_bidders'] == n]
    mean_err_for_n_mod = smf.ols(formula='np.log(err) ~ quota_factor', data=data_for_n)
    mean_err_for_n_res = mean_err_for_n_mod.fit()
    individual_regs.append((n, mean_err_for_n_res))

print [(res[0], res[1].rsquared, res[1].params) for res in individual_regs]
print [res[1].params['quota_factor'] for res in individual_regs]

sm.graphics.plot_fit(mean_err_bidders_res, 'quota_factor')
plt.show()

std_err = grouped_data.agg({'err': 'std'})
# removes 33 outliers out of 317 entries whose group standard deviation is close to 0
std_err = std_err[std_err['err'] >= 1e-10]

std_err_mod = smf.ols(formula='np.log(err) ~ quota_factor', data=std_err)
std_err_res = std_err_mod.fit()
print std_err_res.summary()

sm.graphics.plot_fit(std_err_res, 'quota_factor')
plt.show()