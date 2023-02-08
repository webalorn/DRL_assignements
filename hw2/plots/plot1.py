import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def read_csv(name):
    return pd.read_csv(f'plots/data/{name}.csv')['Value'].to_numpy()

# ========== Q1.2 ==========
# data_dsa = read_csv('q1_sb_no_rtg_dsa_avgreturn')
# data_na = read_csv('q1_sb_no_rtg_na_avgreturn')
# X = np.arange(len(data_na))+1

# plt.plot(X, data_dsa, label='sb_no_rtg_dsa')
# plt.plot(X, data_na, label='sb_no_rtg_na')


# ========== Q2 ==========
r_05, std_05 = read_csv('q2_l0.5_r'), read_csv('q2_l0.5_std')
r_1, std_1 = read_csv('q2_l1_r'), read_csv('q2_l1_std')
r_25, std_25 = read_csv('q2_l2.5_r'), read_csv('q2_l2.5_std')

X = np.arange(len(r_05))+1

plt.plot(X, r_05, label='b=2759, x=0.0996 ($\lambda$=0.5)', color='blue')
plt.fill_between(X, r_05-std_05, r_05+std_05, alpha=0.3, color='blue')

plt.plot(X, r_1, label='b=1097, r=0.0328 ($\lambda$=1)', color='orange')
plt.fill_between(X, r_1-std_1, r_1+std_1, alpha=0.3, color='orange')

plt.plot(X, r_25, label='b=790, r=0.0294 ($\lambda$=2.5)', color='green')
plt.fill_between(X, r_25-std_25, r_25+std_25, alpha=0.3, color='green')

# ==================================================

plt.xlabel('Number of epochs')
plt.ylabel('Average return')

plt.ylim(ymin=0)
plt.legend(loc='lower right')
plt.show()