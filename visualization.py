import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from tsl.utils import numpy_metrics
import matplotlib.dates as mdates
from matplotlib.lines import Line2D


log_dir = 'output.npz'



output = np.load(log_dir)

y_hat, y_true, observed_mask, eval_mask = output['y_hat'], \
                          output['y'], \
                          output['observed_mask'], \
                          output['eval_mask']



# check_mae = numpy_metrics.masked_mae(y_hat, y_true, eval_mask)
#
# n_eval = np.sum(eval_mask)
# print(f'Evalpoint: {n_eval}')
# print(f'Test MAE: {check_mae:.5f}')

all_target_np = output['y'].squeeze(-2)
all_evalpoint_np = output['eval_mask'].squeeze(-2)
all_observed_np = output['observed_mask'].squeeze(-2)

# print how many evalpoints we have
print(f'Evalpoint: {np.sum(all_evalpoint_np)}')

if 'imputed_samples' in output:
    samples = output['imputed_samples']
    samples = samples.squeeze(-1)
else:
    samples = output['y_hat']
    samples = samples.squeeze(-2)[:, np.newaxis, ...]

qlist =[0.05,0.25,0.5,0.75,0.95]
quantiles_imp= []
for q in qlist:
    tmp = np.quantile(samples, q, axis=1)
    quantiles_imp.append(tmp*(1-all_observed_np) + all_target_np * all_observed_np)


start = 0
end = all_target_np.shape[0] - 1

K = all_target_np.shape[1]

dataind = 0

#######################################
plt.rcParams["font.size"] = 16
fig, axes = plt.subplots(nrows=K, ncols=1,figsize=(36, 24.0))



for k in range(K):
    df = pd.DataFrame({"x":np.arange(0, end-start), "val":all_target_np[start:end, k], "y":all_evalpoint_np[start:end,k]})
    df = df[df.y != 0]
    df2 = pd.DataFrame({"x":np.arange(0, end-start), "val":all_target_np[start:end, k], "y":all_observed_np[start:end,k]})
    df2 = df2[df2.y != 0]
    axes[k].plot(range(0,end-start), quantiles_imp[2][start:end,k], color = 'g',linestyle='solid',label='CSDI')
    axes[k].fill_between(range(0,end-start), quantiles_imp[0][start:end,k],quantiles_imp[4][start:end,k],
                    color='g', alpha=0.3)
    axes[k].plot(df2.x,df2.val, color = 'r',marker = 'x', linestyle='None')
    axes[k].plot(df.x, df.val, color='b', marker='o', linestyle='None')


plt.show()
#######################################


