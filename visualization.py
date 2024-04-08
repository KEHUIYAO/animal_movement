import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data import AnimalMovement
deer_id_list = [5094]
model_list = ['csdi', 'interpolation']
missing_percent_list = [20, 50, 80]

for deer_id in deer_id_list:
    for model in model_list:
        for missing_percent in missing_percent_list:

            path = f'./results/{deer_id}_{missing_percent}_percent_missing/{model}/output.npz'

            dataset = AnimalMovement(mode='test', deer_id=deer_id)
            data = np.load(path)
            y_hat = data['y_hat']
            y = data['y']
            eval_mask = data['eval_mask']
            observed_mask = data['observed_mask']
            if model == 'csdi':
                imputed_samples = data['imputed_samples']
            jul = dataset.attributes['covariates'][:, 0, 0]

            # plot the result and save it
            all_target_np = y.squeeze(-2)
            all_evalpoint_np = eval_mask.squeeze(-2)
            all_observed_np = observed_mask.squeeze(-2)
            if model == 'csdi':
                samples = imputed_samples.squeeze(-2)
            else:
                samples = y_hat.squeeze(-2)[np.newaxis, ...]
            qlist = [0.05, 0.25, 0.5, 0.75, 0.95]
            quantiles_imp = []
            for q in qlist:
                tmp = np.quantile(samples, q, axis=0)
                quantiles_imp.append(tmp * (1 - all_observed_np) + all_target_np * all_observed_np)


            #######################################
            offset = 72
            B = 5
            plt.rcParams["font.size"] = 8
            markersize = 2

            # set seed 42
            rng = np.random.default_rng(42)

            # Calculate the maximum possible start index to ensure non-overlapping
            max_start = all_target_np.shape[0] - offset * B

            # Generate non-overlapping start points
            starts = np.sort(rng.choice(np.arange(0, max_start), size=B, replace=False))
            starts = np.array(
                [start * offset for start in range(len(starts))])  # Ensure non-overlapping by multiplying by offset

            fig, axes = plt.subplots(nrows=B, ncols=2, figsize=(10, 15))

            for i, start in enumerate(starts):
                end = start + offset

                for k in range(2):
                    df = pd.DataFrame(
                        {"x": jul[start:end], "val": all_target_np[start:end, k], "y": all_evalpoint_np[start:end, k]})
                    df = df[df.y != 0]
                    df2 = pd.DataFrame(
                        {"x": jul[start:end], "val": all_target_np[start:end, k], "y": all_observed_np[start:end, k]})
                    df2 = df2[df2.y != 0]
                    axes[i, k].plot(jul[start:end], quantiles_imp[2][start:end, k], color='g', linestyle='solid', label='CSDI')
                    axes[i, k].fill_between(jul[start:end], quantiles_imp[0][start:end, k], quantiles_imp[4][start:end, k],
                                         color='g', alpha=0.3)
                    axes[i, k].plot(df2.x, df2.val, color='r', marker='x', markersize=markersize, linestyle='None')
                    axes[i, k].plot(df.x, df.val, color='b', marker='o', markersize=markersize, linestyle='None')


                axes[i, 0].set_xlabel('jul')
                axes[i, 1].set_xlabel('jul')
                axes[i, 0].set_ylabel('X')
                axes[i, 1].set_ylabel('Y')
                # Use plain style for y-axis labels to avoid scientific notation
                axes[i, 0].ticklabel_format(style='plain', axis='y')
                axes[i, 1].ticklabel_format(style='plain', axis='y')


            fig.tight_layout(pad=3.0)  # Adjust the pad parameter as needed

            # save the plot
            plt.savefig(f'./results/{deer_id}_{missing_percent}_percent_missing/{model}/prediction.png', dpi=300)

            plt.close()


######################################### aug #########################################
deer_id = 5094
model = 'csdi'
path = f'./results/{deer_id}_aug/{model}/output.npz'
dataset = AnimalMovement(mode='imputation', deer_id=deer_id)
data = np.load(path)
y_hat = data['y_hat']
y = data['y']
eval_mask = data['eval_mask']
observed_mask = data['observed_mask']
if model == 'csdi':
    imputed_samples = data['imputed_samples']
jul = dataset.attributes['covariates'][:, 0, 0]

# plot the result and save it
all_target_np = y.squeeze(-2)
all_evalpoint_np = eval_mask.squeeze(-2)
all_observed_np = observed_mask.squeeze(-2)
if model == 'csdi':
    samples = imputed_samples.squeeze(-2)
else:
    samples = y_hat.squeeze(-2)[np.newaxis, ...]
qlist = [0.05, 0.25, 0.5, 0.75, 0.95]
quantiles_imp = []
for q in qlist:
    tmp = np.quantile(samples, q, axis=0)
    quantiles_imp.append(tmp * (1 - all_observed_np) + all_target_np * all_observed_np)



offset = 72
B = 5
plt.rcParams["font.size"] = 8
markersize = 2

# set seed 42
rng = np.random.default_rng(42)

# Calculate the maximum possible start index to ensure non-overlapping
max_start = all_target_np.shape[0] - offset * B

# Generate non-overlapping start points
starts = np.sort(rng.choice(np.arange(0, max_start), size=B, replace=False))
starts = np.array(
    [start * offset for start in range(len(starts))])  # Ensure non-overlapping by multiplying by offset

fig, axes = plt.subplots(nrows=B, ncols=2, figsize=(10, 15))

for i, start in enumerate(starts):
    end = start + offset

    for k in range(2):
        df = pd.DataFrame(
            {"x": jul[start:end], "val": all_target_np[start:end, k], "y": all_evalpoint_np[start:end, k]})
        df = df[df.y != 0]
        df2 = pd.DataFrame(
            {"x": jul[start:end], "val": all_target_np[start:end, k], "y": all_observed_np[start:end, k]})
        df2 = df2[df2.y != 0]
        # axes[i, k].plot(jul[start:end], quantiles_imp[2][start:end, k], color='g', linestyle='solid', label='CSDI')
        axes[i, k].fill_between(jul[start:end], quantiles_imp[0][start:end, k], quantiles_imp[4][start:end, k],
                             color='g', alpha=0.3)
        axes[i, k].plot(df2.x, df2.val, color='r', marker='x', markersize=markersize, linestyle='None')
        axes[i, k].plot(df.x, df.val, color='b', marker='o', markersize=markersize, linestyle='None')


    axes[i, 0].set_xlabel('jul')
    axes[i, 1].set_xlabel('jul')
    axes[i, 0].set_ylabel('X')
    axes[i, 1].set_ylabel('Y')
    # Use plain style for y-axis labels to avoid scientific notation
    axes[i, 0].ticklabel_format(style='plain', axis='y')
    axes[i, 1].ticklabel_format(style='plain', axis='y')


fig.tight_layout(pad=3.0)  # Adjust the pad parameter as needed

# save the plot
plt.savefig(f'./results/{deer_id}_aug/{model}/prediction.png', dpi=300)

plt.close()