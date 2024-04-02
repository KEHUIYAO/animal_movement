import numpy as np
import os

def crps_loss(Y_samples, Y_true, mask):
    B, L, K, C = Y_samples.shape
    numerator = 0
    denominator = 0

    for l in range(L):
        for k in range(K):
            for c in range(C):
                if mask[l, k, c] == 1:
                    samples = Y_samples[:, l, k, c]
                    z = Y_true[l, k, c]
                    crps = 0
                    for i in range(1, 20):
                        q = np.quantile(samples, i * 0.05)
                        if z < q:
                            indicator = 1
                        else:
                            indicator = 0
                        loss = 2 * (i * 0.05  - indicator) * (z - q) / 19
                        crps += loss
                    numerator += crps
                    denominator += np.abs(z)

    return numerator / denominator


if __name__ == '__main__':
    # make all files under Female/TagData
    deer_id_list = sorted([int(f.split('.')[0][-4:]) for f in os.listdir('Female/TagData') if f.endswith('.csv')])

    # randomly select 20% of the deer ids as testing data
    rng = np.random.RandomState(42)
    rng.shuffle(deer_id_list)
    deer_id_list = deer_id_list[int(0.8 * len(deer_id_list)):]



    model_list = ['csdi']

    crps_list = []


    for i in deer_id_list:
        for model in model_list:

            print('Running deer_id:', i, 'model:', model)

            data = np.load(f'./results/{i}/{model}/output.npz')
            y_hat = data['y_hat']
            y = data['y']
            eval_mask = data['eval_mask']
            imputed_samples = data['imputed_samples']
            print(y_hat.shape, y.shape, eval_mask.shape, imputed_samples.shape)
            crps = crps_loss(imputed_samples, y, eval_mask)
            crps_list.append(crps)

            # try:
            #     data = np.load(f'./results/{i}/{model}/output.npz')
            #     y_hat = data['y_hat']
            #     y = data['y']
            #     eval_mask = data['eval_mask']
            #     imputed_samples = data['imputed_samples']
            #     print(y_hat.shape, y.shape, eval_mask.shape, imputed_samples.shape)
            #     crps = crps_loss(imputed_samples, y, eval_mask)
            #     crps_list.append(crps)
            #
            #
            # except:
            #     pass

    print('Average CRPS:', np.mean(crps_list))