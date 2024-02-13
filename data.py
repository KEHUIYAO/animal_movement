import pandas as pd
from tsl.datasets.prototypes import PandasDataset
from tsl.datasets.prototypes.mixin import MissingValuesMixin
from tsl.ops.similarities import gaussian_kernel
import numpy as np
from scipy.spatial.distance import cdist
import os
from scipy.special import kv, gamma
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


from tsl.data.datamodule.splitters import Splitter


current_dir = os.path.dirname(os.path.abspath(__file__))
#
#
# class AnimalMovement():
#     def __init__(self):
#         df = pd.read_csv(os.path.join(current_dir,
#         'Female/Processed/deer_movement_all.csv'))
#         # df = pd.read_csv(os.path.join(current_dir,
#         #                               'Female/Processed/LowTag5016.csv'))
#         y = df.loc[:, ['X', 'Y']].values
#
#         L = y.shape[0]
#         C = y.shape[1]
#
#         # y = y.reshape(L, 1, C)
#         y = y.reshape(L, C, 1)
#
#         mask = np.ones_like(y)
#         mask[np.isnan(y)] = 0
#         mask = mask.astype(int)
#
#         # impute missing values with 0
#         y[np.isnan(y)] = 0
#
#         self.y = y
#         self.attributes = {}
#         # space_coords, time_coords = np.meshgrid(np.arange(1), np.arange(L))
#         space_coords, time_coords = np.meshgrid(np.arange(C), np.arange(L))
#         st_coords = np.stack([space_coords, time_coords], axis=-1)
#         self.attributes['st_coords'] = st_coords
#
#         p_missing = 0.2
#         rng = np.random.RandomState(42)
#         time_points_to_eval = rng.choice(L, int(p_missing * L), replace=False)
#         eval_mask = np.zeros_like(y)
#         eval_mask[time_points_to_eval, ...] = 1
#         eval_mask = eval_mask.astype(int)
#         eval_mask = eval_mask & mask
#
#         self.eval_mask = eval_mask
#         self.training_mask = mask & (1-eval_mask)
#
#         # covariates
#         X = df.loc[:, 'covariate']
#         # replace missing values with 0
#         X = X.fillna('nan')
#
#         # one-hot encoding
#         X = pd.get_dummies(X)
#         X = X.values
#         X = X.reshape(L, 1, X.shape[1])
#         # repeat along the spatial dimension
#         X = np.repeat(X, C, axis=1)
#         self.attributes['covariates'] = X
#
#
#
#     def get_splitter(self, val_len, test_len):
#         return AnimalMovementSplitter(val_len, test_len)
#
#
# class AnimalMovementSplitter(Splitter):
#
#     def __init__(self, val_len: int = None, test_len: int = None):
#         super().__init__()
#         self._val_len = val_len
#         self._test_len = test_len
#
#     def fit(self, dataset):
#         idx = np.arange(len(dataset))
#         val_len, test_len = self._val_len, self._test_len
#         if test_len < 1:
#             test_len = int(test_len * len(idx))
#         if val_len < 1:
#             val_len = int(val_len * (len(idx) - test_len))
#
#         # randomly split idx into train, val, test
#         np.random.shuffle(idx)
#         val_start = len(idx) - val_len - test_len
#         test_start = len(idx) - test_len
#
#
#         self.set_indices(idx[:val_start - dataset.samples_offset],
#                          idx[val_start:test_start - dataset.samples_offset],
#                          idx[test_start:])
#
#     @staticmethod
#     def add_argparse_args(parser):
#         parser.add_argument('--val-len', type=float or int, default=0.2)
#         parser.add_argument('--test-len', type=float or int, default=0.2)
#         return parser

class AnimalMovement():
    similarity_options = {'distance'}

    def __init__(self, num_nodes=36, seq_len=5000, seed=42):
        self.original_data = {}
        df, dist, st_coords, X = self.load(num_nodes, seq_len, seed)
        # super().__init__(dataframe=df, similarity_score="distance", attributes=dict(dist=dist, st_coords=st_coords, covariates=X))
        super().__init__(dataframe=df, similarity_score="distance", attributes=dict(dist=dist, st_coords=st_coords))
        eval_mask = np.zeros((seq_len, num_nodes))
        p_missing = 0.9
        rng = np.random.RandomState(seed)
        time_points_to_eval = rng.choice(seq_len, int(p_missing * seq_len), replace=False)
        eval_mask[time_points_to_eval, :] = 1
        self.original_data['eval_mask'] = eval_mask
        self.set_eval_mask(eval_mask)
        mask = np.ones_like(eval_mask)
        mask = mask.astype(int)
        eval_mask = eval_mask.astype(int)

        self.training_mask = mask & (1 - eval_mask)






    def load(self, num_nodes, seq_len, seed):

        rng = np.random.RandomState(seed)
        space_coords = np.random.rand(num_nodes, 2)
        dist = cdist(space_coords, space_coords)
        y = np.zeros((seq_len, num_nodes))
        X = np.zeros((seq_len, num_nodes))
        rng = np.random.RandomState(seed)

        for i in range(num_nodes):
            # Assuming seq_len and rng are defined
            noise_level = 0.2

            # Randomly determine the number of segments and their lengths
            num_segments = np.random.randint(50, 100)  # Random number of segments between 50 and 100
            segment_lengths = np.random.choice(range(20, 50), num_segments)  # Random segment lengths between 20 and 50
            segment_lengths = np.round(segment_lengths / sum(segment_lengths) * seq_len).astype(
                int)  # Adjust to match seq_len

            # Adjust the last segment to ensure the total length matches seq_len
            segment_lengths[-1] = seq_len - sum(segment_lengths[:-1])

            segments = []
            segment_types = []  # List to store segment types
            last_value = 0

            for length in segment_lengths:
                pattern_type = np.random.choice(['upward', 'downward', 'stable'])

                if pattern_type == 'upward':
                    trend = np.linspace(last_value, last_value + 4, length)
                elif pattern_type == 'downward':
                    trend = np.linspace(last_value, last_value - 4, length)
                else:  # stable
                    trend = np.full(length, last_value)

                noise = rng.normal(0, noise_level, length)
                segment = trend + noise
                segments.append(segment)
                segment_types.extend([pattern_type] * length)  # Extend the list with the segment type
                last_value = segment[-1]  # Update last value for the next segment

            # Combine segments
            time_series = np.concatenate(segments)

            y[:, i] = time_series
            # convert segment types to numeric values
            segment_types = [0 if x == 'upward' else 1 if x == 'downward' else 2 for x in segment_types]
            X[:, i] = np.array(segment_types)  # Convert segment types list to a numpy array


        self.y = y
        self.original_data['y'] = y
        # Flatten X to a 1D array
        X_flattened = X.reshape(-1, 1)

        # Apply OneHotEncoder
        encoder = OneHotEncoder(sparse=False)
        X_encoded = encoder.fit_transform(X_flattened)

        # Reshape to 3D format (L, K, C)
        L, K = X.shape
        C = X_encoded.shape[1]  # Number of unique categories
        X = X_encoded.reshape(L, K, C)



        self.original_data['X'] = X


        time_coords = np.arange(0, seq_len)

        plt.figure()
        plt.plot(time_coords, y)
        plt.show()
        df = pd.DataFrame(y)
        df.index = pd.to_datetime(df.index)

        space_coords, time_coords = np.meshgrid(np.arange(df.shape[1]), np.arange(df.shape[0]))
        st_coords = np.stack([space_coords, time_coords], axis=-1)

        return df, dist, st_coords, X



    def compute_similarity(self, method: str, **kwargs):
        if method == "distance":
            theta = np.std(self.dist)
            return gaussian_kernel(self.dist, theta=theta)

    def matern_covariance(self, x1, x2, length_scale=1.0, nu=1.5, sigma=1.0):
        dist = np.linalg.norm(x1 - x2)
        if dist == 0:
            return sigma ** 2
        coeff1 = (2 ** (1 - nu)) / gamma(nu)
        coeff2 = (np.sqrt(2 * nu) * dist) / length_scale
        return sigma ** 2 * coeff1 * (coeff2 ** nu) * kv(nu, coeff2)



if __name__ == '__main__':
    from tsl.ops.imputation import add_missing_values

    num_nodes, seq_len = 5, 4000
    dataset = Cluster(num_nodes, seq_len)
    add_missing_values(dataset, p_fault=0, p_noise=0.25, min_seq=12,
                       max_seq=12 * 4, seed=56789)

    print(dataset.training_mask.shape)
    print(dataset.eval_mask.shape)

    adj = dataset.get_connectivity(threshold=0.1,
                                   include_self=False)

    print(adj)


if __name__ == '__main__':
    dataset = AnimalMovement()

