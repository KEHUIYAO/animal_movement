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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio

from tsl.data.datamodule.splitters import Splitter
from datetime import datetime, timedelta


current_dir = os.path.dirname(os.path.abspath(__file__))


class AnimalMovement():
    def __init__(self, mode='train', deer_id=5016):
        # df = pd.read_csv(os.path.join(current_dir,
        # 'Female/Processed/deer_movement_all.csv'))
        num = deer_id
        df = self.load_data(num)

        y = df.loc[:, ['X', 'Y']].values

        # how many non-missing rows
        non_missing = np.sum(~np.isnan(y), axis=0)

        # remove outliers using IQR
        # set those values that are outside of the range to be nan
        y = y.astype(float)
        Q1 = np.nanpercentile(y, 25, axis=0)
        Q3 = np.nanpercentile(y, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR


        for i in range(y.shape[1]):
            y[(y[:, i] < lower_bound[i]) | (y[:, i] > upper_bound[i]), :] = np.nan


        # count how many outliers are removed
        removed = non_missing - np.sum(~np.isnan(y), axis=0)


        print('Removed outliers:', removed[0])

        # write the print message to a file
        with open(f'results/{num}/outlier_removed.txt', 'w') as f:
            f.write(f'Removed outliers: {removed[0]}\n')
            f.write(f'Original data shape: {non_missing[0]}\n')
            f.write(f'After removing outliers: {np.sum(~np.isnan(y), axis=0)[0]}\n')


        fig, axs = plt.subplots(2)
        axs[0].plot(y[:, 0], 'o', markersize=1)
        axs[1].plot(y[:, 1], 'o', markersize=1)
        plt.show()

        # create a folder called result to save the figure
        if not os.path.exists(f'results/{num}'):
            os.makedirs(f'results/{num}')

        # save fig to file, file name is the deer id
        fig.savefig(f'results/{num}/outlier_removed.png')


        L = y.shape[0]
        C = y.shape[1]
        y = y.reshape(L, 1, C)

        # covariates
        X = df.loc[:, ['month', 'day', 'hour', 'covariate']]
        # replace missing values with 0
        X = X.fillna(0)

        # one-hot encoding for covariates
        covariates = X['covariate']

        covariates = pd.get_dummies(covariates)

        # normalize month, day, and hour to [0, 1]
        month = X['month'] / 12
        day = X['day'] / 31
        hour = X['hour'] / 24

        X = pd.concat([month, day, hour, covariates], axis=1)
        # X = pd.concat([month, day, hour], axis=1)

        X = X.values
        X = X.reshape(L, 1, X.shape[1])


        # randomly set 20% of data to be missing as test data
        mask = np.ones_like(y)
        mask[np.isnan(y)] = 0
        mask = mask.astype(int)
        p_missing = 0.2
        rng = np.random.RandomState(42)
        time_points_to_eval = rng.choice(L, int(p_missing * L), replace=False)
        eval_mask = np.zeros_like(y)
        eval_mask[time_points_to_eval, ...] = 1
        eval_mask = eval_mask.astype(int)
        eval_mask = eval_mask & mask


        if mode == 'train':
            y[time_points_to_eval, :] = np.nan
            X[time_points_to_eval, 3:] = 0

            # randomly set 20% of data to be missing as val data
            mask = np.ones_like(y)
            mask[np.isnan(y)] = 0
            mask = mask.astype(int)
            # impute missing values with 0
            y[np.isnan(y)] = 0
            time_points_to_eval = rng.choice(L, int(p_missing * L), replace=False)
            eval_mask = np.zeros_like(y)
            eval_mask[time_points_to_eval, ...] = 1
            eval_mask = eval_mask.astype(int)
            eval_mask = eval_mask & mask
        else:
            # impute missing values with 0
            y[np.isnan(y)] = 0

        self.eval_mask = eval_mask
        self.training_mask = mask & (1 - eval_mask)
        self.y = y
        self.attributes = {}
        space_coords, time_coords = np.meshgrid(np.arange(1), np.arange(L))
        st_coords = np.stack([space_coords, time_coords], axis=-1)
        self.attributes['st_coords'] = st_coords

        X[time_points_to_eval, 3:] = 0
        self.attributes['covariates'] = X

    def load_data(self, num):

        # if the processed file is already existed, load it
        if os.path.exists(f'./Female/Processed/{num}.csv'):
            return pd.read_csv('Female/Processed/' + str(num) + '.csv')

        # Load the dataset
        file_path = 'Female/TagData/LowTag' + str(num) + '.csv'
        deer_data = pd.read_csv(file_path)

        # load the covariate .tif file
        covariate_file_path = 'Female/NLCDClip/LowTag' + str(num) + 'NLCDclip.tif'
        covariate_file = rasterio.open(covariate_file_path)

        row, col = covariate_file.index(deer_data['X'], deer_data['Y'])
        # Assuming row and col are lists of the same length
        values = []
        roi = covariate_file.read(1)
        for r, c in zip(row, col):
            if r < roi.shape[0] and c < roi.shape[1]:
                values.append(covariate_file.read(1)[r, c])
            else:
                values.append(None)

        deer_data['covariate'] = values

        start_time, end_time = deer_data['jul'].min(), deer_data['jul'].max()

        # calculate the smallest time interval
        smallest_time_interval = deer_data['jul'].diff().min()

        # # index of the smallest time interval
        # idx = deer_data['jul'].diff().idxmin()


        time_interval = 0.08
        tolerance = 0.04

        T_values = np.arange(start_time, end_time, time_interval)
        df = pd.DataFrame(T_values, columns=['T'])



        # Function to find nearest row within tolerance
        def find_nearest_row_within_tolerance(value, tolerance, dataframe, column_name):
            nearest_idx = (dataframe[column_name] - value).abs().argsort()[:1]
            nearest_value = dataframe[column_name].iloc[nearest_idx].values[0]
            if abs(nearest_value - value) <= tolerance:
                return dataframe.iloc[nearest_idx]
            return pd.DataFrame(columns=dataframe.columns)

        # Initialize a list to store dictionaries
        data_list = []

        # Merge data
        for t_value in df['T']:
            matched_row = find_nearest_row_within_tolerance(t_value, tolerance, deer_data, 'jul')
            if not matched_row.empty:
                row_data = {'T': t_value, **matched_row.iloc[0].to_dict()}
            else:
                row_data = {'T': t_value, **{col: np.nan for col in deer_data.columns}}
            data_list.append(row_data)

        # Create DataFrame from list of dictionaries
        df_matched = pd.DataFrame(data_list)

        base_date = datetime(2017, 1, 1, 0, 0, 0)
        df_matched['date'] = [base_date + timedelta(days=x) for x in df_matched['T']]
        df_matched['month'] = [x.month for x in df_matched['date']]
        df_matched['day'] = [x.day for x in df_matched['date']]
        df_matched['hour'] = [x.hour for x in df_matched['date']]

        fig, axs = plt.subplots(2)
        axs[0].plot(df_matched['X'], 'o', markersize=1)
        axs[1].plot(df_matched['Y'], 'o', markersize=1)
        plt.show()

        # create a folder called result to save the figure
        if not os.path.exists(f'results/{num}'):
            os.makedirs(f'results/{num}')

        # save fig to file, file name is the deer id
        fig.savefig(f'results/{num}/original.png')

        # save the dataframe to a csv file
        # if the folder is not existed, create it
        if not os.path.exists(f'Female/Processed'):
            os.makedirs(f'Female/Processed')

        df_matched.to_csv(f'Female/Processed/{num}.csv', index=False)

        return df_matched

    def get_splitter(self, val_len, test_len):
        return AnimalMovementSplitter(val_len, test_len)


class AnimalMovementSplitter(Splitter):

    def __init__(self, val_len: int = None, test_len: int = None):
        super().__init__()
        self._val_len = val_len
        self._test_len = test_len

    def fit(self, dataset):
        idx = np.arange(len(dataset))
        val_len, test_len = self._val_len, self._test_len
        if test_len < 1:
            test_len = int(test_len * len(idx))
        if val_len < 1:
            val_len = int(val_len * (len(idx) - test_len))

        # randomly split idx into train, val, test
        np.random.shuffle(idx)
        val_start = len(idx) - val_len - test_len
        test_start = len(idx) - test_len


        # self.set_indices(idx[:val_start - dataset.samples_offset],
        #                  idx[val_start:test_start - dataset.samples_offset],
        #                  idx[test_start:])
        self.set_indices(idx[:val_start],
                         idx[val_start:test_start],
                         idx[test_start:])

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--val-len', type=float or int, default=0.2)
        parser.add_argument('--test-len', type=float or int, default=0.2)
        return parser


if __name__ == '__main__':
    dataset = AnimalMovement()




#
# class AnimalMovement():
#     def __init__(self, num_nodes=36, seq_len=5000, seed=42):
#         self.attributes = {}
#         self.original_data = {}
#         self.load(num_nodes, seq_len, seed)
#
#         eval_mask = np.zeros((seq_len, num_nodes))
#         p_missing = 0.9
#         rng = np.random.RandomState(seed)
#         time_points_to_eval = rng.choice(seq_len, int(p_missing * seq_len), replace=False)
#         eval_mask[time_points_to_eval, :] = 1
#         self.original_data['eval_mask'] = eval_mask
#         mask = np.ones_like(eval_mask)
#         mask = mask.astype(int)
#         eval_mask = eval_mask.astype(int)
#
#         self.eval_mask = eval_mask
#         self.training_mask = mask & (1 - eval_mask)
#
#
#
#
#
#
#     def load(self, num_nodes, seq_len, seed):
#
#         rng = np.random.RandomState(seed)
#         space_coords = np.random.rand(num_nodes, 2)
#         dist = cdist(space_coords, space_coords)
#         y = np.zeros((seq_len, num_nodes))
#         X = np.zeros((seq_len, num_nodes))
#         rng = np.random.RandomState(seed)
#
#         for i in range(num_nodes):
#             # Assuming seq_len and rng are defined
#             noise_level = 0.2
#
#             # Randomly determine the number of segments and their lengths
#             num_segments = np.random.randint(50, 100)  # Random number of segments between 50 and 100
#             segment_lengths = np.random.choice(range(20, 50), num_segments)  # Random segment lengths between 20 and 50
#             segment_lengths = np.round(segment_lengths / sum(segment_lengths) * seq_len).astype(
#                 int)  # Adjust to match seq_len
#
#             # Adjust the last segment to ensure the total length matches seq_len
#             segment_lengths[-1] = seq_len - sum(segment_lengths[:-1])
#
#             segments = []
#             segment_types = []  # List to store segment types
#             last_value = 0
#
#             for length in segment_lengths:
#                 pattern_type = np.random.choice(['upward', 'downward', 'stable'])
#
#                 if pattern_type == 'upward':
#                     trend = np.linspace(last_value, last_value + 4, length)
#                 elif pattern_type == 'downward':
#                     trend = np.linspace(last_value, last_value - 4, length)
#                 else:  # stable
#                     trend = np.full(length, last_value)
#
#                 noise = rng.normal(0, noise_level, length)
#                 segment = trend + noise
#                 segments.append(segment)
#                 segment_types.extend([pattern_type] * length)  # Extend the list with the segment type
#                 last_value = segment[-1]  # Update last value for the next segment
#
#             # Combine segments
#             time_series = np.concatenate(segments)
#
#             y[:, i] = time_series
#             # convert segment types to numeric values
#             segment_types = [0 if x == 'upward' else 1 if x == 'downward' else 2 for x in segment_types]
#             X[:, i] = np.array(segment_types)  # Convert segment types list to a numpy array
#
#
#         self.y = y
#         self.original_data['y'] = y
#         # Flatten X to a 1D array
#         X_flattened = X.reshape(-1, 1)
#
#         # Apply OneHotEncoder
#         encoder = OneHotEncoder(sparse=False)
#         X_encoded = encoder.fit_transform(X_flattened)
#
#         # Reshape to 3D format (L, K, C)
#         L, K = X.shape
#         C = X_encoded.shape[1]  # Number of unique categories
#         X = X_encoded.reshape(L, K, C)
#
#
#
#         self.original_data['X'] = X
#         self.attributes['covariates'] = X
#
#
#         time_coords = np.arange(0, seq_len)
#
#         plt.figure()
#         plt.plot(time_coords, y)
#         plt.show()
#         df = pd.DataFrame(y)
#         df.index = pd.to_datetime(df.index)
#
#         space_coords, time_coords = np.meshgrid(np.arange(df.shape[1]), np.arange(df.shape[0]))
#         st_coords = np.stack([space_coords, time_coords], axis=-1)
#
#         self.attributes['st_coords'] = st_coords
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
#
#
#
#

# if __name__ == '__main__':
#     from tsl.ops.imputation import add_missing_values
#
#     num_nodes, seq_len = 5, 4000
#     dataset = AnimalMovement(num_nodes, seq_len)
#     add_missing_values(dataset, p_fault=0, p_noise=0.25, min_seq=12,
#                        max_seq=12 * 4, seed=56789)
#
#     print(dataset.training_mask.shape)
#     print(dataset.eval_mask.shape)
#
#     adj = dataset.get_connectivity(threshold=0.1,
#                                    include_self=False)
#
#     print(adj)