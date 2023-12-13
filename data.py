import pandas as pd
from tsl.datasets.prototypes import PandasDataset
from tsl.datasets.prototypes.mixin import MissingValuesMixin
from tsl.ops.similarities import gaussian_kernel
import numpy as np
from scipy.spatial.distance import cdist
import os

from tsl.data.datamodule.splitters import Splitter


current_dir = os.path.dirname(os.path.abspath(__file__))


class AnimalMovement():
    def __init__(self):
        df = pd.read_csv(os.path.join(current_dir,
        'Female/Processed/deer_movement_all.csv'))
        # df = pd.read_csv(os.path.join(current_dir,
        #                               'Female/Processed/LowTag5016.csv'))
        y = df.loc[:, ['X', 'Y']].values

        L = y.shape[0]
        C = y.shape[1]

        # y = y.reshape(L, 1, C)
        y = y.reshape(L, C, 1)

        mask = np.ones_like(y)
        mask[np.isnan(y)] = 0
        mask = mask.astype(int)

        # impute missing values with 0
        y[np.isnan(y)] = 0

        self.y = y
        self.attributes = {}
        # space_coords, time_coords = np.meshgrid(np.arange(1), np.arange(L))
        space_coords, time_coords = np.meshgrid(np.arange(C), np.arange(L))
        st_coords = np.stack([space_coords, time_coords], axis=-1)
        self.attributes['st_coords'] = st_coords

        p_missing = 0.2
        rng = np.random.RandomState(42)
        time_points_to_eval = rng.choice(L, int(p_missing * L), replace=False)
        eval_mask = np.zeros_like(y)
        eval_mask[time_points_to_eval, ...] = 1
        eval_mask = eval_mask.astype(int)
        eval_mask = eval_mask & mask

        self.eval_mask = eval_mask
        self.training_mask = mask & (1-eval_mask)

        # # covariates
        # X = df.loc[:, 'covariate']
        # # replace missing values with 0
        # X = X.fillna('nan')
        #
        # # one-hot encoding
        # X = pd.get_dummies(X)
        # X = X.values
        # X = X.reshape(L, 1, X.shape[1])
        # self.attributes['covariates'] = X



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


        self.set_indices(idx[:val_start - dataset.samples_offset],
                         idx[val_start:test_start - dataset.samples_offset],
                         idx[test_start:])

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--val-len', type=float or int, default=0.2)
        parser.add_argument('--test-len', type=float or int, default=0.2)
        return parser



if __name__ == '__main__':
    dataset = AnimalMovement()

