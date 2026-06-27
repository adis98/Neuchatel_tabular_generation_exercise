from sqlite3 import Row
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os
import json
from copy import deepcopy

from tqdm import tqdm
from utils.download_and_process_datasets import CATCOLS


def get_sdmetrics_metadata(df):
    metadata = {"columns": {}}

    for col, dtype in df.dtypes.items():
        if pd.api.types.is_numeric_dtype(dtype):
            sdtype = "numerical"
        else:
            sdtype = "categorical"

        metadata["columns"][col] = {"sdtype": sdtype}

    return metadata


"""Preprocessing"""


class Preprocessor:
    def __init__(self, dataname):
        self.dataset = dataname
        self.OneHotEncoder = OneHotEncoder(sparse_output=False)
        self.df = pd.read_csv(f"datasets/{dataname}/data.csv")
        self.df_train = pd.read_csv(f"datasets/{dataname}/train.csv")
        self.df_test = pd.read_csv(f"datasets/{dataname}/test.csv")
        self.cat_cols = CATCOLS[dataname]
        self.num_cols = [col for col in self.df.columns if col not in CATCOLS[dataname]]
        self.cat_col_indices = self.df.columns.get_indexer(self.cat_cols)
        self.num_col_indices = self.df.columns.get_indexer(self.num_cols)
        self.OneHotEncoder.fit(self.df[self.cat_cols])
        self.numeric_mean = np.mean(self.df_train[self.num_cols].values, axis=0)
        self.numeric_std = np.std(self.df_train[self.num_cols].values, axis=0)

    # converts a dataframe to a one-hot-encoded numpy array
    def encodeDfToNp(self, df):
        cats = self.OneHotEncoder.transform(df[self.cat_cols])
        nums = df[self.num_cols].values
        return np.concatenate((nums, cats), axis=1)

    def standardize_np(self, arr):  # assume this array has format (nums, cats)
        temp = np.divide(arr[:, : len(self.num_cols)] - self.numeric_mean, self.numeric_std,
                         out=np.zeros_like(arr[:, : len(self.num_cols)]), where=self.numeric_std != 0)
        return np.concatenate((temp, arr[:, len(self.num_cols):]), axis=1)

    def destandardize_np(self, arr):
        temp = (arr[:, : len(self.num_cols)] * self.numeric_std) + self.numeric_mean
        return np.concatenate((temp, arr[:, len(self.num_cols):]), axis=1)

    # converts a numpy array back into the dataframe form
    def decodeNpToDf(self, arr):
        cats_decoded = self.OneHotEncoder.inverse_transform(arr[:, len(self.num_cols):])
        ordered = np.concatenate((arr[:, : len(self.num_cols)], cats_decoded), axis=1)
        reordered = deepcopy(ordered)
        reordered[:, self.num_col_indices] = ordered[:, : len(self.num_cols)]
        reordered[:, self.cat_col_indices] = ordered[:, len(self.num_cols):]
        df = pd.DataFrame(reordered, columns=self.df.columns)
        df = df.astype(self.df_test.dtypes.to_dict())
        return df
