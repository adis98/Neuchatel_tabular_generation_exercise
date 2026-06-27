import json
import os
import zipfile
from urllib import request
import numpy as np
import pandas as pd

DATA_DIR = "datasets"

NAME_URL_DICT_UCI = {
    "adult": "https://archive.ics.uci.edu/static/public/2/adult.zip"
}

CATCOLS = {
    "adult": [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        "salary",
    ]
}


def fetchColumnTypes(dataname, df):
    setAll = set(list(df.columns))
    cat_cols = CATCOLS[dataname]
    setCat = set(cat_cols)
    setNum = setAll.difference(setCat)
    return setCat, setNum


def unzip_file(zip_filepath, dest_path):
    with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
        zip_ref.extractall(dest_path)


def download_from_uci(name):
    print(f"Start processing dataset {name} from UCI.")
    save_dir = f"{DATA_DIR}/{name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        url = NAME_URL_DICT_UCI[name]
        request.urlretrieve(url, f"{save_dir}/{name}.zip")
        print(f"Finish downloading dataset from {url}, data has been saved to {save_dir}.")
        unzip_file(f"{save_dir}/{name}.zip", save_dir)
        print(f"Finish unzipping {name}.")
    else:
        print("Already downloaded.")


def process_adult():
    path = f"{DATA_DIR}/adult/adult.data"
    save_path = f"{DATA_DIR}/adult/data.csv"
    data_df = pd.read_csv(
        path,
        header=None,
        names=[
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "salary",
        ],
    )
    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(save_path, index=False)


def train_test_split(dataname, ratio=0.7):
    data_dir = f"{DATA_DIR}/{dataname}"
    path = f"{DATA_DIR}/{dataname}/data.csv"
    data_df = pd.read_csv(path)
    cat_cols, num_cols = fetchColumnTypes(dataname, data_df)
    metadata = {"cat_cols": list(cat_cols), "num_cols": list(num_cols)}
    with open(f"{DATA_DIR}/{dataname}/info.json", "w") as f:
        json.dump(metadata, f, indent=4)
    total_num = data_df.shape[0]

    num_train = int(total_num * ratio)
    num_test = total_num - num_train
    seed = 42

    np.random.seed(seed)
    shuffled_ids = np.random.permutation(np.arange(total_num))

    train_idx = shuffled_ids[:num_train]
    test_idx = shuffled_ids[-num_test:]

    train_df = data_df.loc[train_idx]
    test_df = data_df.loc[test_idx]

    train_path = f"{data_dir}/train.csv"
    test_path = f"{data_dir}/test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Spliting Trainig and Testing data for {dataname} is done.")
    print(f"Training data shape: {train_df.shape}, Testing data shape: {test_df.shape}")
    print(f"Training data saved at {train_path}, Testing data saved at {test_path}.")


if __name__ == "__main__":
    # Downloading dataset
    for name in NAME_URL_DICT_UCI.keys():
        download_from_uci(name)
        globals()[f"process_{name}"]()

    for name in NAME_URL_DICT_UCI.keys():
        eval(f"process_{name}()")
        train_test_split(name, ratio=0.7)
