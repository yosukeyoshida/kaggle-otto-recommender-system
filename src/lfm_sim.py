import pandas as pd
import pickle
import gc
import math
import glob
import numpy as np
import polars as pl
import os

class CFG:
    input_train_dir = "20230119_4"
    input_test_dir = "20230119_4"


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def read_ranker_dataset():
    train_list = []
    for type in ["clicks", "carts", "orders"]:
        print(type)
        path = f"./input/lgbm_dataset/{CFG.input_train_dir}/{type}/*"
        train = pl.read_parquet(path).to_pandas()
        train = train[["session", "aid"]]
        train_list.append(train)
        del train
        gc.collect()
    train = pd.concat(train_list, axis=0, ignore_index=True)
    del train_list
    gc.collect()
    path = f"./input/lgbm_dataset_test/{CFG.input_test_dir}/*"
    test = pl.read_parquet(path).to_pandas()
    test = test[["session", "aid"]]
    df = pd.concat([train, test], axis=0)
    del train, test
    gc.collect()
    df.drop_duplicates(inplace=True, ignore_index=True)
    return df


def read_interactions():
    train_file_path = "./input/otto-chunk-data-inparquet-format/*_parquet/*"
    train = pl.read_parquet(train_file_path).to_pandas()
    session_aids = train.groupby("session")["aid"].apply(list)
    return session_aids


def read_item_embeddings():
    path = f"./input/lightfm/mapped_item_embeddings.pkl"
    df = pickle.load(open(path, "rb"))
    return df


def dump_pickle(path, o):
    with open(path, "wb") as f:
        pickle.dump(o, f)


def main(output_dir):
    print("read ranker dataset")
    train = read_ranker_dataset()
    print("read interactions")
    interactions = read_interactions()
    print("read item embeddings")
    item_embeddings = read_item_embeddings()
    print(f"train shape: {train.shape}")
    for i in range(len(train)):
        if i % 100000 == 0:
            print(i)
        _train = train.loc[i]
        session = _train["session"]
        candidates_aid = _train["aid"]
        aids = interactions.loc[session]
        sims = []
        for interaction_aid in aids:
            sim = cos_sim(item_embeddings[candidates_aid], item_embeddings[interaction_aid])
            sims.append(sim)
        train.loc[i, "sim_max"] = np.max(sims)
        train.loc[i, "sim_min"] = np.min(sims)
        train.loc[i, "sim_mean"] = np.mean(sims)
        train.loc[i, "sim_sum"] = np.sum(sims)
    dump_pickle(os.path.join(output_dir, "similarity.pkl"), train)


if __name__ == "__main__":
    output_dir = "output/lightfm"
    main(output_dir=output_dir)
