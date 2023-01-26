from annoy import AnnoyIndex
import math
import argparse
import wandb
from tqdm import tqdm
import pickle
import gc
import numpy as np
import polars as pl
import os

class CFG:
    input_train_dir = "20230121"
    input_test_dir = "20230121"
    embedding_size = 16
    wandb = True


def read_ranker_train_dataset(type):
    path = f"./input/lgbm_dataset/{CFG.input_train_dir}/{type}/*"
    df = pl.read_parquet(path, columns=["session", "aid"]).to_pandas()
    for c in ["session", "aid"]:
        df[c] = df[c].astype("int32")
    return df


def read_ranker_test_dataset():
    path = f"./input/lgbm_dataset_test/{CFG.input_test_dir}/*"
    df = pl.read_parquet(path, columns=["session", "aid"]).to_pandas()
    for c in ["session", "aid"]:
        df[c] = df[c].astype("int32")
    return df


def read_test_interactions():
    path = "./input/otto-chunk-data-inparquet-format/test_parquet/*"
    df = pl.read_parquet(path, columns=["session", "aid"]).to_pandas()
    for c in ["session", "aid"]:
        df[c] = df[c].astype("int32")
    session_aids = df.groupby("session")["aid"].apply(list)
    del df
    gc.collect()
    return session_aids


def read_train_interactions():
    path = "./input/otto-validation/test_parquet/*"
    df = pl.read_parquet(path, columns=["session", "aid"]).to_pandas()
    for c in ["session", "aid"]:
        df[c] = df[c].astype("int32")
    session_aids = df.groupby("session")["aid"].apply(list)
    del df
    gc.collect()
    return session_aids


def read_item_embeddings():
    path = "./input/lightfm/components16/mapped_item_embeddings.pkl"
    df = pickle.load(open(path, "rb"))
    return df


def dump_pickle(path, o):
    with open(path, "wb") as f:
        pickle.dump(o, f)


def calc_train_score(index, output_dir):
    print("calc_train_score start")
    session_aids = read_train_interactions()

    for t in ["clicks", "carts", "orders"]:
        print(f"{t} start")
        candidates = read_ranker_train_dataset(type=t)
        candidates_session_aids = candidates.groupby("session")["aid"].apply(list).to_frame().reset_index()
        for c in ["score_mean", "score_std", "score_max", "score_min", "score_length"]:
            candidates_session_aids[c] = np.nan
            candidates_session_aids[c] = candidates_session_aids[c].astype('object')
        del candidates
        gc.collect()

        candidates_session_aids = scoring(candidates_session_aids, session_aids, index)
        candidates_session_aids = candidates_session_aids.explode(["aid", "score_mean", "score_std", "score_max", "score_min", "score_length"], ignore_index=True)
        candidates_session_aids.to_parquet(os.path.join(output_dir, f"train_score_{t}.parquet"))
        del candidates_session_aids
        gc.collect()
    del session_aids
    gc.collect()


def calc_test_score(index, output_dir):
    print("calc_test_score start")
    session_aids = read_test_interactions()

    candidates = read_ranker_test_dataset()
    candidates_session_aids = candidates.groupby("session")["aid"].apply(list).to_frame().reset_index()
    for c in ["score_mean", "score_std", "score_max", "score_min", "score_length"]:
        candidates_session_aids[c] = np.nan
        candidates_session_aids[c] = candidates_session_aids[c].astype('object')
    del candidates
    gc.collect()

    candidates_session_aids = scoring(candidates_session_aids, session_aids, index)
    del session_aids, index
    gc.collect()
    candidates_session_aids = candidates_session_aids.explode(["aid", "score_mean", "score_std", "score_max", "score_min", "score_length"], ignore_index=True)
    candidates_session_aids.to_parquet(os.path.join(output_dir, "test_score.parquet"))
    del candidates_session_aids
    gc.collect()


def scoring(candidates_session_aids, session_aids, index):
    total_iter = len(candidates_session_aids["session"].values)
    for i, session in enumerate(candidates_session_aids["session"].values):
        if i % 100000 == 0:
            print(f"{i}/{total_iter}")
        target_indices = candidates_session_aids.loc[candidates_session_aids["session"] == session].index.values
        assert len(target_indices) == 1
        target_index = target_indices[0]
        score_mean = []
        score_std = []
        score_max = []
        score_min = []
        score_length = []
        for candidate_aid in candidates_session_aids.loc[target_index, "aid"]:
            distances = []
            for session_aid in session_aids[session]:
                dis = index.get_distance(session_aid, candidate_aid)
                distances.append(dis)
            score_mean.append(np.mean(distances))
            score_std.append(np.std(distances))
            score_max.append(np.max(distances))
            score_min.append(np.min(distances))
            score_length.append(len(distances))
        candidates_session_aids.at[target_index, "score_mean"] = score_mean
        candidates_session_aids.at[target_index, "score_std"] = score_std
        candidates_session_aids.at[target_index, "score_max"] = score_max
        candidates_session_aids.at[target_index, "score_min"] = score_min
        candidates_session_aids.at[target_index, "score_length"] = score_length
    return candidates_session_aids


def main(type):
    run_name = None
    if CFG.wandb:
        wandb.init(project="kaggle-otto", job_type="scoring", group=type)
        run_name = wandb.run.name
    if run_name is not None:
        output_dir = os.path.join("output/lightfm_score", run_name)
    else:
        output_dir = "output/lightfm_score"
    os.makedirs(output_dir, exist_ok=True)

    embeddings = read_item_embeddings()
    index = AnnoyIndex(16, "angular")
    for aid in embeddings.keys():
        index.add_item(aid, embeddings[aid])
    index.build(100)
    print("AnnoyIndex build")
    del embeddings
    gc.collect()
    if type == "train":
        calc_train_score(index, output_dir)
    else:
        calc_test_score(index, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str)
    args = parser.parse_args()
    main(args.type)
