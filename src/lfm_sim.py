import pandas as pd
import math
from tqdm import tqdm
import torch
import pickle
import gc
import numpy as np
import polars as pl
import os

class CFG:
    input_train_dir = "20230121"
    input_test_dir = "20230121"
    embedding_size = 16


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def read_ranker_train_dataset(type):
    path = f"./input/lgbm_dataset/{CFG.input_train_dir}/{type}/*"
    df = pl.read_parquet(path, columns=["session", "aid"]).to_pandas()
    return df


def read_ranker_test_dataset():
    path = f"./input/lgbm_dataset_test/{CFG.input_test_dir}/*"
    df = pl.read_parquet(path, columns=["session", "aid"]).to_pandas()
    return df


def read_interactions():
    path = "./input/otto-chunk-data-inparquet-format/*_parquet/*"
    df = pl.read_parquet(path, columns=["session", "aid"]).to_pandas()
    session_aids = df.groupby("session")["aid"].apply(list)
    return session_aids


def read_test_interactions():
    path = "./input/otto-chunk-data-inparquet-format/test_parquet/*"
    df = pl.read_parquet(path, columns=["session", "aid"]).to_pandas()
    session_aids = df.groupby("session")["aid"].apply(list)
    return session_aids


def read_item_embeddings():
    path = "./input/lightfm/components16/mapped_item_embeddings.pkl"
    df = pickle.load(open(path, "rb"))
    return df


def dump_pickle(path, o):
    with open(path, "wb") as f:
        pickle.dump(o, f)


def cosine_similarity(candidates_embeddings, interaction_embeddings):
    Z = candidates_embeddings.to(torch.device("cuda"))
    B = interaction_embeddings.T.to(torch.device("cuda"))
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)
    return ((Z @ B) / (Z_norm @ B_norm)).T


def calc_train_similarity(output_dir):
    print("calc_train_similarity start")
    for t in ["clicks", "carts", "orders"]:
        print(t)
        candidates = read_ranker_train_dataset(type="clicks")
        candidates_session_aids = candidates.groupby("session")["aid"].apply(list)
        for c in ["sim_mean", "sim_sum"]:
            candidates[c] = np.nan
        session_aids = read_interactions()
        embeddings = read_item_embeddings()

        print(f"candidates session size: {len(candidates_session_aids)}")
        for session in tqdm(candidates_session_aids.index.tolist()):
            interaction_embeddings = torch.concat([torch.tensor(embeddings[aid]) for aid in list(set(session_aids[session]))]).reshape(-1, CFG.embedding_size)
            candidates_embeddings = torch.concat([torch.tensor(embeddings[aid]) for aid in list(set(candidates_session_aids[session]))]).reshape(-1, CFG.embedding_size)
            sim = cosine_similarity(candidates_embeddings, interaction_embeddings)
            candidates.loc[candidates["session"] == session, "sim_mean"] = sim.mean(axis=0).tolist()
            candidates.loc[candidates["session"] == session, "sim_sum"] = sim.sum(axis=0).tolist()
        dump_pickle(os.path.join(output_dir, f"candidates_train_{t}.pkl"), candidates)


def calc_test_similarity(output_dir):
    print("calc_test_similarity start")
    candidates = read_ranker_test_dataset()
    candidates_session_aids = candidates.groupby("session")["aid"].apply(list)
    for c in ["sim_mean", "sim_sum"]:
        candidates[c] = np.nan
    session_aids = read_interactions()
    embeddings = read_item_embeddings()

    print(f"candidates session size: {len(candidates_session_aids)}")
    for session in tqdm(candidates_session_aids.index.tolist()):
        interaction_embeddings = torch.concat([torch.tensor(embeddings[aid]) for aid in list(set(session_aids[session]))]).reshape(-1, CFG.embedding_size)
        candidates_embeddings = torch.concat([torch.tensor(embeddings[aid]) for aid in list(set(candidates_session_aids[session]))]).reshape(-1, CFG.embedding_size)
        sim = cosine_similarity(candidates_embeddings, interaction_embeddings)
        candidates.loc[candidates["session"] == session, "sim_mean"] = sim.mean(axis=0).tolist()
        candidates.loc[candidates["session"] == session, "sim_sum"] = sim.sum(axis=0).tolist()
    dump_pickle(os.path.join(output_dir, "candidates_test.pkl"), candidates)


def main(output_dir):
    session_aids = read_test_interactions()
    embeddings = read_item_embeddings()
    session_embeddings = {}
    for session in tqdm(session_aids.index.tolist()):
        session_embeddings[session] = torch.concat([torch.tensor(embeddings[i]) for i in dict.fromkeys(session_aids[session][::-1])[:5]]).reshape(-1, CFG.embedding_size).mean(axis=0).tolist()
    dump_pickle(os.path.join(output_dir, "session_embeddings.pkl"), session_embeddings)
    print("session_embeddings created")
    # session_embeddings = pickle.load(open(os.path.join(output_dir, "session_embeddings.pkl"), "rb"))
    embeddings_tensor = torch.tensor([*embeddings.values()])
    dump_pickle(os.path.join(output_dir, "embeddings_keys.pkl"), [*embeddings.keys()])
    session_embeddings_tensor = torch.tensor([*session_embeddings.values()])
    del embeddings, session_embeddings
    gc.collect()
    print("cosine_similarity start")
    sim = cosine_similarity(embeddings_tensor, session_embeddings_tensor)
    dump_pickle(os.path.join(output_dir, "sims.pkl"), sim)
    # batch_size = 50
    # sims_dir = os.path.join(output_dir, "sims")
    # os.makedirs(sims_dir, exist_ok=True)
    # for i in tqdm(range(math.ceil(len(embeddings_tensor) / batch_size))):
    #     sim = cosine_similarity(embeddings_tensor[i * batch_size:(i + 1) * batch_size], session_embeddings_tensor)
    #     dump_pickle(os.path.join(sims_dir, f"sims{i}.pkl"), sim)
    #     del sim
    #     gc.collect()


if __name__ == "__main__":
    output_dir = "output/lightfm_sim"
    os.makedirs(output_dir, exist_ok=True)
    main(output_dir=output_dir)
