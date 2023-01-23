from lightfm import LightFM
import os
from util import dump_pickle
from lightfm.data import Dataset
import glob
import pandas as pd


class CFG:
    n_epochs = 100

def read_files(path):
    dfs = []
    for file in glob.glob(path):
        df = pd.read_parquet(file)
        dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)


def main(output_dir, **kwargs):
    train_file_path = "./input/otto-chunk-data-inparquet-format/*_parquet/*"
    train = read_files(train_file_path)
    print(train.shape)
    session = train["session"].unique()
    aid = train["aid"].unique()
    print("dataset fit start")
    dataset = Dataset()
    dataset.fit(users=session, items=aid)
    print("dataset fit end")
    user_id_map, user_feature_map, item_id_map, item_feature_map = dataset.mapping()
    num_users, num_topics = dataset.interactions_shape()
    print(f'Number of users: {num_users}, Number of topics: {num_topics}.')
    print("build_interactions start")
    (interactions, weights) = dataset.build_interactions(train[["session", "aid"]].values)
    print("build_interactions end")

    print("train start")
    model = LightFM(loss='warp', no_components=10, random_state=42)
    model.fit(interactions=interactions, epochs=CFG.n_epochs, verbose=1, num_threads=os.cpu_count())
    print("train end")

    item_embeddings = model.item_embeddings
    user_embeddings = model.user_embeddings
    mapped_item_embeddings = {name: item_embeddings[index] for name, index in item_feature_map.items()}
    mapped_user_embeddings = {name: user_embeddings[index] for name, index in user_feature_map.items()}
    dump_pickle(os.path.join(output_dir, "mapped_item_embeddings.pkl"), mapped_item_embeddings)
    dump_pickle(os.path.join(output_dir, "mapped_user_embeddings.pkl"), mapped_user_embeddings)

    _sessions = []
    _embeddings = []
    for name, index in user_feature_map.items():
        _sessions.append(name)
        _embeddings.append(user_embeddings[index])
    df = pd.DataFrame({"session": _sessions, "embedding": _embeddings})
    embed_df = pd.DataFrame(df["embedding"].to_list(), columns=[f"session_embedding{i}" for i in range(10)])
    df = pd.concat([df, embed_df], axis=1)
    df = df.drop(columns=["embedding"])
    df = df.reset_index(drop=True)
    dump_pickle(os.path.join(output_dir, "session_embeddings.pkl"), df)


if __name__ == "__main__":
    output_dir = "output/lightfm"
    os.makedirs(output_dir, exist_ok=True)
    main(output_dir=output_dir)
