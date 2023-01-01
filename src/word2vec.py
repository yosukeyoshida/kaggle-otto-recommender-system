import glob
import argparse
import os
from collections import Counter

import pandas as pd
import wandb
from annoy import AnnoyIndex
from gensim.models import Word2Vec

from util import calc_metrics, dump_pickle


class CFG:
    wandb = True
    cv_only = True
    candidates_num = 30


def read_files(path):
    dfs = []
    for file in glob.glob(path):
        df = pd.read_parquet(file)
        dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)


def main(cv, output_dir, **kwargs):
    if cv:
        train_file_path = "./input/otto-validation/*_parquet/*"
        test_file_path = "./input/otto-validation/test_parquet/*"
    else:
        train_file_path = "./input/otto-chunk-data-inparquet-format/*_parquet/*"
        test_file_path = "./input/otto-chunk-data-inparquet-format/test_parquet/*"
    train = read_files(train_file_path)
    sentences = train.groupby("session")["aid"].apply(list).to_list()
    test = read_files(test_file_path)

    # w2vec = Word2Vec(sentences=sentences, vector_size=32, min_count=1, workers=4, window=3)
    w2vec = Word2Vec(sentences=sentences, vector_size=32, min_count=1, workers=4, window=kwargs["window"], sg=1)
    if CFG.wandb:
        wandb.log({"window": kwargs["window"]})

    aid2idx = {aid: i for i, aid in enumerate(w2vec.wv.index_to_key)}
    index = AnnoyIndex(32, "angular")

    for aid, idx in aid2idx.items():
        index.add_item(idx, w2vec.wv.vectors[idx])

    index.build(100)
    w2vec.save(os.path.join(output_dir, "w2vec.model"))
    test_session_AIDs = test.groupby("session")["aid"].apply(list)
    labels = []
    for AIDs in test_session_AIDs:
        AIDs = list(dict.fromkeys(AIDs[::-1]))
        most_recent_aid = AIDs
        nns = []
        for aid in most_recent_aid:
            nns += [w2vec.wv.index_to_key[i] for i in index.get_nns_by_item(aid2idx[aid], 20)]
        labels.append([aid for aid, cnt in Counter(nns).most_common(CFG.candidates_num)])
    pred_df = pd.DataFrame(data={"session": test_session_AIDs.index, "labels": labels})
    dump_pickle(os.path.join(output_dir, "predictions.pkl"), pred_df)
    pred_df = pred_df.explode("labels")
    pred_df["num"] = list(range(len(pred_df)))
    pred_df["rank"] = pred_df.groupby(["session"])["num"].rank()
    pred_df["rank"] = pred_df["rank"].astype(int)
    pred_df = pred_df.rename(columns={"labels": "aid"})
    pred_df[["session", "aid", "rank"]].to_csv(os.path.join(output_dir, "pred_df.csv"), index=False)
    if cv:
        prediction_dfs = []
        for st in ["clicks", "carts", "orders"]:
            modified_predictions = pred_df.copy()
            modified_predictions["type"] = st
            prediction_dfs.append(modified_predictions)
        prediction_dfs = pd.concat(prediction_dfs).reset_index(drop=True)
        calc_metrics(prediction_dfs, output_dir, CFG.candidates_num, CFG.wandb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=200)
    args = parser.parse_args()

    run_name = None
    if CFG.wandb:
        wandb.init(project="kaggle-otto", job_type="word2vec")
        run_name = wandb.run.name
    if run_name is not None:
        output_dir = os.path.join("output/word2vec", run_name)
    else:
        output_dir = "output/word2vec"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "cv"), exist_ok=True)
    params = {
        "window": args.window
    }
    main(cv=True, output_dir=os.path.join(output_dir, "cv"), **params)
    if not CFG.cv_only:
        main(cv=False, output_dir=output_dir)
