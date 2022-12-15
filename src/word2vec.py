import glob
import wandb
import os
import pickle

import pandas as pd
from annoy import AnnoyIndex
from gensim.models import Word2Vec
from collections import Counter


class CFG:
    wandb = True
    cv_only = False
    candidates_num = 20


def dump_pickle(path, o):
    with open(path, "wb") as f:
        pickle.dump(o, f)


def read_files(path):
    dfs = []
    for file in glob.glob(path):
        df = pd.read_parquet(file)
        dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)


def calc_metrics(pred_df, output_dir):
    score_potential = 0
    score_20 = 0
    weights = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}
    for t in ["clicks", "carts", "orders"]:
        sub = pred_df.loc[pred_df["type"] == t].copy()
        sub = sub.groupby("session")["aid"].apply(list)
        test_labels = pd.read_parquet("./input/otto-validation/test_labels.parquet")
        test_labels = test_labels.loc[test_labels["type"] == t]
        test_labels = test_labels.merge(sub, how="left", on=["session"])
        test_labels = test_labels[test_labels["aid"].notnull()]
        # potential recall
        test_labels["hits"] = test_labels.apply(lambda df: len(set(df["ground_truth"]).intersection(set(df["aid"]))), axis=1)
        test_labels["gt_count"] = test_labels.ground_truth.str.len().clip(0, 20)
        test_labels["recall"] = test_labels["hits"] / test_labels["gt_count"]
        recall = test_labels["hits"].sum() / test_labels["gt_count"].sum()
        score_potential += weights[t] * recall
        dump_pickle(os.path.join(output_dir, f"test_labels_{t}.pkl"), test_labels)
        print(f"{t} recall@{CFG.candidates_num}={recall}")
        if CFG.wandb:
            wandb.log({f"{t} recall@{CFG.candidates_num}": recall})
        # recall@20
        test_labels["aid"] = test_labels["aid"].apply(lambda x: x[:20])
        test_labels["hits"] = test_labels.apply(lambda df: len(set(df["ground_truth"]).intersection(set(df["aid"]))), axis=1)
        test_labels["gt_count"] = test_labels.ground_truth.str.len().clip(0, 20)
        test_labels["recall"] = test_labels["hits"] / test_labels["gt_count"]
        recall = test_labels["hits"].sum() / test_labels["gt_count"].sum()
        score_20 += weights[t] * recall
        print(f"{t} recall@20={recall}")
        if CFG.wandb:
            wandb.log({f"{t} recall@20": recall})
    print(f"total recall@{CFG.candidates_num}={score_potential}")
    print(f"total recall@20={score_20}")
    if CFG.wandb:
        wandb.log({f"total recall@{CFG.candidates_num}": score_potential})
        wandb.log({f"total recall@20": score_20})


def main(cv, output_dir):
    # if cv:
    #     train_file_path = "./input/otto-validation/*_parquet/*"
    #     test_file_path = "./input/otto-validation/test_parquet/*"
    # else:
    #     train_file_path = "./input/otto-chunk-data-inparquet-format/*_parquet/*"
    #     test_file_path = "./input/otto-chunk-data-inparquet-format/test_parquet/*"
    # train = read_files(train_file_path)
    # sentences = train.groupby("session")["aid"].apply(list).to_list()
    # test = read_files(test_file_path)

    # w2vec = Word2Vec(sentences=sentences, vector_size=32, min_count=1, workers=4, window=3)
    # w2vec.save(os.path.join(output_dir, "w2vec.model"))
    w2vec = Word2Vec.load("w2vec.model")
    # test_session_AIDs = test.groupby("session")["aid"].apply(list)
    # dump_pickle("test_session_AIDs.pkl", test_session_AIDs)
    test_session_AIDs = pickle.load(open("input/test_session_aids/aid1.pkl", "rb"))
    labels = []
    i = 0
    for AIDs in test_session_AIDs:
        AIDs = list(dict.fromkeys(AIDs[::-1]))
        labels.append([aid for aid, score in w2vec.wv.most_similar(AIDs, topn=CFG.candidates_num)])
        i += 1
        if i % 10000 == 0:
            print(i)
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
        calc_metrics(prediction_dfs, output_dir)


if __name__ == "__main__":
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
    main(cv=True, output_dir=os.path.join(output_dir, "cv"))
    if not CFG.cv_only:
        main(cv=False, output_dir=output_dir)
