import os
import pickle

import pandas as pd


def dump_pickle(path, o):
    with open(path, "wb") as f:
        pickle.dump(o, f)


def calc_metrics(pred_df, output_dir, wandb, candidates_num):
    score_potential = 0
    score_20 = 0
    weights = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}
    for t in ["clicks", "carts", "orders"]:
        sub = pred_df.loc[pred_df["type"] == t].copy()
        sub = sub.groupby("session")["aid"].apply(list)
        test_labels = pd.read_parquet("./input/otto-validation/test_labels.parquet")
        test_labels = test_labels.loc[test_labels["type"] == t]
        test_labels = test_labels.merge(sub, how="left", on=["session"])
        test_labels["aid"] = test_labels["aid"].fillna("").apply(list)
        # potential recall
        test_labels["hits"] = test_labels.apply(lambda df: len(set(df["ground_truth"]).intersection(set(df["aid"]))), axis=1)
        test_labels["gt_count"] = test_labels.ground_truth.str.len().clip(0, 20)
        test_labels["recall"] = test_labels["hits"] / test_labels["gt_count"]
        recall = test_labels["hits"].sum() / test_labels["gt_count"].sum()
        score_potential += weights[t] * recall
        dump_pickle(os.path.join(output_dir, f"test_labels_{t}.pkl"), test_labels)
        print(f"{t} recall@{candidates_num}={recall}")
        if wandb:
            wandb.log({f"{t} recall@{candidates_num}": recall})
        # recall@20
        test_labels["aid"] = test_labels["aid"].apply(lambda x: x[:20])
        test_labels["hits"] = test_labels.apply(lambda df: len(set(df["ground_truth"]).intersection(set(df["aid"]))), axis=1)
        test_labels["gt_count"] = test_labels.ground_truth.str.len().clip(0, 20)
        test_labels["recall"] = test_labels["hits"] / test_labels["gt_count"]
        recall = test_labels["hits"].sum() / test_labels["gt_count"].sum()
        score_20 += weights[t] * recall
        print(f"{t} recall@20={recall}")
        if wandb:
            wandb.log({f"{t} recall@20": recall})
    print(f"total recall@{candidates_num}={score_potential}")
    print(f"total recall@20={score_20}")
    if wandb:
        wandb.log({f"total recall@{candidates_num}": score_potential})
        wandb.log({f"total recall@20": score_20})
