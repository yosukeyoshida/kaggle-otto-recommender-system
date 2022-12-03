import argparse
import gc
import glob
import os
import pickle

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GroupKFold
from wandb.lightgbm import log_summary, wandb_callback

import wandb


class CFG:
    wandb = True
    num_iterations = 200


def read_files(path):
    dfs = []
    dtypes = {
        "session": "int32",
        "aid": "int32",
        "session_interaction_length": "int16",
        "clicks_cnt": "int16",
        "orders_cnt": "int16",
        "clicks_rank": "int32",
        "carts_rank": "int32",
        "orders_rank": "int32",
    }
    float_cols = [
        "this_aid_clicks_cnt",
        "this_aid_carts_cnt",
        "this_aid_orders_cnt",
        "avg_action_num_reverse_chrono",
        "min_action_num_reverse_chrono",
        "max_action_num_reverse_chrono",
        "avg_sec_since_session_start",
        "min_sec_since_session_start",
        "max_sec_since_session_start",
        "avg_sec_to_session_end",
        "min_sec_to_session_end",
        "max_sec_to_session_end",
        "avg_log_recency_score",
        "min_log_recency_score",
        "max_log_recency_score",
        "avg_type_weighted_log_recency_score",
        "min_type_weighted_log_recency_score",
        "max_type_weighted_log_recency_score",
        "covisit_clicks_candidate_num",
        "covisit_carts_candidate_num",
        "covisit_orders_candidate_num",
        "w2v_candidate_num",
    ]

    for file in glob.glob(path):
        df = pd.read_parquet(file)
        for col, dtype in dtypes.items():
            df[col] = df[col].astype(dtype)
        for col in float_cols:
            df[col] = df[col].astype("float16")
        dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)


def read_train_labels():
    train_labels = pd.read_parquet("./input/otto-validation/test_labels.parquet")
    train_labels = train_labels.explode("ground_truth")
    train_labels["aid"] = train_labels["ground_truth"]
    train_labels = train_labels[["session", "type", "aid"]]
    train_labels["aid"] = train_labels["aid"].astype("int32")
    train_labels["session"] = train_labels["session"].astype("int32")
    return train_labels


def dump_pickle(path, o):
    with open(path, "wb") as f:
        pickle.dump(o, f)


def run_train(type, output_dir):
    train = read_files("./input/lgbm_dataset/*")
    train_labels_all = read_train_labels()
    train_labels = train_labels_all[train_labels_all["type"] == type]
    train_labels["gt"] = 1
    train = train.merge(train_labels, how="left", on=["session", "aid"])
    del train_labels_all
    gc.collect()
    train["gt"].fillna(0, inplace=True)
    train["gt"] = train["gt"].astype("int8")
    train = train.reset_index(drop=True)
    # print(train.dtypes)

    feature_cols = train.drop(columns=["gt", "session", "type"]).columns.tolist()
    targets = train["gt"]
    group = train["session"]
    train = train[feature_cols + ["session"]]

    kf = GroupKFold(n_splits=5)
    for fold, (train_indices, valid_indices) in enumerate(kf.split(train, targets, group)):
        X_train, X_valid = train.loc[train_indices], train.loc[valid_indices]
        y_train, y_valid = targets.loc[train_indices], targets.loc[valid_indices]

        X_train = X_train.sort_values(["session", "aid"])
        y_train = y_train.loc[X_train.index]
        X_valid = X_valid.sort_values(["session", "aid"])
        y_valid = y_valid.loc[X_valid.index]

        session_length = X_train.groupby("session").size().to_frame().rename(columns={0: "session_length"}).reset_index()
        session_lengths_train = session_length["session_length"].values
        X_train = X_train.merge(session_length, on="session")
        X_train["session_length"] = X_train["session_length"].astype("int16")
        del session_length
        gc.collect()

        session_length = X_valid.groupby("session").size().to_frame().rename(columns={0: "session_length"}).reset_index()
        session_lengths_valid = session_length["session_length"].values
        X_valid = X_valid.merge(session_length, on="session")
        X_valid["session_length"] = X_valid["session_length"].astype("int16")
        del session_length
        gc.collect()

        X_train = X_train[feature_cols]
        # X_valid = X_valid[feature_cols]

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "boosting_type": "gbdt",
            # 'lambdarank_truncation_level': 10,
            # 'ndcg_eval_at': [10, 5, 20],
            "num_iterations": CFG.num_iterations,
            "random_state": 42,
        }
        _train = lgb.Dataset(X_train, y_train, group=session_lengths_train)
        _valid = lgb.Dataset(X_valid[feature_cols], y_valid, reference=_train, group=session_lengths_valid)
        del X_train, y_train, y_valid, session_lengths_train, session_lengths_valid
        gc.collect()
        # lgb.early_stopping(stopping_rounds=100, verbose=True),
        ranker = lgb.train(params, _train, valid_sets=[_valid], callbacks=[wandb_callback()])
        log_summary(ranker, save_model_checkpoint=True)
        dump_pickle(os.path.join(output_dir, f"ranker_{type}.pkl"), ranker)
        X_valid = X_valid.sort_values(["session", "aid"])
        scores = ranker.predict(X_valid[feature_cols])
        del ranker
        gc.collect()
        X_valid["score"] = scores
        X_valid = X_valid.sort_values(["session", "score"]).groupby("session").tail(20)
        X_valid = X_valid.groupby("session")["aid"].apply(list).to_frame()
        train_labels = train_labels.groupby("session")["aid"].apply(list).to_frame()
        train_labels = train_labels.rename(columns={"aid": "ground_truth"})
        joined = X_valid.merge(train_labels, how="left", on=["session"])
        del X_valid, train_labels
        gc.collect()
        joined = joined[joined["ground_truth"].notnull()]
        joined["hits"] = joined.apply(lambda df: len(set(df.aid).intersection(set(df.ground_truth))), axis=1)
        joined["gt_count"] = joined.ground_truth.str.len().clip(0, 20)
        joined["recall"] = joined["hits"] / joined["gt_count"]
        recall = joined["hits"].sum() / joined["gt_count"].sum()
        break
    if CFG.wandb:
        wandb.log({f"{type} recall": recall})
    return recall


def inference(output_dir):
    test = read_files("./input/lgbm_dataset_test/*")
    # session_length = test.groupby("session").size().to_frame().rename(columns={0: "session_length"}).reset_index()
    # test = test.merge(session_length, on="session")
    feature_cols = test.drop(columns=["session"]).columns.tolist()
    dfs = []
    for type in ["clicks", "carts", "orders"]:
        ranker = pickle.load(open(os.path.join(output_dir, f"ranker_{type}.pkl"), "rb"))
        scores = ranker.predict(test[feature_cols])
        test["score"] = scores
        test_predictions = test.sort_values(["session", "score"]).groupby("session").tail(20)
        test_predictions = test_predictions.groupby("session")["aid"].apply(list)
        test_predictions = test_predictions.to_frame().reset_index()
        test_predictions["session_type"] = test_predictions["session"].apply(lambda x: str(x) + f"_{type}")
        dfs.append(test_predictions)
    sub = pd.concat(dfs)
    sub["labels"] = sub["aid"].apply(lambda x: " ".join(map(str, x)))
    sub[["session_type", "labels"]].to_csv(os.path.join(output_dir, "submission.csv"), index=False)


def main():
    run_name = None
    if CFG.wandb:
        wandb.init(project="kaggle-otto")
        run_name = wandb.run.name
    if run_name is not None:
        output_dir = os.path.join("output/lgbm", run_name)
    else:
        output_dir = "output/lgbm"
    os.makedirs(output_dir, exist_ok=True)

    clicks_recall = run_train("clicks", output_dir)
    carts_recall = run_train("carts", output_dir)
    orders_recall = run_train("orders", output_dir)
    weights = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}
    total_recall = clicks_recall * weights["clicks"] + carts_recall * weights["carts"] + orders_recall * weights["orders"]
    if CFG.wandb:
        wandb.log({"total recall": total_recall})
    inference(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iterations", type=int, default=200)
    args = parser.parse_args()
    CFG.num_iterations = args.num_iterations
    main()
