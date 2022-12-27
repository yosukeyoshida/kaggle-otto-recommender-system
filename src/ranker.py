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
    n_folds = 5
    dtypes = {
        "session": "int32",
        "aid": "int32",
        "session_clicks_cnt": "int16",
        "session_carts_cnt": "int16",
        "session_orders_cnt": "int16",
        "session_aid_clicks_cnt": "int16",
        "session_aid_carts_cnt": "int16",
        "session_aid_orders_cnt": "int16",
        "clicks_rank": "int32",
        "carts_rank": "int32",
        "orders_rank": "int32",
        "session_clicks_unique_aid": "int16",
        "session_carts_unique_aid": "int16",
        "session_orders_unique_aid": "int16",
        "clicks_uu_rank": "int32",
        "carts_uu_rank": "int32",
        "orders_uu_rank": "int32",
    }
    float_cols = [
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
        "gru4rec_candidate_num",
        "narm_candidate_num",
        "sasrec_candidate_num",
        "session_clicks_carts_ratio",
        "session_carts_orders_ratio",
        "session_clicks_orders_ratio",
        "avg_sec_clicks_carts",
        "min_sec_clicks_carts",
        "max_sec_clicks_carts",
        "avg_sec_carts_orders",
        "min_sec_carts_orders",
        "max_sec_carts_orders",
        "avg_clicks_cnt",
        "avg_carts_cnt",
        "avg_orders_cnt",
        "clicks_carts_ratio",
        "carts_orders_ratio",
        "clicks_orders_ratio",
        "avg_sec_clicks_carts",
        "min_sec_clicks_carts",
        "max_sec_clicks_carts",
        "avg_sec_carts_orders",
        "min_sec_carts_orders",
        "max_sec_carts_orders",
        "avg_sec_session_clicks_carts",
        "min_sec_session_clicks_carts",
        "max_sec_session_clicks_carts",
        "avg_sec_session_carts_orders",
        "min_sec_session_carts_orders",
        "max_sec_session_carts_orders",
    ]


def read_files(path):
    dfs = []

    for file in glob.glob(path):
        df = pd.read_parquet(file)
        for col, dtype in CFG.dtypes.items():
            df[col] = df[col].astype(dtype)
        for col in CFG.float_cols:
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


def run_train(type, output_dir, single_fold):
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
    print(train.dtypes)
    # print(train.dtypes)^M
    positives = train.loc[train["gt"] == 1]
    negatives = train.loc[train["gt"] == 0].sample(n=len(positives) * 80, random_state=42)
    train = pd.concat([positives, negatives], axis=0, ignore_index=True)
    if CFG.wandb:
        wandb.log(
            {
                f"[{type}] train positive size": len(positives),
                f"[{type}] train negative size": len(negatives),
            }
        )
    del positives, negatives
    gc.collect()

    feature_cols = train.drop(columns=["gt", "session", "type"]).columns.tolist()
    targets = train["gt"]
    group = train["session"]
    train = train[feature_cols + ["session"]]
    print(f"train shape: {train.shape}")

    train_labels = train_labels.groupby("session")["aid"].apply(list).to_frame()
    train_labels = train_labels.rename(columns={"aid": "ground_truth"})

    dfs = []
    kf = GroupKFold(n_splits=CFG.n_folds)
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
            # "bagging_fraction": 0.5,
            # "bagging_freq": 10,
        }
        _train = lgb.Dataset(X_train, y_train, group=session_lengths_train)
        _valid = lgb.Dataset(X_valid[feature_cols], y_valid, reference=_train, group=session_lengths_valid)
        del X_train, y_train, y_valid, session_lengths_train, session_lengths_valid
        gc.collect()
        # lgb.early_stopping(stopping_rounds=100, verbose=True),
        print("train start")
        ranker = lgb.train(params, _train, valid_sets=[_valid], callbacks=[wandb_callback()])
        print("train end")
        # log_summary(ranker, save_model_checkpoint=True)
        if CFG.wandb:
            wandb.log({f"[{type}] best_iteration": ranker.best_iteration})
        dump_pickle(os.path.join(output_dir, f"ranker_{type}_fold{fold}.pkl"), ranker)
        X_valid = X_valid.sort_values(["session", "aid"])
        scores = ranker.predict(X_valid[feature_cols])
        del ranker
        gc.collect()
        X_valid["score"] = scores
        X_valid = X_valid.sort_values(["session", "score"]).groupby("session").tail(20)
        X_valid = X_valid.groupby("session")["aid"].apply(list).to_frame()
        joined = X_valid.merge(train_labels, how="left", on=["session"])
        del X_valid
        gc.collect()
        joined = joined[joined["ground_truth"].notnull()]
        joined["hits"] = joined.apply(lambda df: len(set(df.aid).intersection(set(df.ground_truth))), axis=1)
        joined["gt_count"] = joined.ground_truth.str.len().clip(0, 20)
        joined["recall"] = joined["hits"] / joined["gt_count"]
        dump_pickle(os.path.join(output_dir, f"preds_{type}_fold{fold}.pkl"), joined)
        recall = joined["hits"].sum() / joined["gt_count"].sum()
        if CFG.wandb:
            wandb.log({f"[{type}][fold{fold}] recall": recall})
        dfs.append(joined)
        if single_fold:
            break
    joined = pd.concat(dfs)
    recall = joined["hits"].sum() / joined["gt_count"].sum()
    return recall


def cast_cols(df):
    for col, dtype in CFG.dtypes.items():
        df[col] = df[col].astype(dtype)
    for col in CFG.float_cols:
        df[col] = df[col].astype("float16")
    return df


def split_list(l, n):
    for idx in range(0, len(l), n):
        yield l[idx : idx + n]


def run_inference(output_dir, single_fold):
    path = "./input/lgbm_dataset_test/*"
    files = glob.glob(path)
    preds = []
    files_list = split_list(files, 50)
    for files in files_list:
        dfs = []
        for file in files:
            df = pd.read_parquet(file)
            df = cast_cols(df)
            dfs.append(df)
        test = pd.concat(dfs)
        del dfs
        gc.collect()
        feature_cols = test.drop(columns=["session"]).columns.tolist()
        for type in ["clicks", "carts", "orders"]:
            print(f"type={type}")
            pred_folds = []
            for fold in range(CFG.n_folds):
                print(f"fold={fold}")
                ranker = pickle.load(open(os.path.join(output_dir, f"ranker_{type}_fold{fold}.pkl"), "rb"))
                pred = test[["session", "aid"]]
                pred["score"] = ranker.predict(test[feature_cols])
                pred["score"] = pred["score"].astype("float16")
                pred["type"] = type
                pred_folds.append(pred)
                del pred, ranker
                gc.collect()
            pred = pred_folds[0]
            for pf in pred_folds[1:]:
                pred["score"] += pf["score"]
            if not single_fold:
                pred["score"] = pred["score"] / CFG.n_folds
            preds.append(pred)
            del pred_folds
            gc.collect()
            if single_fold:
                break
        del test
        gc.collect()
    preds = pd.concat(preds)
    dump_pickle(os.path.join(output_dir, "preds.pkl"), preds)
    dfs = []
    for type in ["clicks", "carts", "orders"]:
        print(type)
        _preds = preds[preds["type"] == type]
        _preds = _preds.sort_values(["session", "score"]).groupby("session").tail(20)
        _preds = _preds.groupby("session")["aid"].apply(list)
        _preds = _preds.to_frame().reset_index()
        _preds["session_type"] = _preds["session"].apply(lambda x: str(x) + f"_{type}")
        dfs.append(_preds)
        del _preds
        gc.collect()
    sub = pd.concat(dfs)
    sub["labels"] = sub["aid"].apply(lambda x: " ".join(map(str, x)))
    sub[["session_type", "labels"]].to_csv(os.path.join(output_dir, "submission.csv"), index=False)


def main(single_fold):
    run_name = None
    if CFG.wandb:
        wandb.init(project="kaggle-otto", job_type="ranker")
        run_name = wandb.run.name
    if run_name is not None:
        output_dir = os.path.join("output/lgbm", run_name)
    else:
        output_dir = "output/lgbm"
    os.makedirs(output_dir, exist_ok=True)

    clicks_recall = run_train("clicks", output_dir, single_fold)
    carts_recall = run_train("carts", output_dir, single_fold)
    orders_recall = run_train("orders", output_dir, single_fold)
    weights = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}
    total_recall = clicks_recall * weights["clicks"] + carts_recall * weights["carts"] + orders_recall * weights["orders"]
    if CFG.wandb:
        wandb.log({"total recall": total_recall})
    run_inference(output_dir, single_fold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iterations", type=int, default=200)
    parser.add_argument("--single_fold", action='store_true')
    args = parser.parse_args()
    CFG.num_iterations = args.num_iterations
    main(args.single_fold)
