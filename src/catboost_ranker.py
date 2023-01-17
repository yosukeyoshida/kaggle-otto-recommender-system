import argparse
from catboost import CatBoostRanker, Pool
import math
import gc
import glob
import os
import pickle
import pandas as pd
import wandb
from sklearn.model_selection import GroupKFold


class CFG:
    wandb = True
    num_iterations = 5000
    cv_only = False
    save_score = True
    chunk_split_size = 20
    chunk_session_split_size = 20
    n_folds = 5
    input_train_dir = "20230116"
    input_test_dir = "20230116"
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
        "min_day_num": "int8",
        "max_day_num": "int8",
        "clicks_rank_day1": "int32",
        "clicks_rank_day2": "int32",
        "clicks_rank_day3": "int32",
        "clicks_rank_day4": "int32",
        "clicks_rank_day5": "int32",
        "clicks_rank_day6": "int32",
        "clicks_rank_day7": "int32",
        "clicks_rank_day8": "int32",
        "carts_rank_day1": "int32",
        "carts_rank_day2": "int32",
        "carts_rank_day3": "int32",
        "carts_rank_day4": "int32",
        "carts_rank_day5": "int32",
        "carts_rank_day6": "int32",
        "carts_rank_day7": "int32",
        "carts_rank_day8": "int32",
        "orders_rank_day1": "int32",
        "orders_rank_day2": "int32",
        "orders_rank_day3": "int32",
        "orders_rank_day4": "int32",
        "orders_rank_day5": "int32",
        "orders_rank_day6": "int32",
        "orders_rank_day7": "int32",
        "orders_rank_day8": "int32",
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
        "srgnn_candidate_num",
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

global max_score
global best_iteration
max_score = 0
best_iteration = 0

def save_model(i, type, save_model_dir):
    def callback(env):
        global max_score
        global best_iteration
        iteration = env.iteration
        score = env.evaluation_result_list[0][2]
        if iteration % 100 == 0:
            print("iteration {}, score= {:.05f}".format(iteration, score))
        if score > max_score:
            max_score = score
            # print('High Score: iteration {}, score={:.05f}'.format(iteration, score))
            best_iteration = iteration
            env.model.save_model(f"{save_model_dir}/lgb_fold{i}")
    callback.order = 0
    return callback


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
    train_labels_all = read_train_labels()
    train_labels = train_labels_all[train_labels_all["type"] == type]
    train_labels["gt"] = 1

    path = f"./input/lgbm_dataset/{CFG.input_train_dir}/{type}/*"
    files = glob.glob(path)
    chunk_size = math.ceil(len(files) / 3)
    files_list = split_list(files, chunk_size)
    train_list = []
    for i, files in enumerate(files_list):
        print(f"chunk{i}")
        dfs = []
        for file in files:
            df = pd.read_parquet(file)
            df = cast_cols(df)
            dfs.append(df)
        _train = pd.concat(dfs, axis=0, ignore_index=True)
        del dfs
        gc.collect()

        _train = _train.merge(train_labels, how="left", on=["session", "aid"])
        _train["gt"].fillna(0, inplace=True)
        _train["gt"] = _train["gt"].astype("int8")
        train_list.append(_train)
    train = pd.concat(train_list, axis=0, ignore_index=True)
    train = train.sample(frac=1, random_state=42, ignore_index=True)
    del train_labels_all
    gc.collect()

    feature_cols = train.drop(columns=["gt", "session", "type"]).columns.tolist()
    if CFG.wandb:
        wandb.log({f"feature size": len(feature_cols)})
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

        group_id_train = X_train["session"]
        group_id_valid = X_valid["session"]

        X_train = X_train[feature_cols]

        params = {
            'iterations': CFG.num_iterations,
            'custom_metric': ['NDCG', "AUC:type=Ranking"],
            'random_seed': 42,
            "has_time": True,
            'early_stopping_rounds': 100,
            "use_best_model": True,
            "task_type": "GPU",
        }
        _train = Pool(
            data=X_train,
            label=y_train,
            group_id=group_id_train
        )
        _valid = Pool(
            data=X_valid[feature_cols],
            label=y_valid,
            group_id=group_id_valid
        )
        del X_train, y_train, y_valid
        gc.collect()
        print("train start")
        ranker = CatBoostRanker(**params)
        ranker.fit(_train, eval_set=_valid, use_best_model=True)
        print("train end")
        if CFG.wandb:
            wandb.log({f"[{type}] best_iteration": ranker.get_best_iteration()})
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
    path = f"./input/lgbm_dataset_test/{CFG.input_test_dir}/*"
    files = glob.glob(path)
    preds = []
    chunk_size = math.ceil(len(files) / CFG.chunk_split_size)
    files_list = split_list(files, chunk_size)
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
                if single_fold:
                    break
            pred = pred_folds[0]
            if not single_fold:
                for pf in pred_folds[1:]:
                    pred["score"] += pf["score"]
                pred["score"] = pred["score"] / CFG.n_folds
            preds.append(pred)
            del pred_folds
            gc.collect()
        del test
        gc.collect()
    preds = pd.concat(preds)

    if CFG.save_score:
        sessions = sorted(preds["session"].unique())
        chunk_size = math.ceil(len(sessions) / CFG.chunk_session_split_size)
        sessions_list = split_list(sessions, chunk_size)
        preds_save_dir = os.path.join(output_dir, "preds")
        os.makedirs(preds_save_dir, exist_ok=True)
        for i, sessions in enumerate(sessions_list):
            _preds = preds[preds["session"].isin("sessions")]
            dump_pickle(os.path.join(preds_save_dir, f"preds_{i}.pkl"), _preds)
            del _preds
            gc.collect()
    else:
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
        wandb.init(project="kaggle-otto", job_type="ranker", group="main")
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
    if not CFG.cv_only:
        run_inference(output_dir, single_fold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--single_fold", action="store_true")
    args = parser.parse_args()
    main(args.single_fold)
