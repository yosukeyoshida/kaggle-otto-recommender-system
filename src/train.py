import gc
import glob
import itertools
import os
import pickle
from collections import Counter

import cudf
import numpy as np
import optuna
import pandas as pd

import wandb


class CFG:
    top_n_clicks = 20
    top_n_carts_orders = 15
    top_n_buy2buy = 15
    use_saved_models = False
    use_saved_pred = False
    wandb = True
    cv_only = False
    debug = False
    LABEL_TYPE_CLICKS = "clicks"
    LABEL_TYPE_CARTS = "carts"
    LABEL_TYPE_ORDERS = "orders"


def load_test(cv: bool):
    dfs = []
    if cv:
        file_path = "./input/otto-validation/test_parquet/*"
    else:
        file_path = "./input/otto-chunk-data-inparquet-format/test_parquet/*"
    for e, chunk_file in enumerate(glob.glob(file_path)):
        chunk = pd.read_parquet(chunk_file)
        dfs.append(chunk)
        if CFG.debug:
            break
    df = pd.concat(dfs).reset_index(drop=True).astype({"ts": "datetime64[ms]"})
    if CFG.debug:
        df = df.iloc[:100]
    return df


def dump_pickle(path, o):
    with open(path, "wb") as f:
        pickle.dump(o, f)


def suggest_clicks(df, top_n_clicks, type_weight_multipliers):
    # USER HISTORY AIDS AND TYPES
    aids = df.aid.tolist()
    types = df.type.tolist()
    unique_aids = list(dict.fromkeys(aids[::-1]))
    # RERANK CANDIDATES USING WEIGHTS
    if len(unique_aids) >= 20:
        weights = np.logspace(0.1, 1, len(aids), base=2, endpoint=True) - 1
        aids_temp = Counter()
        # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
        for aid, w, t in zip(aids, weights, types):
            aids_temp[aid] += w * type_weight_multipliers[t]
        sorted_aids = [k for k, v in aids_temp.most_common(20)]
        return sorted_aids
    # USE "CLICKS" CO-VISITATION MATRIX
    aids2 = list(itertools.chain(*[top_n_clicks[aid] for aid in unique_aids if aid in top_n_clicks]))
    # RERANK CANDIDATES
    top_aids2 = [aid2 for aid2, cnt in Counter(aids2).most_common(20) if aid2 not in unique_aids]
    result = unique_aids + top_aids2[: 20 - len(unique_aids)]
    return result


def suggest_buys(df, top_n_buy2buy, top_n_buys, type_weight_multipliers):
    # USER HISTORY AIDS AND TYPES
    aids = df.aid.tolist()
    types = df.type.tolist()
    # UNIQUE AIDS AND UNIQUE BUYS
    unique_aids = list(dict.fromkeys(aids[::-1]))
    df = df.loc[(df["type"] == "carts") | (df["type"] == "orders")]
    unique_buys = list(dict.fromkeys(df.aid.tolist()[::-1]))
    # RERANK CANDIDATES USING WEIGHTS
    if len(unique_aids) >= 20:
        weights = np.logspace(0.5, 1, len(aids), base=2, endpoint=True) - 1
        aids_temp = Counter()
        # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
        for aid, w, t in zip(aids, weights, types):
            aids_temp[aid] += w * type_weight_multipliers[t]
        # RERANK CANDIDATES USING "BUY2BUY" CO-VISITATION MATRIX
        aids3 = list(itertools.chain(*[top_n_buy2buy[aid] for aid in unique_buys if aid in top_n_buy2buy]))
        for aid in aids3:
            aids_temp[aid] += 0.1
        sorted_aids = [k for k, v in aids_temp.most_common(20)]
        return sorted_aids
    # USE "CART ORDER" CO-VISITATION MATRIX
    aids2 = list(itertools.chain(*[top_n_buys[aid] for aid in unique_aids if aid in top_n_buys]))
    # USE "BUY2BUY" CO-VISITATION MATRIX
    aids3 = list(itertools.chain(*[top_n_buy2buy[aid] for aid in unique_buys if aid in top_n_buy2buy]))
    # RERANK CANDIDATES
    top_aids2 = [aid2 for aid2, cnt in Counter(aids2 + aids3).most_common(20) if aid2 not in unique_aids]
    result = unique_aids + top_aids2[: 20 - len(unique_aids)]
    return result


def suggest_carts(df, top_n_buys, top_n_clicks, type_weight_multipliers):
    # User history aids and types:vs
    aids = df.aid.tolist()
    types = df.type.tolist()

    # UNIQUE AIDS AND UNIQUE BUYS
    unique_aids = list(dict.fromkeys(aids[::-1]))
    df = df.loc[(df["type"] == CFG.LABEL_TYPE_CLICKS) | (df["type"] == CFG.LABEL_TYPE_CARTS)]
    unique_buys = list(dict.fromkeys(df.aid.tolist()[::-1]))

    # Rerank candidates using weights
    if len(unique_aids) >= 20:
        weights = np.logspace(0.5, 1, len(aids), base=2, endpoint=True) - 1
        aids_temp = Counter()

        # Rerank based on repeat items and types of items
        for aid, w, t in zip(aids, weights, types):
            aids_temp[aid] += w * type_weight_multipliers[t]

        # Rerank candidates using"top_20_carts" co-visitation matrix
        aids2 = list(itertools.chain(*[top_n_buys[aid] for aid in unique_buys if aid in top_n_buys]))
        for aid in aids2:
            aids_temp[aid] += 0.1
        sorted_aids = [k for k, v in aids_temp.most_common(20)]
        return sorted_aids

    # Use "cart order" and "clicks" co-visitation matrices
    aids1 = list(itertools.chain(*[top_n_clicks[aid] for aid in unique_aids if aid in top_n_clicks]))
    aids2 = list(itertools.chain(*[top_n_buys[aid] for aid in unique_aids if aid in top_n_buys]))

    # RERANK CANDIDATES
    top_aids2 = [aid2 for aid2, cnt in Counter(aids1 + aids2).most_common(20) if aid2 not in unique_aids]
    result = unique_aids + top_aids2[: 20 - len(unique_aids)]
    return result


def read_file(f):
    df = cudf.read_parquet(f)
    df.ts = (df.ts / 1000).astype("int32")
    return df


def calc_top_buy2buy(files, CHUNK, output_dir, n):
    DISK_PIECES = 1
    SIZE = 1.86e6 / DISK_PIECES

    # COMPUTE IN PARTS FOR MEMORY MANGEMENT
    for PART in range(DISK_PIECES):
        # MERGE IS FASTEST PROCESSING CHUNKS WITHIN CHUNKS
        # => OUTER CHUNKS
        for j in range(6):
            a = j * CHUNK
            b = min((j + 1) * CHUNK, len(files))

            # => INNER CHUNKS
            for k in range(a, b):
                # READ FILE
                df = read_file(files[k])
                df = df.loc[df["type"].isin([CFG.LABEL_TYPE_CARTS, CFG.LABEL_TYPE_ORDERS])]
                df = df.sort_values(["session", "ts"], ascending=[True, False])
                # USE TAIL OF SESSION
                df = df.reset_index(drop=True)
                df["n"] = df.groupby("session").cumcount()
                df = df.loc[df.n < 30].drop("n", axis=1)
                # CREATE PAIRS
                df = df.merge(df, on="session")
                df = df.loc[((df.ts_x - df.ts_y).abs() < 14 * 24 * 60 * 60) & (df.aid_x != df.aid_y)]  # 14 DAYS
                # MEMORY MANAGEMENT COMPUTE IN PARTS
                df = df.loc[(df.aid_x >= PART * SIZE) & (df.aid_x < (PART + 1) * SIZE)]
                # ASSIGN WEIGHTS
                df = df[["session", "aid_x", "aid_y", "type_y"]].drop_duplicates(["session", "aid_x", "aid_y", "type_y"])
                df["wgt"] = 1
                df = df[["aid_x", "aid_y", "wgt"]]
                df.wgt = df.wgt.astype("float32")
                df = df.groupby(["aid_x", "aid_y"]).wgt.sum()
                # COMBINE INNER CHUNKS
                if k == a:
                    tmp2 = df
                else:
                    tmp2 = tmp2.add(df, fill_value=0)
            # COMBINE OUTER CHUNKS
            if a == 0:
                tmp = tmp2
            else:
                tmp = tmp.add(tmp2, fill_value=0)
            del tmp2, df
            gc.collect()
        # CONVERT MATRIX TO DICTIONARY
        tmp = tmp.reset_index()
        tmp = tmp.sort_values(["aid_x", "wgt"], ascending=[True, False])
        tmp = tmp.reset_index(drop=True)
        tmp["n"] = tmp.groupby("aid_x").aid_y.cumcount()
        tmp = tmp.loc[tmp.n < n].drop("n", axis=1)
        # SAVE PART TO DISK
        df = tmp.to_pandas().groupby("aid_x").aid_y.apply(list)
        dump_pickle(os.path.join(output_dir, f"top_{n}_buy2buy_{PART}.pkl"), df.to_dict())


def calc_top_clicks(files, CHUNK, output_dir, n):
    DISK_PIECES = 4
    SIZE = 1.86e6 / DISK_PIECES

    # COMPUTE IN PARTS FOR MEMORY MANGEMENT
    for PART in range(DISK_PIECES):
        # MERGE IS FASTEST PROCESSING CHUNKS WITHIN CHUNKS
        # => OUTER CHUNKS
        for j in range(6):
            a = j * CHUNK
            b = min((j + 1) * CHUNK, len(files))

            # => INNER CHUNKS
            for k in range(a, b):
                # READ FILE
                df = read_file(files[k])
                df = df.sort_values(["session", "ts"], ascending=[True, False])
                # USE TAIL OF SESSION
                df = df.reset_index(drop=True)
                df["n"] = df.groupby("session").cumcount()
                df = df.loc[df.n < 30].drop("n", axis=1)
                # CREATE PAIRS
                df = df.merge(df, on="session")
                df = df.loc[((df.ts_x - df.ts_y).abs() < 24 * 60 * 60) & (df.aid_x != df.aid_y)]
                # MEMORY MANAGEMENT COMPUTE IN PARTS
                df = df.loc[(df.aid_x >= PART * SIZE) & (df.aid_x < (PART + 1) * SIZE)]
                # ASSIGN WEIGHTS
                df = df[["session", "aid_x", "aid_y", "ts_x"]].drop_duplicates(["session", "aid_x", "aid_y"])
                df["wgt"] = 1 + 3 * (df.ts_x - 1659304800) / (1662328791 - 1659304800)
                df = df[["aid_x", "aid_y", "wgt"]]
                df.wgt = df.wgt.astype("float32")
                df = df.groupby(["aid_x", "aid_y"]).wgt.sum()
                # COMBINE INNER CHUNKS
                if k == a:
                    tmp2 = df
                else:
                    tmp2 = tmp2.add(df, fill_value=0)
            # COMBINE OUTER CHUNKS
            if a == 0:
                tmp = tmp2
            else:
                tmp = tmp.add(tmp2, fill_value=0)
            del tmp2, df
            gc.collect()
        # CONVERT MATRIX TO DICTIONARY
        tmp = tmp.reset_index()
        tmp = tmp.sort_values(["aid_x", "wgt"], ascending=[True, False])
        # SAVE TOP 40
        tmp = tmp.reset_index(drop=True)
        tmp["n"] = tmp.groupby("aid_x").aid_y.cumcount()
        tmp = tmp.loc[tmp.n < n].drop("n", axis=1)
        # SAVE PART TO DISK
        df = tmp.to_pandas().groupby("aid_x").aid_y.apply(list)
        dump_pickle(os.path.join(output_dir, f"top_{n}_clicks_{PART}.pkl"), df.to_dict())


def calc_top_carts_orders(files, CHUNK, output_dir, n, type_weight):
    DISK_PIECES = 4
    SIZE = 1.86e6 / DISK_PIECES

    # COMPUTE IN PARTS FOR MEMORY MANGEMENT
    for PART in range(DISK_PIECES):
        # MERGE IS FASTEST PROCESSING CHUNKS WITHIN CHUNKS
        # => OUTER CHUNKS
        for j in range(6):
            a = j * CHUNK
            b = min((j + 1) * CHUNK, len(files))

            # => INNER CHUNKS
            for k in range(a, b):
                # READ FILE
                df = read_file(files[k])
                df = df.sort_values(["session", "ts"], ascending=[True, False])
                # USE TAIL OF SESSION
                df = df.reset_index(drop=True)
                df["n"] = df.groupby("session").cumcount()
                df = df.loc[df.n < 30].drop("n", axis=1)
                # CREATE PAIRS
                df = df.merge(df, on="session")
                df = df.loc[((df.ts_x - df.ts_y).abs() < 24 * 60 * 60) & (df.aid_x != df.aid_y)]
                # MEMORY MANAGEMENT COMPUTE IN PARTS
                df = df.loc[(df.aid_x >= PART * SIZE) & (df.aid_x < (PART + 1) * SIZE)]
                # ASSIGN WEIGHTS
                df = df[["session", "aid_x", "aid_y", "type_y"]].drop_duplicates(["session", "aid_x", "aid_y", "type_y"])
                df["wgt"] = df.type_y.map(type_weight)
                df = df[["aid_x", "aid_y", "wgt"]]
                df.wgt = df.wgt.astype("float32")
                df = df.groupby(["aid_x", "aid_y"]).wgt.sum()
                # COMBINE INNER CHUNKS
                if k == a:
                    tmp2 = df
                else:
                    tmp2 = tmp2.add(df, fill_value=0)
            # COMBINE OUTER CHUNKS
            if a == 0:
                tmp = tmp2
            else:
                tmp = tmp.add(tmp2, fill_value=0)
            del tmp2, df
            gc.collect()
        # CONVERT MATRIX TO DICTIONARY
        tmp = tmp.reset_index()
        tmp = tmp.sort_values(["aid_x", "wgt"], ascending=[True, False])
        # SAVE TOP 40
        tmp = tmp.reset_index(drop=True)
        tmp["n"] = tmp.groupby("aid_x").aid_y.cumcount()
        tmp = tmp.loc[tmp.n < n].drop("n", axis=1)
        # SAVE PART TO DISK
        df = tmp.to_pandas().groupby("aid_x").aid_y.apply(list)
        dump_pickle(os.path.join(output_dir, f"top_{n}_carts_orders_{PART}.pkl"), df.to_dict())


def main(cv: bool, output_dir: str, **kwargs):
    if cv:
        file_path = "./input/otto-validation/*_parquet/*"
    else:
        file_path = "./input/otto-chunk-data-inparquet-format/*_parquet/*"
    files = glob.glob(file_path)
    CHUNK = int(np.ceil(len(files) / 6))

    if not CFG.use_saved_models:
        # top_n_buys -> suggest_buys
        calc_top_carts_orders(
            files, CHUNK, output_dir, CFG.top_n_carts_orders, type_weight={CFG.LABEL_TYPE_CLICKS: 0.07197733833680556, CFG.LABEL_TYPE_CARTS: 0.708280136807459, CFG.LABEL_TYPE_ORDERS: 0.05318170583899917}
        )
        # top_n_buy2buy -> suggest_buys
        calc_top_buy2buy(files, CHUNK, output_dir, CFG.top_n_buy2buy)
        # top_n_clicks ->  suggest_clicks
        calc_top_clicks(files, CHUNK, output_dir, CFG.top_n_clicks)
    else:
        print("use saved models!!!")

    DISK_PIECES = 4

    test_df = load_test(cv=cv)
    # THREE CO-VISITATION MATRICES
    top_n_clicks = pickle.load(open(os.path.join(output_dir, f"top_{CFG.top_n_clicks}_clicks_0.pkl"), "rb"))
    for k in range(1, DISK_PIECES):
        top_n_clicks.update(pickle.load(open(os.path.join(output_dir, f"top_{CFG.top_n_clicks}_clicks_{k}.pkl"), "rb")))
    top_n_buys = pickle.load(open(os.path.join(output_dir, f"top_{CFG.top_n_carts_orders}_carts_orders_0.pkl"), "rb"))
    for k in range(1, DISK_PIECES):
        top_n_buys.update(pickle.load(open(os.path.join(output_dir, f"top_{CFG.top_n_carts_orders}_carts_orders_{k}.pkl"), "rb")))
    top_n_buy2buy = pickle.load(open(os.path.join(output_dir, f"top_{CFG.top_n_buy2buy}_buy2buy_0.pkl"), "rb"))

    # TOP CLICKS AND ORDERS IN TEST
    top_clicks = test_df.loc[test_df["type"] == "clicks", "aid"].value_counts().index.values[:20]
    top_orders = test_df.loc[test_df["type"] == "orders", "aid"].value_counts().index.values[:20]
    top_carts = test_df.loc[test_df["type"] == "carts", "aid"].value_counts().index.values[:20]

    type_weight_multipliers = {"clicks": 1, "carts": 6, "orders": 3}

    # clicks
    if CFG.use_saved_pred:
        pred_df_clicks = pickle.load(open(os.path.join(output_dir, "pred_df_clicks.pkl"), "rb"))
    else:
        pred_df_clicks = (
            test_df.sort_values(["session", "ts"])
            .groupby(["session"])
            .apply(lambda x: suggest_clicks(x, top_n_clicks, type_weight_multipliers))
            .to_frame()
            .rename(columns={0: "top_n"})
        )
        dump_pickle(os.path.join(output_dir, "pred_df_clicks.pkl"), pred_df_clicks)
    pred_df_clicks["top"] = pred_df_clicks["top_n"].apply(lambda x: list(top_clicks)[: 20 - len(x)])
    pred_df_clicks["labels"] = pred_df_clicks.apply(lambda x: x["top_n"] + x["top"], axis=1)
    pred_df_clicks.index = pred_df_clicks.index.astype(str)
    pred_df_clicks.index += "_clicks"

    # orders
    if CFG.use_saved_pred:
        pred_df_orders = pickle.load(open(os.path.join(output_dir, "pred_df_orders.pkl"), "rb"))
    else:
        pred_df_orders = (
            (
                test_df.sort_values(["session", "ts"])
                .groupby(["session"])
                .apply(lambda x: suggest_buys(x, top_n_buy2buy, top_n_buys, type_weight_multipliers))
            )
            .to_frame()
            .rename(columns={0: "top_n"})
        )
        dump_pickle(os.path.join(output_dir, "pred_df_orders.pkl"), pred_df_orders)

    pred_df_orders["top"] = pred_df_orders["top_n"].apply(lambda x: list(top_orders)[: 20 - len(x)])
    pred_df_orders["labels"] = pred_df_orders.apply(lambda x: x["top_n"] + x["top"], axis=1)
    pred_df_orders.index = pred_df_orders.index.astype(str)
    pred_df_orders.index += "_orders"

    # carts
    if CFG.use_saved_pred:
        pred_df_carts = pickle.load(open(os.path.join(output_dir, "pred_df_carts.pkl"), "rb"))
    else:
        pred_df_carts = (
            test_df.sort_values(["session", "ts"])
            .groupby(["session"])
            .apply(lambda x: suggest_carts(x, top_n_buys, top_n_clicks, type_weight_multipliers))
            .to_frame()
            .rename(columns={0: "top_n"})
        )

    pred_df_carts["top"] = pred_df_carts["top_n"].apply(lambda x: list(top_carts)[: 20 - len(x)])
    pred_df_carts["labels"] = pred_df_carts.apply(lambda x: x["top_n"] + x["top"], axis=1)
    pred_df_carts.index = pred_df_carts.index.astype(str)
    pred_df_carts.index += "_carts"

    # concat
    pred_df = pd.concat([pred_df_clicks, pred_df_orders, pred_df_carts])
    pred_df = pred_df.reset_index()
    pred_df = pred_df.rename(columns={"session": "session_type"})
    pred_df["labels"] = pred_df.labels.apply(lambda x: " ".join(map(str, x)))
    if cv:
        output_file_name = "validation_preds.csv"
    else:
        output_file_name = "submission.csv"
    pred_df["top_n"] = pred_df["top_n"].apply(lambda x: " ".join([str(v) for v in x]))
    pred_df["top"] = pred_df["top"].apply(lambda x: " ".join([str(v) for v in x]))
    pred_df[["session_type", "labels", "top_n", "top"]].to_csv(os.path.join(output_dir, f"full_columns_{output_file_name}"), index=False)
    pred_df[["session_type", "labels"]].to_csv(os.path.join(output_dir, output_file_name), index=False)

    if cv:
        # FREE MEMORY
        del pred_df_clicks, pred_df_orders, pred_df_carts, top_n_clicks, top_n_buy2buy, top_n_buys, top_clicks, top_orders, test_df
        _ = gc.collect()

        # COMPUTE METRIC
        score = 0
        weights = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}
        for t in ["clicks", "carts", "orders"]:
            sub = pred_df.loc[pred_df.session_type.str.contains(t)].copy()
            sub["session"] = sub.session_type.apply(lambda x: int(x.split("_")[0]))
            sub.labels = sub.labels.apply(lambda x: [int(i) for i in x.split(" ")[:20]])
            test_labels = pd.read_parquet("./input/otto-validation/test_labels.parquet")
            test_labels = test_labels.loc[test_labels["type"] == t]
            test_labels = test_labels.merge(sub, how="left", on=["session"])
            test_labels = test_labels[test_labels["labels"].notnull()]
            test_labels["hits"] = test_labels.apply(lambda df: len(set(df.ground_truth).intersection(set(df.labels))), axis=1)
            test_labels["gt_count"] = test_labels.ground_truth.str.len().clip(0, 20)
            test_labels["recall"] = test_labels["hits"] / test_labels["gt_count"]
            recall = test_labels["hits"].sum() / test_labels["gt_count"].sum()
            score += weights[t] * recall
            dump_pickle(os.path.join(output_dir, f"test_labels_{t}.pkl"), test_labels)
            print(f"{t} recall={recall}")
            if CFG.wandb:
                wandb.log({f"{t} recall": recall})
        print(f"total recall={score}")
        if CFG.wandb:
            wandb.log({f"total recall": score})
        return score


def run_optuna():
    output_dir = "output"

    def objective(trial):
        params = {
            "click": trial.suggest_float("click", 0, 1.0),
            "cart": trial.suggest_float("cart", 0, 1.0),
            "order": trial.suggest_float("order", 0, 1.0),
        }
        score = main(cv=True, output_dir=output_dir, type_weight={0: params["click"], 1: params["cart"], 2: params["order"]})
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)


def run_train():
    run_name = None
    if CFG.wandb:
        wandb.init(project="kaggle-otto")
        run_name = wandb.run.name
    if run_name is not None:
        output_dir = os.path.join("output", run_name)
    else:
        output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "cv"), exist_ok=True)
    main(cv=True, output_dir=os.path.join(output_dir, "cv"))
    if CFG.wandb:
        wandb.finish()
    if not CFG.cv_only:
        main(cv=False, output_dir=output_dir)


if __name__ == "__main__":
    # run_optuna()
    run_train()
