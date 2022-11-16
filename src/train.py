import gc
import wandb
import glob
import itertools
import os
import pickle
from collections import Counter

import cudf
import numpy as np
import pandas as pd


class CFG:
    type_labels = {"clicks": 0, "carts": 1, "orders": 2}
    type_weight_multipliers = {"clicks": 1, "carts": 6, "orders": 3}
    type_weight = {0: 1, 1: 6, 2: 3}
    use_saved_models = False
    wandb = True
    top_n_clicks = 20
    top_n_carts_orders = 15
    top_n_buy2buy = 15


def load_test(cv: bool):
    dfs = []
    if cv:
        file_path = "./input/otto-validation/test_parquet/*"
    else:
        file_path = "./input/otto-chunk-data-inparquet-format/test_parquet/*"
    for e, chunk_file in enumerate(glob.glob(file_path)):
        chunk = pd.read_parquet(chunk_file)
        dfs.append(chunk)
    df = pd.concat(dfs).reset_index(drop=True).astype({"ts": "datetime64[ms]"})
    return df


def suggest_clicks(df, top_n_clicks, top_clicks):
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
            aids_temp[aid] += w * CFG.type_weight_multipliers[t]
        sorted_aids = [k for k, v in aids_temp.most_common(20)]
        return sorted_aids
    # USE "CLICKS" CO-VISITATION MATRIX
    aids2 = list(itertools.chain(*[top_n_clicks[aid] for aid in unique_aids if aid in top_n_clicks]))
    # RERANK CANDIDATES
    top_aids2 = [aid2 for aid2, cnt in Counter(aids2).most_common(20) if aid2 not in unique_aids]
    result = unique_aids + top_aids2[: 20 - len(unique_aids)]
    # USE TOP20 TEST CLICKS
    return result + list(top_clicks)[: 20 - len(result)]


def suggest_buys(df, top_n_buy2buy, top_n_buys, top_orders):
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
            aids_temp[aid] += w * CFG.type_weight_multipliers[t]
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
    # USE TOP20 TEST ORDERS
    return result + list(top_orders)[: 20 - len(result)]


def read_file(f):
    df = cudf.read_parquet(f)
    df.ts = (df.ts / 1000).astype("int32")
    df["type"] = df["type"].map(CFG.type_labels).astype("int8")
    return df


def calc_top_buy2buy(files, CHUNK, output_dir, n):
    DISK_PIECES = 1
    SIZE = 1.86e6 / DISK_PIECES

    # COMPUTE IN PARTS FOR MEMORY MANGEMENT
    for PART in range(DISK_PIECES):
        print()
        print("### DISK PART", PART + 1)

        # MERGE IS FASTEST PROCESSING CHUNKS WITHIN CHUNKS
        # => OUTER CHUNKS
        for j in range(6):
            a = j * CHUNK
            b = min((j + 1) * CHUNK, len(files))
            print(f"Processing files {a} thru {b - 1}...")

            # => INNER CHUNKS
            for k in range(a, b):
                # READ FILE
                df = read_file(files[k])
                df = df.loc[df["type"].isin([1, 2])]  # ONLY WANT CARTS AND ORDERS
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
                print(k, ", ", end="")
            print()
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
        with open(os.path.join(output_dir, f"top_{n}_buy2buy_{PART}.pkl"), "wb") as f:
            pickle.dump(df.to_dict(), f)


def calc_top_clicks(files, CHUNK, output_dir, n):
    DISK_PIECES = 4
    SIZE = 1.86e6 / DISK_PIECES

    # COMPUTE IN PARTS FOR MEMORY MANGEMENT
    for PART in range(DISK_PIECES):
        print()
        print("### DISK PART", PART + 1)

        # MERGE IS FASTEST PROCESSING CHUNKS WITHIN CHUNKS
        # => OUTER CHUNKS
        for j in range(6):
            a = j * CHUNK
            b = min((j + 1) * CHUNK, len(files))
            print(f"Processing files {a} thru {b - 1}...")

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
                print(k, ", ", end="")
            print()
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
        with open(os.path.join(output_dir, f"top_{n}_clicks_{PART}.pkl"), "wb") as f:
            pickle.dump(df.to_dict(), f)


def calc_top_carts_orders(files, CHUNK, output_dir, n):
    DISK_PIECES = 4
    SIZE = 1.86e6 / DISK_PIECES

    # COMPUTE IN PARTS FOR MEMORY MANGEMENT
    for PART in range(DISK_PIECES):
        print()
        print("### DISK PART", PART + 1)

        # MERGE IS FASTEST PROCESSING CHUNKS WITHIN CHUNKS
        # => OUTER CHUNKS
        for j in range(6):
            a = j * CHUNK
            b = min((j + 1) * CHUNK, len(files))
            print(f"Processing files {a} thru {b - 1}...")

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
                df["wgt"] = df.type_y.map(CFG.type_weight)
                df = df[["aid_x", "aid_y", "wgt"]]
                df.wgt = df.wgt.astype("float32")
                df = df.groupby(["aid_x", "aid_y"]).wgt.sum()
                # COMBINE INNER CHUNKS
                if k == a:
                    tmp2 = df
                else:
                    tmp2 = tmp2.add(df, fill_value=0)
                print(k, ", ", end="")
            print()
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
        with open(os.path.join(output_dir, f"top_{n}_carts_orders_{PART}.pkl"), "wb") as f:
            pickle.dump(df.to_dict(), f)


def main(cv: bool, output_dir: str):
    if cv:
        file_path = "./input/otto-validation/*_parquet/*"
    else:
        file_path = "./input/otto-chunk-data-inparquet-format/*_parquet/*"
    files = glob.glob(file_path)
    CHUNK = int(np.ceil(len(files) / 6))
    print(f"We will process {len(files)} files in chunk size {CHUNK} files.")

    if not CFG.use_saved_models:
        calc_top_carts_orders(files, CHUNK, output_dir, CFG.top_n_carts_orders)
        calc_top_buy2buy(files, CHUNK, output_dir, CFG.top_n_buy2buy)
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
    pred_df_clicks = test_df.sort_values(["session", "ts"]).groupby(["session"]).apply(lambda x: suggest_clicks(x, top_n_clicks, top_clicks))
    pred_df_buys = (
        test_df.sort_values(["session", "ts"]).groupby(["session"]).apply(lambda x: suggest_buys(x, top_n_buy2buy, top_n_buys, top_orders))
    )

    clicks_pred_df = pd.DataFrame(pred_df_clicks.add_suffix("_clicks"), columns=["labels"]).reset_index()
    orders_pred_df = pd.DataFrame(pred_df_buys.add_suffix("_orders"), columns=["labels"]).reset_index()
    carts_pred_df = pd.DataFrame(pred_df_buys.add_suffix("_carts"), columns=["labels"]).reset_index()

    pred_df = pd.concat([clicks_pred_df, orders_pred_df, carts_pred_df])
    pred_df.columns = ["session_type", "labels"]
    pred_df["labels"] = pred_df.labels.apply(lambda x: " ".join(map(str, x)))
    if cv:
        output_file_name = "validation_preds.csv"
    else:
        output_file_name = "submission.csv"
    pred_df.to_csv(os.path.join(output_dir, output_file_name), index=False)

    if cv:
        # FREE MEMORY
        del pred_df_clicks, pred_df_buys, clicks_pred_df, orders_pred_df, carts_pred_df
        del top_n_clicks, top_n_buy2buy, top_n_buys, top_clicks, top_orders, test_df
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
            test_labels["hits"] = test_labels.apply(lambda df: len(set(df.ground_truth).intersection(set(df.labels))), axis=1)
            test_labels["gt_count"] = test_labels.ground_truth.str.len().clip(0, 20)
            recall = test_labels["hits"].sum() / test_labels["gt_count"].sum()
            score += weights[t] * recall
            if CFG.wandb:
                wandb.log({f"{t} recall": recall})
        if CFG.wandb:
            wandb.log({f"total recall": score})

if __name__ == "__main__":
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
    wandb.finish()
    main(cv=False, output_dir=output_dir)
