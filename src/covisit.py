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

from util import calc_metrics, dump_pickle


class CFG:
    type_labels = {"clicks": 0, "carts": 1, "orders": 2}
    top_n_clicks = 15
    top_n_carts_orders = 15
    top_n_buy2buy = 15
    debug = False
    cv_only = False
    wandb = True
    candidates_num = 100


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


def suggest_clicks(df, top_n_clicks):
    aids = df.aid.tolist()
    unique_aids = list(dict.fromkeys(aids[::-1]))
    aids2 = list(itertools.chain(*[top_n_clicks[aid] for aid in unique_aids if aid in top_n_clicks]))
    top_aids2 = [aid2 for aid2, cnt in Counter(aids2).most_common(CFG.candidates_num)]
    return top_aids2


def suggest_buys(df, top_n_buy2buy, top_n_buys):
    aids = df.aid.tolist()
    unique_aids = list(dict.fromkeys(aids[::-1]))
    df = df.loc[(df["type"] == "carts") | (df["type"] == "orders")]
    unique_buys = list(dict.fromkeys(df.aid.tolist()[::-1]))
    aids2 = list(itertools.chain(*[top_n_buys[aid] for aid in unique_aids if aid in top_n_buys]))
    aids3 = list(itertools.chain(*[top_n_buy2buy[aid] for aid in unique_buys if aid in top_n_buy2buy]))
    top_aids2 = [aid2 for aid2, cnt in Counter(aids2 + aids3).most_common(CFG.candidates_num)]
    return top_aids2


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
        # MERGE IS FASTEST PROCESSING CHUNKS WITHIN CHUNKS
        # => OUTER CHUNKS
        for j in range(6):
            a = j * CHUNK
            b = min((j + 1) * CHUNK, len(files))

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

    # top_n_buys -> suggest_buys
    calc_top_carts_orders(
        files, CHUNK, output_dir, CFG.top_n_carts_orders, type_weight={0: 0.07197733833680556, 1: 0.708280136807459, 2: 0.05318170583899917}
    )
    # top_n_buy2buy -> suggest_buys
    calc_top_buy2buy(files, CHUNK, output_dir, CFG.top_n_buy2buy)
    # top_n_clicks ->  suggest_clicks
    calc_top_clicks(files, CHUNK, output_dir, CFG.top_n_clicks)

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

    # suggest clicks
    pred_df_clicks = (
        test_df.sort_values(["session", "ts"])
        .groupby(["session"])
        .apply(lambda x: suggest_clicks(x, top_n_clicks))
        .to_frame()
        .rename(columns={0: "labels"})
    )
    dump_pickle(os.path.join(output_dir, "pred_df_clicks.pkl"), pred_df_clicks)
    pred_df_clicks.index = pred_df_clicks.index.astype(str)
    pred_df_clicks["type"] = "clicks"
    clicks_pred_df = pred_df_clicks

    # suggest buys
    pred_df_buys = (
        (test_df.sort_values(["session", "ts"]).groupby(["session"]).apply(lambda x: suggest_buys(x, top_n_buy2buy, top_n_buys)))
        .to_frame()
        .rename(columns={0: "labels"})
    )
    dump_pickle(os.path.join(output_dir, "pred_df_buys.pkl"), pred_df_buys)

    orders_pred_df = pred_df_buys.copy()
    orders_pred_df.index = orders_pred_df.index.astype(str)
    orders_pred_df["type"] = "orders"

    carts_pred_df = pred_df_buys.copy()
    carts_pred_df.index = carts_pred_df.index.astype(str)
    carts_pred_df["type"] = "carts"

    # concat
    pred_df = pd.concat([clicks_pred_df, orders_pred_df, carts_pred_df])
    pred_df = pred_df.reset_index()
    pred_df["session"] = pred_df["session"].astype("int64")
    dump_pickle(os.path.join(output_dir, "pred_df.pkl"), pred_df)
    pred_df = pred_df.explode("labels")
    pred_df["num"] = list(range(len(pred_df)))
    pred_df["rank"] = pred_df.groupby(["session", "type"])["num"].rank()
    pred_df["rank"] = pred_df["rank"].astype(int)
    pred_df = pred_df.rename(columns={"labels": "aid"})
    pred_df[["session", "aid", "type", "rank"]].to_csv(os.path.join(output_dir, "pred_df.csv"), index=False)
    if cv:
        calc_metrics(pred_df, output_dir, CFG.candidates_num, CFG.wandb)


def run_train():
    run_name = None
    if CFG.wandb:
        wandb.init(project="kaggle-otto", job_type="covisit")
        run_name = wandb.run.name
    if run_name is not None:
        output_dir = os.path.join("output/covisit", run_name)
    else:
        output_dir = "output/covisit"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "cv"), exist_ok=True)
    main(cv=True, output_dir=os.path.join(output_dir, "cv"))
    if not CFG.cv_only:
        main(cv=False, output_dir=output_dir)


if __name__ == "__main__":
    run_train()
