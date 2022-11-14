import gc
import glob
import itertools
import os
import pickle
from collections import Counter

import cudf
import numpy as np
import pandas as pd


class CFG:
    type_weight_multipliers = {"clicks": 1, "carts": 6, "orders": 3}
    VER = 2


def load_test():
    dfs = []
    for e, chunk_file in enumerate(glob.glob("./input/otto-chunk-data-inparquet-format/test_parquet/*")):
        chunk = pd.read_parquet(chunk_file)
        dfs.append(chunk)
    return pd.concat(dfs).reset_index(drop=True).astype({"ts": "datetime64[ms]"})


def suggest_clicks(df, top_20_clicks, top_clicks):
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
    aids2 = list(itertools.chain(*[top_20_clicks[aid] for aid in unique_aids if aid in top_20_clicks]))
    # RERANK CANDIDATES
    top_aids2 = [aid2 for aid2, cnt in Counter(aids2).most_common(20) if aid2 not in unique_aids]
    result = unique_aids + top_aids2[: 20 - len(unique_aids)]
    # USE TOP20 TEST CLICKS
    return result + list(top_clicks)[: 20 - len(result)]


def suggest_buys(df, top_20_buy2buy, top_20_buys, top_orders):
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
        aids3 = list(itertools.chain(*[top_20_buy2buy[aid] for aid in unique_buys if aid in top_20_buy2buy]))
        for aid in aids3:
            aids_temp[aid] += 0.1
        sorted_aids = [k for k, v in aids_temp.most_common(20)]
        return sorted_aids
    # USE "CART ORDER" CO-VISITATION MATRIX
    aids2 = list(itertools.chain(*[top_20_buys[aid] for aid in unique_aids if aid in top_20_buys]))
    # USE "BUY2BUY" CO-VISITATION MATRIX
    aids3 = list(itertools.chain(*[top_20_buy2buy[aid] for aid in unique_buys if aid in top_20_buy2buy]))
    # RERANK CANDIDATES
    top_aids2 = [aid2 for aid2, cnt in Counter(aids2 + aids3).most_common(20) if aid2 not in unique_aids]
    result = unique_aids + top_aids2[: 20 - len(unique_aids)]
    # USE TOP20 TEST ORDERS
    return result + list(top_orders)[: 20 - len(result)]


def main():
    output_dir = "output"
    files = glob.glob("./input/otto-chunk-data-inparquet-format/*_parquet/*")
    CHUNK = int(np.ceil(len(files) / 6))
    print(f"We will process {len(files)} files in chunk size {CHUNK} files.")

    type_labels = {"clicks": 0, "carts": 1, "orders": 2}
    type_weight = {0: 1, 1: 6, 2: 3}

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
                df = cudf.read_parquet(files[k])
                df.ts = (df.ts / 1000).astype("int32")
                df["type"] = df["type"].map(type_labels).astype("int8")
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
                df = df[["session", "aid_x", "aid_y", "type_y"]].drop_duplicates(["session", "aid_x", "aid_y"])
                df["wgt"] = df.type_y.map(type_weight)
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
        tmp = tmp.loc[tmp.n < 40].drop("n", axis=1)
        # SAVE PART TO DISK
        df = tmp.to_pandas().groupby("aid_x").aid_y.apply(list)
        with open(os.path.join(output_dir, f"top_40_carts_orders_v{CFG.VER}_{PART}.pkl"), "wb") as f:
            pickle.dump(df.to_dict(), f)

    type_labels = {"clicks": 0, "carts": 1, "orders": 2}

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
                df = cudf.read_parquet(files[k])
                df.ts = (df.ts / 1000).astype("int32")
                df["type"] = df["type"].map(type_labels).astype("int8")
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
                df = df[["session", "aid_x", "aid_y", "type_y"]].drop_duplicates(["session", "aid_x", "aid_y"])
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
        tmp = tmp.loc[tmp.n < 40].drop("n", axis=1)
        # SAVE PART TO DISK
        df = tmp.to_pandas().groupby("aid_x").aid_y.apply(list)
        with open(os.path.join(output_dir, f"top_40_buy2buy_v{CFG.VER}_{PART}.pkl"), "wb") as f:
            pickle.dump(df.to_dict(), f)

    type_labels = {"clicks": 0, "carts": 1, "orders": 2}

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
                df = cudf.read_parquet(files[k])
                df.ts = (df.ts / 1000).astype("int32")
                df["type"] = df["type"].map(type_labels).astype("int8")
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
        tmp = tmp.loc[tmp.n < 40].drop("n", axis=1)
        # SAVE PART TO DISK
        df = tmp.to_pandas().groupby("aid_x").aid_y.apply(list)
        with open(os.path.join(output_dir, f"top_40_clicks_v{CFG.VER}_{PART}.pkl"), "wb") as f:
            pickle.dump(df.to_dict(), f)

    test_df = load_test()
    # THREE CO-VISITATION MATRICES
    top_20_clicks = pickle.load(open(os.path.join(output_dir, f"top_40_clicks_v{CFG.VER}_0.pkl"), "rb"))
    for k in range(1, DISK_PIECES):
        top_20_clicks.update(pickle.load(open(os.path.join(output_dir, f"top_40_clicks_v{CFG.VER}_{k}.pkl"), "rb")))
    top_20_buys = pickle.load(open(os.path.join(output_dir, f"top_40_carts_orders_v{CFG.VER}_0.pkl"), "rb"))
    for k in range(1, DISK_PIECES):
        top_20_buys.update(pickle.load(open(os.path.join(f"top_40_carts_orders_v{CFG.VER}_{k}.pkl"), "rb")))
    top_20_buy2buy = pickle.load(open(os.path.join(output_dir, f"top_40_buy2buy_v{CFG.VER}_0.pkl"), "rb"))

    # TOP CLICKS AND ORDERS IN TEST
    top_clicks = test_df.loc[test_df["type"] == "clicks", "aid"].value_counts().index.values[:20]
    top_orders = test_df.loc[test_df["type"] == "orders", "aid"].value_counts().index.values[:20]
    print("Here are size of our 3 co-visitation matrices:")
    len(top_20_clicks), len(top_20_buy2buy), len(top_20_buys)

    pred_df_clicks = test_df.sort_values(["session", "ts"]).groupby(["session"]).apply(lambda x: suggest_clicks(x, top_20_clicks, top_clicks))

    pred_df_buys = (
        test_df.sort_values(["session", "ts"]).groupby(["session"]).apply(lambda x: suggest_buys(x, top_20_buy2buy, top_20_buys, top_orders))
    )

    clicks_pred_df = pd.DataFrame(pred_df_clicks.add_suffix("_clicks"), columns=["labels"]).reset_index()
    orders_pred_df = pd.DataFrame(pred_df_buys.add_suffix("_orders"), columns=["labels"]).reset_index()
    carts_pred_df = pd.DataFrame(pred_df_buys.add_suffix("_carts"), columns=["labels"]).reset_index()

    pred_df = pd.concat([clicks_pred_df, orders_pred_df, carts_pred_df])
    pred_df.columns = ["session_type", "labels"]
    pred_df["labels"] = pred_df.labels.apply(lambda x: " ".join(map(str, x)))
    pred_df.to_csv(os.path.join(output_dir, "submission.csv"), index=False)


if __name__ == "__main__":
    main()
