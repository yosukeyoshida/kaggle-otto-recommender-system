import gc
import glob
import itertools
import multiprocessing
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


class CFG:
    output_dir = "output"
    DEBUG = True
    TOP_20_CACHE = "./input/otto-pickles/top_40_aids_v4.pkl"
    type_weight_multipliers = {"clicks": 1, "carts": 6, "orders": 3}


def gen_pairs(df, SAMPLING=1):
    df = df.query("session % @SAMPLING == 0").groupby("session", as_index=False, sort=False).apply(lambda g: g.tail(30)).reset_index(drop=True)
    df = pd.merge(df, df, on="session")
    pairs = df.query("abs(ts_x - ts_y) < 24 * 60 * 60 * 1000 and aid_x != aid_y")[["session", "aid_x", "aid_y", "ts_x", "type_y"]].drop_duplicates(
        ["session", "aid_x", "aid_y"]
    )
    return pairs[["aid_x", "aid_y", "ts_x", "type_y"]].values


def gen_aid_pairs():
    all_pairs = defaultdict(lambda: Counter())
    with tqdm(glob.glob("./input/otto-chunk-data-inparquet-format/*_parquet/*"), desc="Chunks") as prog:
        with multiprocessing.Pool(4) as p:
            for idx, chunk_file in enumerate(prog):
                chunk = pd.read_parquet(chunk_file)  # .drop(columns=['type'])
                pair_chunks = p.map(gen_pairs, np.array_split(chunk.head(100000000 if not CFG.DEBUG else 10000), 120))
                for pairs in pair_chunks:
                    for aid1, aid2, ts, typ in pairs:
                        w = 1 + 3 * (ts - 1659304800025) / (1662328791563 - 1659304800025)
                        # HERE WE CAN BOOST WEIGHT, i.e. IF TYP=="ORDERS": W *= 10.0
                        # THEN SAVE THIS MATRIX AS THE "ORDERS" MATRIX
                        # WE CAN MAKE 3 DIFFERENT CO-VISITATION MATRICES
                        all_pairs[aid1][aid2] += w
                prog.set_description(f"Mem: {sys.getsizeof(object) // (2 ** 20)}MB")

                if CFG.DEBUG and idx >= 2:
                    break
                del chunk, pair_chunks
                gc.collect()
    return all_pairs


def load_test():
    dfs = []
    for e, chunk_file in enumerate(tqdm(glob.glob("./input/otto-chunk-data-inparquet-format/test_parquet/*"))):
        chunk = pd.read_parquet(chunk_file)
        dfs.append(chunk)
        if CFG.DEBUG:
            break
    return pd.concat(dfs).reset_index(drop=True).astype({"ts": "datetime64[ms]"})


def suggest_aids(df, top_20):
    # REMOVE DUPLICATE AIDS AND REVERSE ORDER OF LIST

    ###CHANGED PART-----------------------------------------------------------------------------
    aids = df.aid.tolist()
    types = df.type.tolist()
    unique_aids = list(dict.fromkeys(aids[::-1]))
    if len(unique_aids) < 20:
        aids = unique_aids
    else:
        # THIS IS A BASIC IDEA -> you can think about exponential weight decay, or even using df.ts and time-dependent weight decay
        weights = np.logspace(0.1, 1, len(aids), base=2, endpoint=True) - 1
        aids_temp = defaultdict(lambda: 0)
        # to each aids give the sum of the weights of its occurences within the session
        for aid, w, t in zip(aids, weights, types):
            aids_temp[aid] += w * CFG.type_weight_multipliers[t]
        # order in descending order the aids depending on its weight
        sorted_aids = [k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]

        # just a sanity check to see if they get correctly sorted
        # if len(unique_aids)<22:
        #    print(aids_temp)
        #    print(sorted_aids)
        #    break

        # get first 20 aids with highest weight (visited more recently or visited frequently enough to overcome not being recent)
        return sorted_aids[:20]
    ###END CHANGED PART-----------------------------------------------------------------------------

    # Append it with AIDs from the co-visitation matrix.
    aids2 = list(itertools.chain(*[top_20[aid] for aid in aids if aid in top_20]))
    top_aids2 = [aid2 for aid2, cnt in Counter(aids2).most_common(20) if aid2 not in aids]
    return list(aids) + top_aids2[: 20 - len(aids)]


def suggest_orders(df, top_20_orders):
    # REMOVE DUPLICATE AIDS AND REVERSE ORDER OF LIST
    ###CHANGED PART-----------------------------------------------------------------------------
    aids = df.aid.tolist()
    types = df.type.tolist()
    unique_aids = list(dict.fromkeys(aids[::-1]))
    if len(unique_aids) < 20:
        aids = unique_aids
    else:
        weights = np.logspace(0.1, 1, len(aids), base=2, endpoint=True) - 1
        aids_temp = defaultdict(lambda: 0)
        for aid, w, t in zip(aids, weights, types):
            aids_temp[aid] += w * CFG.type_weight_multipliers[t]
        sorted_aids = [k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]
        return sorted_aids[:20]
    ###END CHANGED PART-----------------------------------------------------------------------------

    # Append it with AIDs from the co-visitation matrix.
    aids2 = list(itertools.chain(*[top_20_orders[aid] for aid in aids if aid in top_20_orders]))
    top_aids2 = [aid2 for aid2, cnt in Counter(aids2).most_common(20) if aid2 not in aids]
    return list(aids) + top_aids2[: 20 - len(aids)]


def suggest_carts(df, top_20_carts):
    # REMOVE DUPLICATE AIDS AND REVERSE ORDER OF LIST
    ###CHANGED PART-----------------------------------------------------------------------------
    aids = df.aid.tolist()
    types = df.type.tolist()
    unique_aids = list(dict.fromkeys(aids[::-1]))
    if len(unique_aids) < 20:
        aids = unique_aids
    else:
        weights = np.logspace(0.1, 1, len(aids), base=2, endpoint=True) - 1
        aids_temp = defaultdict(lambda: 0)
        for aid, w, t in zip(aids, weights, types):
            aids_temp[aid] += w * CFG.type_weight_multipliers[t]
        sorted_aids = [k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]
        return sorted_aids[:20]
    ###END CHANGED PART-----------------------------------------------------------------------------

    # Append it with AIDs from the co-visitation matrix.
    aids2 = list(itertools.chain(*[top_20_carts[aid] for aid in aids if aid in top_20_carts]))
    top_aids2 = [aid2 for aid2, cnt in Counter(aids2).most_common(20) if aid2 not in aids]
    return list(aids) + top_aids2[: 20 - len(aids)]


def main():
    if os.path.exists(CFG.TOP_20_CACHE):
        print("Reading top20 AIDs from cache")
        top_20 = pickle.load(open(CFG.TOP_20_CACHE, "rb"))
    else:
        all_pairs = gen_aid_pairs()
        df_top_20 = []
        for aid, cnt in tqdm(all_pairs.items()):
            df_top_20.append({"aid1": aid, "aid2": [aid2 for aid2, freq in cnt.most_common(20)]})
        df_top_20 = pd.DataFrame(df_top_20).set_index("aid1")
        top_20 = df_top_20.aid2.to_dict()
        with open(os.path.join(CFG.output_dir, "top_20_aids.pkl"), "wb") as f:
            pickle.dump(top_20, f)
    top_20_orders = pickle.load(open("./input/otto-pickles-4/top_40_orders_v12.pkl", "rb"))
    top_20_carts = pickle.load(open("./input/otto-pickles-4/top_40_carts_v13.pkl", "rb"))

    for i, (k, v) in enumerate(top_20.items()):
        print(k, v)
        if i > 10:
            break

    test_df = load_test()
    tqdm.pandas()  # enable progress_apply in pandas

    pred_df = test_df.sort_values(["session", "ts"]).groupby(["session"]).progress_apply(lambda x: suggest_aids(x, top_20))

    ##################
    # BELOW IS CODE ADDED BY CHRIS

    pred_df_orders = test_df.sort_values(["session", "ts"]).groupby(["session"]).progress_apply(lambda x: suggest_orders(x, top_20_orders))

    pred_df_carts = test_df.sort_values(["session", "ts"]).groupby(["session"]).progress_apply(lambda x: suggest_carts(x, top_20_carts))

    clicks_pred_df = pd.DataFrame(pred_df.add_suffix("_clicks"), columns=["labels"]).reset_index()
    orders_pred_df = pd.DataFrame(pred_df_orders.add_suffix("_orders"), columns=["labels"]).reset_index()
    carts_pred_df = pd.DataFrame(pred_df_carts.add_suffix("_carts"), columns=["labels"]).reset_index()

    pred_df = pd.concat([clicks_pred_df, orders_pred_df, carts_pred_df])
    pred_df.columns = ["session_type", "labels"]
    pred_df["labels"] = pred_df.labels.apply(lambda x: " ".join(map(str, x)))
    pred_df.to_csv(os.path.join(CFG.output_dir, "submission.csv"), index=False)


if __name__ == "__main__":
    main()
