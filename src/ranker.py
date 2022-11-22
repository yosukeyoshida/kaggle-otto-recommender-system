import pandas as pd
import glob
import numpy as np
import polars as pl
from lightgbm.sklearn import LGBMRanker

class CFG:
    debug = True


def add_action_num_reverse_chrono(df):
    return df.select([pl.col("*"), pl.col("session").cumcount().reverse().over("session").alias("action_num_reverse_chrono")])


def add_session_length(df):
    return df.select([pl.col("*"), pl.col("session").count().over("session").alias("session_length")])


def add_log_recency_score(df):
    linear_interpolation = 0.1 + ((1 - 0.1) / (df["session_length"] - 1)) * (df["session_length"] - df["action_num_reverse_chrono"] - 1)
    return df.with_columns(pl.Series(2 ** linear_interpolation - 1).alias("log_recency_score")).fill_nan(1)


def add_type_weighted_log_recency_score(df):
    type_weights = {0: 1, 1: 6, 2: 3}
    type_weighted_log_recency_score = pl.Series(df["type"].apply(lambda x: type_weights[x]) * df["log_recency_score"])
    return df.with_column(type_weighted_log_recency_score.alias("type_weighted_log_recency_score"))


def apply(df, pipeline):
    for f in pipeline:
        df = f(df)
    return df


def get_session_lenghts(df):
    return df.groupby("session").agg([pl.col("session").count().alias("session_length")])["session_length"].to_numpy()


def load_files(file_path):
    dfs = []
    for e, chunk_file in enumerate(glob.glob(file_path)):
        chunk = pd.read_parquet(chunk_file)
        dfs.append(chunk)
        if CFG.debug:
            break
    df = pd.concat(dfs).reset_index(drop=True).astype({"ts": "datetime64[ms]"})
    if CFG.debug:
        df = df.iloc[:100]
    return df


def main():
    # train
    # ┌──────────┬─────────┬────────────┬──────┐
    # │ session  ┆ aid     ┆ ts         ┆ type │
    # │ ---      ┆ ---     ┆ ---        ┆ ---  │
    # │ i32      ┆ i32     ┆ i32        ┆ u8   │
    # ╞══════════╪═════════╪════════════╪══════╡
    # │ 11098528 ┆ 11830   ┆ 1661119200 ┆ 0    │
    # ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
    # │ 11098529 ┆ 1105029 ┆ 1661119200 ┆ 0    │
    # ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
    # │ 11098530 ┆ 264500  ┆ 1661119200 ┆ 0    │
    # ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
    # │ 11098530 ┆ 264500  ┆ 1661119288 ┆ 0    │
    # ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
    # │ 11098530 ┆ 409236  ┆ 1661119369 ┆ 0    │
    # └──────────┴─────────┴────────────┴──────┘
    #
    # train_labels
    # ┌──────────┬────────┬───────────────────────────────┐
    # │ session  ┆ type   ┆ ground_truth                  │
    # │ ---      ┆ ---    ┆ ---                           │
    # │ i64      ┆ str    ┆ list[i64]                     │
    # ╞══════════╪════════╪═══════════════════════════════╡
    # │ 11098528 ┆ clicks ┆ [1679529]                     │
    # ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ 11098528 ┆ carts  ┆ [1199737]                     │
    # ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ 11098528 ┆ orders ┆ [990658, 950341, ... 1033148] │
    # ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ 11098529 ┆ clicks ┆ [1105029]                     │
    # ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ ...      ┆ ...    ┆ ...                           │
    # ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ 12899775 ┆ clicks ┆ [1760714]                     │
    # ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ 12899776 ┆ clicks ┆ [1737908]                     │
    # train = pl.read_parquet("./input/otto-train-and-test-data-for-local-validation/test.parquet")
    file_path = "./input/otto-validation/train_parquet/*"
    files = glob.glob(file_path)
    train = load_files(file_path)
    train_labels = pl.read_parquet("./input/otto-train-and-test-data-for-local-validation/test_labels.parquet")
    pipeline = [add_action_num_reverse_chrono, add_session_length, add_log_recency_score, add_type_weighted_log_recency_score]
    train = apply(train, pipeline)
    train.write_csv("ranker_train.csv")
    type2id = {"clicks": 0, "carts": 1, "orders": 2}
    train_labels = train_labels.explode("ground_truth").with_columns(
        [pl.col("ground_truth").alias("aid"), pl.col("type").apply(lambda x: type2id[x])]
    )[["session", "type", "aid"]]
    train_labels = train_labels.with_columns(
        [pl.col("session").cast(pl.datatypes.Int32), pl.col("type").cast(pl.datatypes.UInt8), pl.col("aid").cast(pl.datatypes.Int32)]
    )
    train_labels = train_labels.with_column(pl.lit(1).alias("gt"))
    # train_labels
    # ┌──────────┬──────┬─────────┬─────┐
    # │ session  ┆ type ┆ aid     ┆ gt  │
    # │ ---      ┆ ---  ┆ ---     ┆ --- │
    # │ i32      ┆ u8   ┆ i32     ┆ i32 │
    # ╞══════════╪══════╪═════════╪═════╡
    # │ 11098528 ┆ 0    ┆ 1679529 ┆ 1   │
    # ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    # │ 11098528 ┆ 1    ┆ 1199737 ┆ 1   │
    # ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    # │ 11098528 ┆ 2    ┆ 990658  ┆ 1   │
    # ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    # │ 11098528 ┆ 2    ┆ 950341  ┆ 1   │
    # ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    # │ ...      ┆ ...  ┆ ...     ┆ ... │
    # ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    # │ 12899775 ┆ 0    ┆ 1760714 ┆ 1   │
    # ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    # │ 12899776 ┆ 0    ┆ 1737908 ┆ 1   │
    # ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    # │ 12899777 ┆ 0    ┆ 384045  ┆ 1   │
    # ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    # │ 12899778 ┆ 0    ┆ 32070   ┆ 1   │
    # └──────────┴──────┴─────────┴─────┘
    train = train.join(train_labels, how="left", on=["session", "type", "aid"]).with_column(pl.col("gt").fill_null(0))
    session_lengths_train = get_session_lenghts(train)  # array([ 1, 11,  2, ...,  3,  1,  1], dtype=uint32)
    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="dart",
        n_estimators=20,
        importance_type="gain",
    )
    feature_cols = ["aid", "type", "action_num_reverse_chrono", "session_length", "log_recency_score", "type_weighted_log_recency_score"]
    target = "gt"
    ranker = ranker.fit(
        train[feature_cols].to_pandas(),
        train[target].to_pandas(),
        group=session_lengths_train,
    )
    test = pl.read_parquet("./input/otto-full-optimized-memory-footprint/test.parquet")
    test = apply(test, pipeline)
    scores = ranker.predict(test[feature_cols].to_pandas())
    pl.Series(name="predictions", values=scores)
    test = test.with_columns(pl.Series(name="score", values=scores))
    test_predictions = test.sort(["session", "score"], reverse=True).groupby("session").agg([pl.col("aid").limit(20).list()])
    session_types = []
    labels = []
    for session, preds in zip(test_predictions["session"].to_numpy(), test_predictions["aid"].to_numpy()):
        l = " ".join(str(p) for p in preds)
        for session_type in ["clicks", "carts", "orders"]:
            labels.append(l)
            session_types.append(f"{session}_{session_type}")
    submission = pl.DataFrame({"session_type": session_types, "labels": labels})
    submission.write_csv("submission.csv")

    pred_df = pd.DataFrame({"session_type": session_types, "labels": labels})
    # COMPUTE METRIC
    score = 0
    weights = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}
    for t in ["clicks", "carts", "orders"]:
        sub = pred_df.loc[pred_df.session_type.str.contains(t)].copy()
        sub["session"] = sub.session_type.apply(lambda x: int(x.split("_")[0]))
        sub.labels = sub.labels.apply(lambda x: [int(i) for i in x.split(" ")[:20]])
        test_labels = pd.read_parquet("./input/otto-validation/test_labels.parquet")
        # test_labels = pd.read_parquet("./input/otto-full-optimized-memory-footprint/test.parquet")
        test_labels["type"] = test_labels["type"].map({v: k for k, v in type2id.items()})
        test_labels = test_labels.loc[test_labels["type"] == t]
        test_labels = test_labels.merge(sub, how="left", on=["session"])
        test_labels = test_labels[test_labels["labels"].notnull()]
        import pdb; pdb.set_trace()
        test_labels["hits"] = test_labels.apply(lambda df: len(set(df.ground_truth).intersection(set(df.labels))), axis=1)
        test_labels["gt_count"] = test_labels.ground_truth.str.len().clip(0, 20)
        test_labels["recall"] = test_labels["hits"] / test_labels["gt_count"]
        recall = test_labels["hits"].sum() / test_labels["gt_count"].sum()
        score += weights[t] * recall
        # dump_pickle(os.path.join(output_dir, f"test_labels_{t}.pkl"), test_labels)
        print(f"{t} recall={recall}")
        # if CFG.wandb:
        #     wandb.log({f"{t} recall": recall})
    print(f"total recall={score}")

if __name__ == "__main__":
    main()