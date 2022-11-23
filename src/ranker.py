import pandas as pd
import glob
import polars as pl
from lightgbm.sklearn import LGBMRanker

class CFG:
    debug = True


def main():
    train = pl.read_parquet("./input/otto-train-and-test-data-for-local-validation/test.parquet")
    train_labels = pl.read_parquet("./input/otto-train-and-test-data-for-local-validation/test_labels.parquet")
    train = train.to_pandas()

    train["action_num_reverse_chrono"] = train.groupby("session").cumcount(ascending=False)
    session_length = train.groupby("session").size().to_frame().rename(columns={0: "session_length"}).reset_index()
    train = train.merge(session_length, on="session")
    linear_interpolation = 0.1 + ((1 - 0.1) / (train["session_length"] - 1)) * (train["session_length"] - train["action_num_reverse_chrono"] - 1)
    train["log_recency_score"] = 2 ** linear_interpolation - 1
    train["log_recency_score"].fillna(1, inplace=True)
    type_weights = {0: 1, 1: 6, 2: 3}
    train["type_weighted_log_recency_score"] = train["type"].apply(lambda x: type_weights[x]) * train["log_recency_score"]

    type2id = {"clicks": 0, "carts": 1, "orders": 2}
    train_labels = train_labels.to_pandas()
    train_labels = train_labels.explode("ground_truth")
    train_labels["aid"] = train_labels["ground_truth"]
    train_labels["type"] = train_labels["type"].apply(lambda x: type2id[x])
    train_labels = train_labels[["session", "type", "aid"]]
    train_labels["gt"] = 1
    train = train.merge(train_labels, how="left", on=["session", "type", "aid"])
    train["gt"].fillna(0, inplace=True)
    train["gt"] = train["gt"].astype(int)
    train["aid"] = train["aid"].astype(int)
    session_lengths_train = session_length["session_length"].values

    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="dart",
        n_estimators=20,
        importance_type="gain",
    )
    feature_cols = ["aid", "type", "action_num_reverse_chrono", "session_length", "log_recency_score", "type_weighted_log_recency_score"]
    target = "gt"

    # train
    ranker = ranker.fit(
        train[feature_cols],
        train[target],
        group=session_lengths_train,
    )

    # test
    test = pl.read_parquet("./input/otto-full-optimized-memory-footprint/test.parquet")
    test = test.to_pandas()

    test["action_num_reverse_chrono"] = test.groupby("session").cumcount(ascending=False)
    session_length = test.groupby("session").size().to_frame().rename(columns={0: "session_length"}).reset_index()
    test = test.merge(session_length, on="session")
    linear_interpolation = 0.1 + ((1 - 0.1) / (test["session_length"] - 1)) * (
                test["session_length"] - test["action_num_reverse_chrono"] - 1)
    test["log_recency_score"] = 2 ** linear_interpolation - 1
    test["log_recency_score"].fillna(1, inplace=True)
    type_weights = {0: 1, 1: 6, 2: 3}
    test["type_weighted_log_recency_score"] = test["type"].apply(lambda x: type_weights[x]) * test["log_recency_score"]

    scores = ranker.predict(test[feature_cols])
    test["score"] = scores
    test_predictions = test.sort_values(["session", "score"]).groupby("session").tail(20)
    test_predictions = test_predictions.groupby("session")["aid"].apply(list)
    test_predictions = test_predictions.to_frame().reset_index()
    session_types = []
    labels = []
    for session, preds in zip(test_predictions["session"].to_numpy(), test_predictions["aid"].to_numpy()):
        l = " ".join(str(p) for p in preds)
        for session_type in ["clicks", "carts", "orders"]:
            labels.append(l)
            session_types.append(f"{session}_{session_type}")
    submission = pd.DataFrame({"session_type": session_types, "labels": labels})
    submission.to_csv("submission.csv", index=False)

    # pred_df = pd.DataFrame({"session_type": session_types, "labels": labels})
    # # COMPUTE METRIC
    # score = 0
    # weights = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}
    # for t in ["clicks", "carts", "orders"]:
    #     sub = pred_df.loc[pred_df.session_type.str.contains(t)].copy()
    #     sub["session"] = sub.session_type.apply(lambda x: int(x.split("_")[0]))
    #     sub.labels = sub.labels.apply(lambda x: [int(i) for i in x.split(" ")[:20]])
    #     test_labels = pd.read_parquet("./input/otto-validation/test_labels.parquet")
    #     # test_labels = pd.read_parquet("./input/otto-full-optimized-memory-footprint/test.parquet")
    #     test_labels["type"] = test_labels["type"].map({v: k for k, v in type2id.items()})
    #     test_labels = test_labels.loc[test_labels["type"] == t]
    #     test_labels = test_labels.merge(sub, how="left", on=["session"])
    #     test_labels = test_labels[test_labels["labels"].notnull()]
    #     test_labels["hits"] = test_labels.apply(lambda df: len(set(df.ground_truth).intersection(set(df.labels))), axis=1)
    #     test_labels["gt_count"] = test_labels.ground_truth.str.len().clip(0, 20)
    #     test_labels["recall"] = test_labels["hits"] / test_labels["gt_count"]
    #     recall = test_labels["hits"].sum() / test_labels["gt_count"].sum()
    #     score += weights[t] * recall
    #     # dump_pickle(os.path.join(output_dir, f"test_labels_{t}.pkl"), test_labels)
    #     print(f"{t} recall={recall}")
    #     # if CFG.wandb:
    #     #     wandb.log({f"{t} recall": recall})
    # print(f"total recall={score}")

if __name__ == "__main__":
    main()