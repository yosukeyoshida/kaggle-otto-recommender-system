import pandas as pd
import gc
import pickle
import glob
from lightgbm.sklearn import LGBMRanker


def read_files(path):
    dfs = []
    for file in glob.glob(path):
        df = pd.read_parquet(file)
        df["aid"] = df["aid"].astype(int)
        dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)


def read_train_labels():
    train_labels = pd.read_parquet("./input/otto-validation/test_labels.parquet")
    train_labels = train_labels.explode("ground_truth")
    train_labels["aid"] = train_labels["ground_truth"]
    train_labels = train_labels[["session", "type", "aid"]]
    train_labels["aid"] = train_labels["aid"].astype(int)
    return train_labels


def dump_pickle(path, o):
    with open(path, "wb") as f:
        pickle.dump(o, f)

def main():
    train_all = read_files("./input/lgbm_dataset/*")
    print("read train files")
    train_labels_all = read_train_labels()

    for type in ["clicks", "carts", "orders"]:
        print(f"type={type} start")
        train_labels = train_labels_all[train_labels_all["type"] == type]
        train_labels["gt"] = 1
        train = train_all.merge(train_labels, how="left", on=["session", "aid"])
        del train_labels
        gc.collect()
        train["gt"].fillna(0, inplace=True)
        train["gt"] = train["gt"].astype(int)

        ranker = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            boosting_type="dart",
            n_estimators=20,
            importance_type="gain",
        )
        target = "gt"
        feature_cols = train.drop(columns=[target, "session", "type"]).columns.tolist()

        # train
        print(f"train shape: {train.shape}")
        print(f"group sum: {train['session_length'].sum()}")
        ranker = ranker.fit(
            train[feature_cols],
            train[target],
            group=train["session_length"].tolist(),
        )
        dump_pickle(f"output/lgbm/ranker_{type}.pkl", ranker)
        del train, ranker
        gc.collect()

    # test
    # test = read_files("./input/lgbm_dataset_test/*")
    # for type in ["clicks", "carts", "orders"]:
    #     scores = ranker.predict(test[feature_cols])
    #     test["score"] = scores
    #     test_predictions = test.sort_values(["session", "score"]).groupby("session").tail(20)
    #     test_predictions = test_predictions.groupby("session")["aid"].apply(list)
    #     test_predictions = test_predictions.to_frame().reset_index()
    #     dump_pickle(f"output/lgbm/test_predictions_{type}.pkl", test_predictions)
    #     print(f"type={type} finish")


if __name__ == "__main__":
    main()
