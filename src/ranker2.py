import pandas as pd
import gc
import pickle
import glob
from lightgbm.sklearn import LGBMRanker


def read_files(path):
    dfs = []
    dtypes = {'session': 'int32', 'aid': 'int32', 'session_interaction_length': 'int16', 'clicks_cnt': 'int16', 'orders_cnt': 'int16'}
    for file in glob.glob(path):
        df = pd.read_parquet(file)
        for col, dtype in dtypes.items():
            df[col] = df[col].astype(dtype)
        dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)


def read_train_labels():
    train_labels = pd.read_parquet("./input/otto-validation/test_labels.parquet")
    train_labels = train_labels.explode("ground_truth")
    train_labels["aid"] = train_labels["ground_truth"]
    train_labels = train_labels[["session", "type", "aid"]]
    train_labels["aid"] = train_labels["aid"].astype('int32')
    train_labels["session"] = train_labels["session"].astype('int32')
    return train_labels


def dump_pickle(path, o):
    with open(path, "wb") as f:
        pickle.dump(o, f)

def main():
    for type in ["clicks", "carts", "orders"]:
        train = read_files("./input/lgbm_dataset/*")
        train = train.drop(columns=["session_length"])
        train_labels_all = read_train_labels()
        session_length = train.groupby("session").size().to_frame().rename(columns={0: "session_length"}).reset_index()
        session_lengths_train = session_length["session_length"].values
        train = train.merge(session_length, on="session")
        del session_length
        gc.collect()

        train_labels = train_labels_all[train_labels_all["type"] == type]
        train_labels["gt"] = 1
        train = train.merge(train_labels, how="left", on=["session", "aid"])
        del train_labels
        gc.collect()
        train["gt"].fillna(0, inplace=True)
        train["gt"] = train["gt"].astype('int8')

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
        ranker = ranker.fit(
            train[feature_cols],
            train[target],
            group=session_lengths_train,
        )
        dump_pickle(f"output/lgbm/ranker_{type}.pkl", ranker)
        del train
        gc.collect()

        test = read_files("./input/lgbm_dataset_test/*")
        test = test.drop(columns=["session_length"])
        session_length = test.groupby("session").size().to_frame().rename(columns={0: "session_length"}).reset_index()
        test = test.merge(session_length, on="session")
        for type in ["clicks", "carts", "orders"]:
            scores = ranker.predict(test[feature_cols])
            test["score"] = scores
            test_predictions = test.sort_values(["session", "score"]).groupby("session").tail(20)
            test_predictions = test_predictions.groupby("session")["aid"].apply(list)
            test_predictions = test_predictions.to_frame().reset_index()
            dump_pickle(f"output/lgbm/test_predictions_{type}.pkl", test_predictions)
            print(f"type={type} finish")
        del test, ranker
        gc.collect()
        print(f"type={type} finish")


if __name__ == "__main__":
    main()
