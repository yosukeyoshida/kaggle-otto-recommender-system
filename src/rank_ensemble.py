import pickle
import os
import pandas as pd
import gc


def main(output_dir):
    dfs = []
    for i in range(20):
        df1 = pickle.load(open(f"absurd-violet-724/preds/preds_{i}.pkl", "rb"))
        df2 = pickle.load(open(f"splendid-mountain-725/preds/preds_{i}.pkl", "rb"))
        for type in ["clicks", "carts", "orders"]:
            _df1 = df1[df1["type"] == type]
            _df2 = df2[df2["type"] == type]
            _df1["rank"] = _df1.groupby(["session", "type"])["score"].rank(ascending=False)
            _df2["rank"] = _df2.groupby(["session", "type"])["score"].rank(ascending=False)
            joined = _df1.merge(_df2, on=["session", "aid", "type"])
            del _df1, _df2
            gc.collect()
            joined["rank"] = (joined["rank_x"] + joined["rank_y"]) / 2
            _preds = joined.sort_values(["session", "rank"]).groupby("session").head(20)
            _preds = _preds.groupby("session")["aid"].apply(list)
            _preds = _preds.to_frame().reset_index()
            _preds["session_type"] = _preds["session"].apply(lambda x: str(x) + f"_{type}")
            dfs.append(_preds)
            del _preds
            gc.collect()
    sub = pd.concat(dfs)
    sub["labels"] = sub["aid"].apply(lambda x: " ".join(map(str, x)))
    sub[["session_type", "labels"]].to_csv(os.path.join(output_dir, "submission.csv"), index=False)


if __name__ == "__main__":
    output_dir = "output/ensemble/20230118"
    os.makedirs(output_dir, exist_ok=True)
    main(output_dir=output_dir)