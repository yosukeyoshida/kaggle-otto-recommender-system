import pickle
import pandas as pd
import os
import gc


def main():
    output_dir = "output/ensemble"
    df_list = []
    for type in ["clicks", "carts", "orders"]:
        ids = ["royal-vortex-636", "valiant-grass-637", "wandering-rain-638", "vivid-brook-639", "misunderstood-waterfall-640"]
        df = pickle.load(open(f"./output/lgbm/{ids[0]}/preds_clicks.pkl", "rb"))
        for id in ids:
            _df = pickle.load(open(f"./output/lgbm/{id}/preds_clicks.pkl", "rb"))
            df["score"] += _df["score"]
            del _df
            gc.collect()
        # df["score"] = df["score"] / len(ids)
        df = df.sort_values(["session", "score"]).groupby("session").tail(20)
        df = df.groupby("session")["aid"].apply(list)
        df = df.to_frame().reset_index()
        df["session_type"] = df["session"].apply(lambda x: str(x) + f"_{type}")
        df_list.append(df)
        del df
        gc.collect()
    sub = pd.concat(df_list)
    sub["labels"] = sub["aid"].apply(lambda x: " ".join(map(str, x)))
    sub[["session_type", "labels"]].to_csv(os.path.join(output_dir, "submission.csv"), index=False)


if __name__ == "__main__":
    main()
