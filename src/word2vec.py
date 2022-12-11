import glob
import wandb
import os
import pickle

import pandas as pd
from annoy import AnnoyIndex
from gensim.models import Word2Vec


class CFG:
    calc_metrics = True
    wandb = True
    cv_only = True
    use_model = True


def dump_pickle(path, o):
    with open(path, "wb") as f:
        pickle.dump(o, f)


def read_files(path):
    dfs = []
    for file in glob.glob(path):
        df = pd.read_parquet(file)
        dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)


def calc_metrics(pred_df):
    score = 0
    weights = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}
    for t in ["clicks", "carts", "orders"]:
        sub = pred_df.loc[pred_df["type"] == t].copy()
        sub = sub.groupby("session")["aid"].apply(list)
        test_labels = pd.read_parquet("./input/otto-validation/test_labels.parquet")
        test_labels = test_labels.loc[test_labels["type"] == t]
        test_labels = test_labels.merge(sub, how="left", on=["session"])
        test_labels = test_labels[test_labels["aid"].notnull()]
        test_labels["aid"] = test_labels["aid"].apply(lambda x: x[:20])
        test_labels["hits"] = test_labels.apply(lambda df: len(set(df["ground_truth"]).intersection(set(df["aid"]))), axis=1)
        test_labels["gt_count"] = test_labels.ground_truth.str.len().clip(0, 20)
        test_labels["recall"] = test_labels["hits"] / test_labels["gt_count"]
        recall = test_labels["hits"].sum() / test_labels["gt_count"].sum()
        score += weights[t] * recall
        dump_pickle(os.path.join(output_dir, f"test_labels_{t}.pkl"), test_labels)
        print(f"{t} recall={recall}")
        if CFG.wandb:
            wandb.log({f"{t} recall": recall})
    print(f"total recall={score}")
    if CFG.wandb:
        wandb.log({f"total recall": score})
    return score



def main(cv, output_dir):
    if cv:
        train_file_path = "./input/otto-validation/*_parquet/*"
        test_file_path = "./input/otto-validation/test_parquet/*"
    else:
        train_file_path = "./input/otto-chunk-data-inparquet-format/*_parquet/*"
        test_file_path = "./input/otto-chunk-data-inparquet-format/test_parquet/*"
    train = read_files(train_file_path)
    sentences = train.groupby("session")["aid"].apply(list).to_list()
    test = read_files(test_file_path)

    if CFG.use_model:
        w2vec = Word2Vec(sentences=sentences, vector_size=32, min_count=1, workers=4)
        w2vec.save(os.path.join(output_dir, "w2vec.model"))
    else:
        w2vec = Word2Vec.load("w2vec.model")
    test_session_AIDs = test.groupby("session")["aid"].apply(list)
    labels = []
    for AIDs in test_session_AIDs:
        AIDs = list(dict.fromkeys(AIDs[::-1]))
        most_recent_aid = AIDs[-5:]
        nns = [i for i, score in w2vec.predict_output_word(most_recent_aid, topn=20)]
        labels.append(nns)
    pred_df = pd.DataFrame(data={"session": test_session_AIDs.index, "labels": labels})
    dump_pickle(os.path.join(output_dir, "predictions.pkl"), pred_df)
    pred_df = pred_df.explode("labels")
    pred_df["num"] = list(range(len(pred_df)))
    pred_df["rank"] = pred_df.groupby(["session"])["num"].rank()
    pred_df["rank"] = pred_df["rank"].astype(int)
    pred_df = pred_df.rename(columns={"labels": "aid"})
    pred_df[["session", "aid", "rank"]].to_csv(os.path.join(output_dir, "pred_df.csv"), index=False)
    if CFG.calc_metrics:
        prediction_dfs = []
        for st in ["clicks", "carts", "orders"]:
            modified_predictions = pred_df.copy()
            modified_predictions["type"] = st
            prediction_dfs.append(modified_predictions)
        prediction_dfs = pd.concat(prediction_dfs).reset_index(drop=True)
        calc_metrics(prediction_dfs)


if __name__ == "__main__":
    run_name = None
    if CFG.wandb:
        wandb.init(project="kaggle-otto", job_type="word2vec")
        run_name = wandb.run.name
    if run_name is not None:
        output_dir = os.path.join("output/word2vec", run_name)
    else:
        output_dir = "output/word2vec"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "cv"), exist_ok=True)
    main(cv=True, output_dir=os.path.join(output_dir, "cv"))
    if not CFG.cv_only:
        main(cv=False, output_dir=output_dir)
