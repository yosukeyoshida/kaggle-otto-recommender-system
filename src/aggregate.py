import os
import pickle
from collections import Counter

def topk(x, k=30):
    ix = [int(v) for v in x]
    c = Counter(ix)
    return [aid for aid, cnt in c.most_common(k)]


def main(cv, output_dir):
    preds = []
    # gru4rec
    # pred1 = pickle.load(open("./output/gru4rec/dry-bush-442/cv/predictions.pkl", "rb"))
    # pred2 = pickle.load(open("./output/gru4rec/sage-spaceship-443/cv/predictions.pkl", "rb"))
    # preds += pickle.load(open("./output/gru4rec/vivid-yogurt-392/cv/predictions.pkl", "rb"))
    if cv:
        preds.append(pickle.load(open("./output/gru4rec/crisp-glitter-347/cv/predictions.pkl", "rb")))
    else:
        preds.append(pickle.load(open("./output/gru4rec/rich-sun-348/predictions.pkl", "rb")))

    # word2vec
    if cv:
        preds.append(pickle.load(open("./output/word2vec/elated-silence-293/cv/predictions.pkl", "rb")))
    else:
        preds.append(pickle.load(open("./output/word2vec/elated-silence-293/predictions.pkl", "rb")))

    # narm
    if cv:
        preds.append(pickle.load(open("./output/narm/jolly-planet-363/cv/predictions.pkl", "rb")))
    else:
        preds.append(pickle.load(open("./output/narm/giddy-voice-364/predictions.pkl", "rb")))

    # sasrec
    if cv:
        preds.append(pickle.load(open("./output/sasrec/smooth-dream-367/cv/predictions.pkl", "rb")))
    else:
        preds.append(pickle.load(open("./output/sasrec/expert-microwave-382/predictions.pkl", "rb")))

    labels_all = preds[0]["labels"]
    for pred in preds[1:]:
        labels_all += pred["labels"]

    pred_df = preds[0][["session"]]
    pred_df.loc[:, "labels"] = labels_all.apply(lambda x: topk(x, k=100))
    pred_df.to_parquet(os.path.join(output_dir, "pred_df.parquet"))
    # pred_df = pred_df.explode("labels")
    # pred_df["num"] = list(range(len(pred_df)))
    # pred_df["rank"] = pred_df.groupby(["session"])["num"].rank()
    # pred_df["rank"] = pred_df["rank"].astype(int)
    # pred_df = pred_df.rename(columns={"labels": "aid"})
    # pred_df[["session", "aid", "rank"]].to_csv(os.path.join(output_dir, "pred_df.csv"), index=False)


if __name__ == "__main__":
    output_dir = "output/aggregate"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "cv"), exist_ok=True)
    main(cv=True, output_dir=os.path.join(output_dir, "cv"))
    # main(cv=False, output_dir=output_dir)
