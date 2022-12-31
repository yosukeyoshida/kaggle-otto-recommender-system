import os
import argparse
import pickle
from collections import Counter

def topk(x, k=30):
    ix = [int(v) for v in x]
    c = Counter(ix)
    return [aid for aid, cnt in c.most_common(k)]


def main(cv, model_name, output_dir):
    preds = []
    # gru4rec
    if model_name == "gru4rec":
        if cv:
            preds.append(pickle.load(open("./output/gru4rec/crisp-glitter-347/cv/predictions.pkl", "rb")))
            preds.append(pickle.load(open("./output/gru4rec/dry-bush-442/cv/predictions.pkl", "rb")))
            preds.append(pickle.load(open("./output/gru4rec/sage-spaceship-443/cv/predictions.pkl", "rb")))
            preds.append(pickle.load(open("./output/gru4rec/eager-galaxy-445/cv/predictions.pkl", "rb")))
        else:
            preds.append(pickle.load(open("./output/gru4rec/rich-sun-348/predictions.pkl", "rb")))
            preds.append(pickle.load(open("./output/gru4rec/pretty-wildflower-447/predictions.pkl", "rb")))
            preds.append(pickle.load(open("./output/gru4rec/gentle-silence-458/predictions.pkl", "rb")))
            preds.append(pickle.load(open("./output/gru4rec/proud-butterfly-465/predictions.pkl", "rb")))
    elif model_name == "narm":
        if cv:
            preds.append(pickle.load(open("./output/narm/jolly-planet-363/cv/predictions.pkl", "rb")))
            preds.append(pickle.load(open("./output/narm/dry-donkey-448/cv/predictions.pkl", "rb")))
            preds.append(pickle.load(open("./output/narm/silvery-snow-455/cv/predictions.pkl", "rb")))
            preds.append(pickle.load(open("./output/narm/charmed-hill-456/cv/predictions.pkl", "rb")))
        else:
            preds.append(pickle.load(open("./output/narm/giddy-voice-364/predictions.pkl", "rb")))
            preds.append(pickle.load(open("./output/narm/worldly-rain-464/predictions.pkl", "rb")))
            preds.append(pickle.load(open("./output/narm/comfy-durian-466/predictions.pkl", "rb")))
            preds.append(pickle.load(open("./output/narm/eager-wave-472/predictions.pkl", "rb")))
    # # word2vec
    # if cv:
    #     preds.append(pickle.load(open("./output/word2vec/elated-silence-293/cv/predictions.pkl", "rb")))
    # else:
    #     preds.append(pickle.load(open("./output/word2vec/elated-silence-293/predictions.pkl", "rb")))
    #
    #
    # # sasrec
    # if cv:
    #     preds.append(pickle.load(open("./output/sasrec/smooth-dream-367/cv/predictions.pkl", "rb")))
    # else:
    #     preds.append(pickle.load(open("./output/sasrec/expert-microwave-382/predictions.pkl", "rb")))

    labels_all = preds[0]["labels"]
    for pred in preds[1:]:
        labels_all += pred["labels"]

    pred_df = preds[0][["session"]]
    pred_df.loc[:, "labels"] = labels_all.apply(lambda x: topk(x, k=40))
    pred_df.to_parquet(os.path.join(output_dir, "pred_df.parquet"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    for model_name in ["gru4rec", "narm"]:
        print(model_name)
        output_dir = f"output/aggregate/{model_name}"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "cv"), exist_ok=True)
        main(cv=True, model_name=model_name, output_dir=os.path.join(output_dir, "cv"))
        main(cv=False, model_name=model_name, output_dir=output_dir)
