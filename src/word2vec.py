from collections import defaultdict
import glob
import pickle

import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from gensim.models import Word2Vec


def dump_pickle(path, o):
    with open(path, "wb") as f:
        pickle.dump(o, f)


def read_files(path):
    dfs = []
    for file in glob.glob(path):
        df = pd.read_parquet(file)
        dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)


def main(cv):
    if cv:
        train_file_path = "./input/otto-validation/*_parquet/*"
        test_file_path = "./input/otto-validation/test_parquet/*"
    else:
        train_file_path = "./input/otto-chunk-data-inparquet-format/*_parquet/*"
        test_file_path = "./input/otto-chunk-data-inparquet-format/test_parquet/*"
    train = read_files(train_file_path)
    train["sentence"] = train.groupby("session")["aid"].apply(list)
    test = read_files(test_file_path)

    sentences = train["sentence"].to_list()

    w2vec = Word2Vec(sentences=sentences, vector_size=32, min_count=1, workers=4)

    aid2idx = {aid: i for i, aid in enumerate(w2vec.wv.index_to_key)}
    index = AnnoyIndex(32, "euclidean")

    for aid, idx in aid2idx.items():
        index.add_item(idx, w2vec.wv.vectors[idx])

    index.build(10)

    test_session_AIDs = test.groupby("session")["aid"].apply(list)
    test_session_types = test.groupby("session")["type"].apply(list)

    labels = []

    type_weight_multipliers = {0: 1, 1: 6, 2: 3}
    for AIDs, types in zip(test_session_AIDs, test_session_types):
        if len(AIDs) >= 20:
            # if we have enough aids (over equals 20) we don't need to look for candidates! we just use the old logic
            weights = np.logspace(0.1, 1, len(AIDs), base=2, endpoint=True) - 1
            aids_temp = defaultdict(lambda: 0)
            for aid, w, t in zip(AIDs, weights, types):
                aids_temp[aid] += w * type_weight_multipliers[t]

            sorted_aids = [k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]
            labels.append(sorted_aids[:20])
        else:
            # here we don't have 20 aids to output -- we will use word2vec embeddings to generate candidates!
            AIDs = list(dict.fromkeys(AIDs[::-1]))

            # let's grab the most recent aid
            most_recent_aid = AIDs[0]

            # and look for some neighbors!
            nns = [w2vec.wv.index_to_key[i] for i in index.get_nns_by_item(aid2idx[most_recent_aid], 21)[1:]]

            labels.append((AIDs + nns)[:20])

    predictions = pd.DataFrame(data={"session": test_session_AIDs.index, "labels": labels})
    if cv:
        dump_pickle("output/word2vec/predictions_cv.pkl", predictions)
    else:
        dump_pickle("output/word2vec/predictions.pkl", predictions)


if __name__ == "__main__":
    main(cv=True)
    main(cv=False)
