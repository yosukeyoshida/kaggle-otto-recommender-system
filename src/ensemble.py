import pandas as pd
from collections import Counter

df0 = pd.read_csv("./output/lgbm/honest-plant-635/submission.csv")
df1 = pd.read_csv("./output/lgbm/winter-rain-631/submission.csv")
# df2 = pd.read_csv("./output/lgbm/fragrant-energy-603/submission.csv")
# df3 = pd.read_csv("./output/lgbm/stellar-oath-604/submission.csv")
# df4 = pd.read_csv("../output/lgbm/soft-deluge-574/submission.csv")

df0["labels"] = df0["labels"].apply(lambda x: x.split())
df1["labels"] = df1["labels"].apply(lambda x: x.split())
# df2["labels"] = df2["labels"].apply(lambda x: x.split())
# df3["labels"] = df3["labels"].apply(lambda x: x.split())

joined = df0.merge(df1, on="session_type").rename(columns={"labels_x": "labels0", "labels_y": "labels1"})
# joined = joined.merge(df2, on="session_type").rename(columns={"labels": "labels2"})
# joined = joined.merge(df3, on="session_type").rename(columns={"labels": "labels3"})

# FIXME
joined["labels"] = joined.apply(lambda x: Counter(x["labels0"]+x["labels1"]).most_common(20), axis=1)
joined["labels"] = joined["labels"].apply(lambda x: " ".join([v[0] for v in x]))
joined[["session_type", "labels"]].to_csv("output/ensemble/submission.csv", index=False)
