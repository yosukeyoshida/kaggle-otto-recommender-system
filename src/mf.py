import cudf
import os
import pandas as pd
import torch
from cuml.neighbors import NearestNeighbors
from merlin.io import Dataset
from merlin.loader.torch import Loader
from torch import nn
from torch.optim import SparseAdam


class MatrixFactorization(nn.Module):
    def __init__(self, n_aids, n_factors):
        super().__init__()
        self.aid_factors = nn.Embedding(n_aids, n_factors, sparse=True)

    def forward(self, aid1, aid2):
        aid1 = self.aid_factors(aid1)
        aid2 = self.aid_factors(aid2)

        return (aid1 * aid2).sum(dim=1)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def main(cv, output_dir):
    if cv:
        train_file_path = "../input/otto-validation/*_parquet/*"
        test_file_path = "../input/otto-validation/test_parquet/*"
    else:
        train_file_path = "../input/otto-chunk-data-inparquet-format/*_parquet/*"
        test_file_path = "../input/otto-chunk-data-inparquet-format/test_parquet/*"
    train = cudf.read_parquet(train_file_path)
    test = cudf.read_parquet(test_file_path)
    train_pairs = cudf.concat([train, test])[["session", "aid"]]
    del train, test

    train_pairs["aid_next"] = train_pairs.groupby("session").aid.shift(-1)
    train_pairs = train_pairs[["aid", "aid_next"]].dropna().reset_index(drop=True)
    cardinality_aids = max(train_pairs["aid"].max(), train_pairs["aid_next"].max())

    train_pairs[:-10_000_000].to_pandas().to_parquet("train_pairs.parquet")
    train_pairs[-10_000_000:].to_pandas().to_parquet("valid_pairs.parquet")

    train_ds = Dataset("train_pairs.parquet")
    train_dl_merlin = Loader(train_ds, 65536, True)

    valid_ds = Dataset("valid_pairs.parquet")
    valid_dl_merlin = Loader(valid_ds, 65536, True)

    num_epochs = 1
    lr = 0.1

    model = MatrixFactorization(cardinality_aids + 1, 32)
    optimizer = SparseAdam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.to("cuda")
    for epoch in range(num_epochs):
        for batch, _ in train_dl_merlin:
            model.train()
            losses = AverageMeter("Loss", ":.4e")

            aid1, aid2 = batch["aid"], batch["aid_next"]
            aid1 = aid1.to("cuda")
            aid2 = aid2.to("cuda")
            output_pos = model(aid1, aid2)
            output_neg = model(aid1, aid2[torch.randperm(aid2.shape[0])])

            output = torch.cat([output_pos, output_neg])
            targets = torch.cat([torch.ones_like(output_pos), torch.zeros_like(output_pos)])
            loss = criterion(output, targets)
            losses.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            accuracy = AverageMeter("accuracy")
            for batch, _ in valid_dl_merlin:
                aid1, aid2 = batch["aid"], batch["aid_next"]
                output_pos = model(aid1, aid2)
                output_neg = model(aid1, aid2[torch.randperm(aid2.shape[0])])
                accuracy_batch = torch.cat([output_pos.sigmoid() > 0.5, output_neg.sigmoid() < 0.5]).float().mean()
                accuracy.update(accuracy_batch, aid1.shape[0])

        print(f"{epoch + 1:02d}: * TrainLoss {losses.avg:.3f}  * Accuracy {accuracy.avg:.3f}")
    embeddings = model.aid_factors.weight.detach().cpu().numpy()
    knn = NearestNeighbors(n_neighbors=21, metric="euclidean")
    knn.fit(embeddings)
    _, aid_nns = knn.kneighbors(embeddings)
    aid_nns = aid_nns[:, 1:]
    test = cudf.read_parquet("../input/otto-full-optimized-memory-footprint/test.parquet")
    gr = test.reset_index(drop=True).to_pandas().groupby("session")
    test_session_AIDs = gr["aid"].apply(list)
    labels = []
    for AIDs in test_session_AIDs:
        AIDs = list(dict.fromkeys(AIDs[::-1]))
        most_recent_aid = AIDs[0]
        nns = list(aid_nns[most_recent_aid])
        labels.append(nns[:20])

    pred_df = pd.DataFrame(data={"session": test_session_AIDs.index, "labels": labels})
    pred_df = pred_df.explode("labels")
    pred_df["num"] = list(range(len(pred_df)))
    pred_df["rank"] = pred_df.groupby(["session"])["num"].rank()
    pred_df["rank"] = pred_df["rank"].astype(int)
    pred_df = pred_df.rename(columns={"labels": "aid"})
    pred_df[["session", "aid", "rank"]].to_csv(os.path.join(output_dir, "pred_df.csv"), index=False)
