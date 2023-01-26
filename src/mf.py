import os

import pandas as pd
import polars as pl
import torch
import wandb
from annoy import AnnoyIndex
from merlin.io import Dataset
from merlin.loader.torch import Loader
from torch import nn
from torch.optim import SparseAdam

from util import calc_metrics, dump_pickle


class CFG:
    wandb = True
    cv_only = False
    candidate_num = 20
    num_epochs = 10
    lr = 0.1


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
        train_file_path = "./input/otto-validation/*_parquet/*"
        test_file_path = "./input/otto-validation/test_parquet/*"
    else:
        train_file_path = "./input/otto-chunk-data-inparquet-format/*_parquet/*"
        test_file_path = "./input/otto-chunk-data-inparquet-format/test_parquet/*"
    train = pl.read_parquet(train_file_path)
    train_pairs = (
        train.groupby("session").agg([pl.col("aid"), pl.col("aid").shift(-1).alias("aid_next")]).explode(["aid", "aid_next"]).drop_nulls()
    )[["aid", "aid_next"]]
    cardinality_aids = max(train_pairs["aid"].max(), train_pairs["aid_next"].max())

    train_pairs[:-10_000_000].to_pandas().to_parquet(os.path.join(output_dir, "train_pairs.parquet"))
    train_pairs[-10_000_000:].to_pandas().to_parquet(os.path.join(output_dir, "valid_pairs.parquet"))

    train_ds = Dataset(os.path.join(output_dir, "train_pairs.parquet"), cpu=True)
    train_dl_merlin = Loader(train_ds, 65536, True)

    valid_ds = Dataset(os.path.join(output_dir, "valid_pairs.parquet"), cpu=True)
    valid_dl_merlin = Loader(valid_ds, 65536, True)

    model = MatrixFactorization(cardinality_aids + 1, 32)
    optimizer = SparseAdam(model.parameters(), lr=CFG.lr)
    criterion = nn.BCEWithLogitsLoss()

    print("train start")
    for epoch in range(CFG.num_epochs):
        print(f"epoch={epoch}")
        for batch, _ in train_dl_merlin:
            model.train()
            losses = AverageMeter("Loss", ":.4e")

            aid1, aid2 = batch["aid"], batch["aid_next"]
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
    dump_pickle(os.path.join(output_dir, "model.pkl"), model)
    embeddings = model.aid_factors.weight.detach().cpu().numpy()

    index = AnnoyIndex(32, "angular")
    for i, v in enumerate(embeddings):
        index.add_item(i, v)

    index.build(50)
    test = pl.read_parquet(test_file_path)
    test_session_AIDs = test.to_pandas().reset_index(drop=True).groupby("session")["aid"].apply(list)
    dump_pickle(os.path.join(output_dir, "test_session_AIDs.pkl"), test_session_AIDs)
    labels = []
    print("inference start")
    for AIDs in test_session_AIDs:
        AIDs = list(dict.fromkeys(AIDs[::-1]))
        most_recent_aid = AIDs[0]
        nns = index.get_nns_by_item(most_recent_aid, CFG.candidate_num)
        labels.append(nns)
    pred_df = pd.DataFrame(data={"session": test_session_AIDs.index, "labels": labels})
    dump_pickle(os.path.join(output_dir, "predictions.pkl"), pred_df)
    pred_df = pred_df.explode("labels")
    pred_df["num"] = list(range(len(pred_df)))
    pred_df["rank"] = pred_df.groupby(["session"])["num"].rank()
    pred_df["rank"] = pred_df["rank"].astype(int)
    pred_df = pred_df.rename(columns={"labels": "aid"})
    pred_df[["session", "aid", "rank"]].to_csv(os.path.join(output_dir, "pred_df.csv"), index=False)
    print("calc metrics start")
    if cv:
        prediction_dfs = []
        for st in ["clicks", "carts", "orders"]:
            modified_predictions = pred_df.copy()
            modified_predictions["type"] = st
            prediction_dfs.append(modified_predictions)
        prediction_dfs = pd.concat(prediction_dfs).reset_index(drop=True)
        calc_metrics(prediction_dfs, output_dir, CFG.candidate_num, CFG.wandb)


if __name__ == "__main__":
    run_name = None
    if CFG.wandb:
        wandb.init(project="kaggle-otto", job_type="mf")
        run_name = wandb.run.name
    if run_name is not None:
        output_dir = os.path.join("output/mf", run_name)
    else:
        output_dir = "output/mf"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "cv"), exist_ok=True)
    main(cv=True, output_dir=os.path.join(output_dir, "cv"))
    if not CFG.cv_only:
        main(cv=False, output_dir=output_dir)
