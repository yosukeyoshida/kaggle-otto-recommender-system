import gc
import pickle
import os
from typing import Any, List

import numpy as np
import pandas as pd
import polars as pl
import torch
from pydantic import BaseModel
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.model.sequential_recommender import GRU4Rec
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed

import wandb
from word2vec import calc_metrics, dump_pickle


class CFG:
    MAX_ITEM = 30
    wandb = True
    cv_only = False
    debug = False
    use_saved_dataset = True
    model_name = "gru4rec"


class ItemHistory(BaseModel):
    sequence: List[str]
    topk: int


class RecommendedItems(BaseModel):
    score_list: List[float]
    item_list: List[str]


def pred_user_to_item(item_history: ItemHistory, dataset: Any, model: Any):
    item_history_dict = item_history.dict()
    item_sequence = item_history_dict["sequence"]
    item_length = len(item_sequence)
    pad_length = CFG.MAX_ITEM  # pre-defined by recbole

    padded_item_sequence = torch.nn.functional.pad(
        torch.tensor(dataset.token2id(dataset.iid_field, item_sequence)),
        (0, pad_length - item_length),
        "constant",
        0,
    )

    input_interaction = Interaction(
        {
            "aid_list": padded_item_sequence.reshape(1, -1),
            "item_length": torch.tensor([item_length]),
        }
    )
    scores = model.full_sort_predict(input_interaction.to(model.device))
    scores = scores.view(-1, dataset.item_num)
    scores[:, 0] = -np.inf  # pad item score -> -inf
    topk_score, topk_iid_list = torch.topk(scores, item_history_dict["topk"])

    predicted_score_list = topk_score.tolist()[0]
    predicted_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.tolist()).tolist()

    recommended_items = {
        "score_list": predicted_score_list,
        "item_list": predicted_item_list,
    }
    return recommended_items


def main(cv, output_dir):
    if cv:
        train_file_path = "./input/otto-validation/*_parquet/*"
        test_file_path = "./input/otto-validation/test_parquet/*"
    else:
        train_file_path = "./input/otto-chunk-data-inparquet-format/*_parquet/*"
        test_file_path = "./input/otto-chunk-data-inparquet-format/test_parquet/*"
    if not CFG.use_saved_dataset:
        train = pl.read_parquet(train_file_path)
        # test = pl.read_parquet(test_file_path)
        if CFG.debug:
            unique_session = list(set(train["session"].unique()))
            train = train.filter(train["session"].is_in(unique_session[:200]))
            # test = test.filter(test["session"].is_in(unique_session[:200]))
        # df = pl.concat([train, test])
        train = train.sort(["session", "aid", "ts"])
        train = train.with_columns((pl.col("ts") * 1e9).alias("ts"))
        train = train.rename({"session": "session:token", "aid": "aid:token", "ts": "ts:float"})
        dataset_dir = os.path.join(output_dir, "recbox_data")
        os.makedirs(dataset_dir, exist_ok=True)
        train["session:token", "aid:token", "ts:float"].write_csv(os.path.join(dataset_dir, "recbox_data.inter"), sep="\t")
        del train
        gc.collect()

    parameter_dict = {
        "data_path": output_dir,
        "USER_ID_FIELD": "session",
        "ITEM_ID_FIELD": "aid",
        "TIME_FIELD": "ts",
        "user_inter_num_interval": "[5,inf)",
        "item_inter_num_interval": "[5,inf)",
        "load_col": {"inter": ["session", "aid", "ts"]},
        "train_neg_sample_args": None,
        "epochs": 10,
        "stopping_step": 3,
        "eval_batch_size": 1024,
        "MAX_ITEM_LIST_LENGTH": CFG.MAX_ITEM,
        "eval_args": {"split": {"RS": [9, 1, 0]}, "group_by": "user", "order": "TO", "mode": "full"},
        "save_dataset": True,
        "save_dataloaders": True,
        "checkpoint_dir": os.path.join(output_dir, "checkpoint"),
    }

    config = Config(model="GRU4Rec", dataset="recbox_data", config_dict=parameter_dict)
    # print(config)
    init_seed(config["seed"], config["reproducibility"])

    if CFG.use_saved_dataset:
        print("load dataset")
        dataset = pickle.load(open("output/gru4rec/sage-fog-266/cv/checkpoint/recbox_data-dataset.pth", "rb"))
    else:
        print("create_dataset start")
        dataset = create_dataset(config)
    print(dataset)

    print("data_preparation start")
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = GRU4Rec(config, train_data.dataset).to(config["device"])
    trainer = Trainer(config, model)
    print("train start")
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
    del train_data, valid_data
    gc.collect()
    print("train end")
    test = pl.read_parquet(test_file_path)
    test_session_AIDs = test.to_pandas().reset_index(drop=True).groupby("session")["aid"].apply(list)
    test_session_AIDs = test_session_AIDs.loc[test["session"].unique()]
    labels = []
    for AIDs in test_session_AIDs:
        AIDs = list(dict.fromkeys(AIDs))
        item = ItemHistory(sequence=AIDs, topk=20)
        try:
            nns = pred_user_to_item(item, dataset, model)["item_list"]
            labels.append(nns)
            print(f"nns len: {len(nns)}")
        except Exception as e:
            labels.append([])
    pred_df = pd.DataFrame(data={"session": test_session_AIDs.index, "labels": labels})
    dump_pickle(os.path.join(output_dir, "predictions.pkl"), pred_df)
    pred_df = pred_df.explode("labels")
    pred_df["num"] = list(range(len(pred_df)))
    pred_df["rank"] = pred_df.groupby(["session"])["num"].rank()
    pred_df["rank"] = pred_df["rank"].astype(int)
    pred_df = pred_df.rename(columns={"labels": "aid"})
    pred_df[["session", "aid", "rank"]].to_csv(os.path.join(output_dir, "pred_df.csv"), index=False)
    if cv:
        prediction_dfs = []
        for st in ["clicks", "carts", "orders"]:
            modified_predictions = pred_df.copy()
            modified_predictions["type"] = st
            prediction_dfs.append(modified_predictions)
        prediction_dfs = pd.concat(prediction_dfs).reset_index(drop=True)
        calc_metrics(prediction_dfs, output_dir)


if __name__ == "__main__":
    run_name = None
    if CFG.wandb:
        wandb.init(project="kaggle-otto", job_type=CFG.model_name)
        run_name = wandb.run.name
    if run_name is not None:
        output_dir = os.path.join(f"output/{CFG.model_name}", run_name)
    else:
        output_dir = f"output/{CFG.model_name}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "cv"), exist_ok=True)
    main(cv=True, output_dir=os.path.join(output_dir, "cv"))
    if not CFG.cv_only:
        main(cv=False, output_dir=output_dir)
