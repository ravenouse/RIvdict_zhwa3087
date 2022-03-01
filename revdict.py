import argparse
import itertools
import json
import logging
import pathlib
import sys
import os


logger = logging.getLogger(pathlib.Path(__file__).name)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logger.addHandler(handler)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import tqdm

import data
import models

import math
import multi_task_optimization as mto

def get_parser(
    parser=argparse.ArgumentParser(
        description="Run a reverse dictionary baseline.\nThe task consists in reconstructing an embedding from the glosses listed in the datasets"
    ),
):
    parser.add_argument(
        "--do_train", action="store_true", help="whether to train a model from scratch"
    )
    parser.add_argument(
        "--do_pred", action="store_true", help="whether to produce predictions"
    )
    parser.add_argument(
        "--train_file", type=pathlib.Path, help="path to the train file"
    )
    parser.add_argument("--dev_file", type=pathlib.Path, help="path to the dev file")
    parser.add_argument("--test_file", type=pathlib.Path, help="path to the test file")
    parser.add_argument("--tokenizer", type=pathlib.Path, help="path to tokenizer")
    parser.add_argument(
        "--device",
        type=torch.device,
        default=torch.device("cuda") if torch.cuda.is_available() else "cpu",
        help="path to the train file",
    )
    parser.add_argument(
        "--target_arch",
        type=str,
        default="sgns",
        choices=("sgns", "char", "electra"),
        help="embedding architecture to use as target",
    )
    parser.add_argument(
        "--summary_logdir",
        type=pathlib.Path,
        default=pathlib.Path("logs") / f"revdict-baseline",
        help="write logs for future analysis",
    )
    parser.add_argument(
        "--save_dir",
        type=pathlib.Path,
        default=pathlib.Path("models") / f"revdict-baseline",
        help="where to save model & vocab",
    )
    parser.add_argument(
        "--pred_dir",
        type=pathlib.Path,
        default=pathlib.Path("revdict-baseline-preds.json"),
        help="where to save predictions",
    )
    parser.add_argument(
        "--if_electra",
        default=True,
        help="Choose whether predict sgns and char or electra",
    )
    parser.add_argument(
        "--continue_training",
        default='no',
        help="Choose whether continue training or not",
    )
    return parser


def train(args):
    assert args.train_file is not None, "Missing dataset for training"
    # 1. get data, vocabulary, summary writer
    logger.debug("Preloading data")
    ## make datasets
    train_dataset = data.JSONDataset(args.train_file,tokenizer=args.tokenizer)
    if args.dev_file:
        dev_dataset = data.JSONDataset(args.dev_file, tokenizer=args.tokenizer,vocab=train_dataset.vocab)
    ## assert they correspond to the task
    assert train_dataset.has_gloss, "Training dataset contains no gloss."
    if args.target_arch == "electra":
        assert train_dataset.has_electra, "Training datatset contains no vector."
    else:
        assert train_dataset.has_vecs, "Training datatset contains no vector."
    if args.dev_file:
        assert dev_dataset.has_gloss, "Development dataset contains no gloss."
        if args.target_arch == "electra":
            assert dev_dataset.has_electra, "Development dataset contains no vector."
        else:
            assert dev_dataset.has_vecs, "Development dataset contains no vector."
    ## make dataloader
    train_dataloader = data.get_dataloader(train_dataset, batch_size=512)
    dev_dataloader = data.get_dataloader(dev_dataset, shuffle=False, batch_size=1024)
    ## make summary writer
    summary_writer = SummaryWriter(args.summary_logdir)
    train_step = itertools.count()  # to keep track of the training steps for logging

    # 2. construct model
    ## Hyperparams
    logger.debug("Setting up training environment")
    if args.continue_training == 'no':
        print('will train the model from begining')
        model = models.RevdictModel(dev_dataset.new_vocab).to(args.device)
    else:
        print('will continue to train the model from last time')
        model = models.DefmodModel.load(args.save_dir / "model.pt").to(args.device)
    model.train()

    # 3. declare optimizer & criterion
    ## Hyperparams
    EPOCHS, LEARNING_RATE, BETA1, BETA2, WEIGHT_DECAY = 15, 1.0e-4, 0.9, 0.999, 1.0e-6
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.MSELoss()


    vec_tensor_key1 = "sgns_tensor"
    vec_tensor_key2 = "char_tensor"
    if args.if_electra:
        vec_tensor_key3 ='electra_tensor'


    # 4. train model
    for epoch in tqdm.trange(EPOCHS, desc="Epochs"):
        ## train loop
        pbar = tqdm.tqdm(
            desc=f"Train {epoch}", total=len(train_dataset), disable=None, leave=False
        )

        loss_tt = [1,1,1]
        loss_t = [1,1,1]
        for batch in train_dataloader:
            optimizer.zero_grad()
            data.retokenize(batch,args.tokenizer)
            gls = batch["new_gloss_tensor"].to(args.device)
            vec1 = batch[vec_tensor_key1].to(args.device)
            vec2 = batch[vec_tensor_key2].to(args.device)
            if args.if_electra:
                vec3 = batch[vec_tensor_key3].to(args.device)
            pred1,pred2,pred3 = model(gls)
            loss1 = criterion(pred1, vec1)
            loss2 = criterion(pred2,vec2)
            if args.if_electra:
                loss3=criterion(pred3,vec3)
            else:
                loss3=0

            wa,wb,wc = mto.DWA(loss_tt,loss_t,if_electra=args.if_electra)

            loss = wa*loss1+wb*loss2+wc*loss3
            loss.backward()

            loss_tt = loss_t
            loss_t = [loss1,loss2,loss3]

            # keep track of the train loss for this step
            next_step = next(train_step)
            summary_writer.add_scalar(
                "revdict-train/cos",
                F.cosine_similarity(pred1, vec1).mean().item(),
                next_step,
            )
            summary_writer.add_scalar("revdict-train/mse", loss.item(), next_step)
            optimizer.step()
            pbar.update(vec1.size(0))
        pbar.close()
        ## eval loop
        if args.dev_file:
            model.eval()
            with torch.no_grad():
                sum_dev_loss1, sum_dev_loss2,sum_dev_loss3, sum_cosine = 0.0,0.0, 0.0, 0.0
                pbar = tqdm.tqdm(
                    desc=f"Eval {epoch}",
                    total=len(dev_dataset),
                    disable=None,
                    leave=False,
                )
                for batch in dev_dataloader:
                    data.retokenize(batch,args.tokenizer)
                    gls = batch["new_gloss_tensor"].to(args.device)

                    vec1 = batch[vec_tensor_key1].to(args.device)
                    vec2 = batch[vec_tensor_key2].to(args.device)
                    if args.if_electra:
                        vec3 = batch[vec_tensor_key3].to(args.device)

                    pred = model(gls)


                    pred1, pred2, pred3 = model(gls)
                    sum_dev_loss1 += (
                        F.mse_loss(pred1, vec1, reduction="none").mean(1).sum().item()
                    )
                    sum_dev_loss2 += (
                        F.mse_loss(pred2, vec2, reduction="none").mean(1).sum().item()
                    )
                    if args.if_electra:
                        sum_dev_loss3 += (
                            F.mse_loss(pred3, vec3, reduction="none").mean(1).sum().item()
                        )
                    sum_cosine += F.cosine_similarity(pred1, vec1).sum().item()
                    pbar.update(vec1.size(0))
                # keep track of the average loss on dev set for this epoch
                summary_writer.add_scalar(
                    "revdict-dev/cos", sum_cosine / len(dev_dataset), epoch
                )
                summary_writer.add_scalar(
                    "revdict-dev/mse", sum_dev_loss1 / len(dev_dataset), epoch
                )
                summary_writer.add_scalar(
                    "revdict-dev/mse2", sum_dev_loss2 / len(dev_dataset), epoch
                )
                summary_writer.add_scalar(
                    "revdict-dev/mse3", sum_dev_loss3 / len(dev_dataset), epoch
                )
                pbar.close()
            model.train()

    # 5. save result
    model.save(args.save_dir / "model.pt")
    train_dataset.save(args.save_dir / "train_dataset.pt")
    dev_dataset.save(args.save_dir / "dev_dataset.pt")


def pred(args):
    assert args.test_file is not None, "Missing dataset for test"
    # 1. retrieve vocab, dataset, model
    model = models.DefmodModel.load(args.save_dir / "model.pt")
    train_vocab = data.JSONDataset.load(args.save_dir / "train_dataset.pt").vocab
    test_dataset = data.JSONDataset(
        args.test_file, tokenizer=args.tokenizer,vocab=train_vocab, freeze_vocab=True, maxlen=512
    )
    test_dataloader = data.get_dataloader(test_dataset, shuffle=False, batch_size=1024)
    model.eval()

    vec_tensor_key1 = "sgns_tensor"
    vec_tensor_key2 = "char_tensor"
    if test_dataset.has_electra:
        vec_tensor_key3 = "electra_tensor"
    assert test_dataset.has_gloss, "File is not usable for the task"
    # 2. make predictions
    predictions1 = []
    predictions2 = []
    if args.if_electra:
        predictions3 = []
    with torch.no_grad():
        pbar = tqdm.tqdm(desc="Pred.", total=len(test_dataset))
        for batch in test_dataloader:
            data.retokenize(batch,args.tokenizer)
            vecs1,vecs2,vecs3 = model(batch["new_gloss_tensor"].to(args.device))
            for id, vec in zip(batch["id"], vecs1.unbind()):
                predictions1.append(
                    {"id": id, "sgns": vec.view(-1).cpu().tolist()}
                )
            pbar.update(vecs1.size(0))

            for id, vec in zip(batch["id"], vecs2.unbind()):
                predictions2.append(
                    {"id": id, "char": vec.view(-1).cpu().tolist()}
                )
            pbar.update(vecs2.size(0))

            if args.if_electra:
                for id, vec in zip(batch["id"], vecs3.unbind()):
                    predictions3.append(
                        {"id": id, "electra": vec.view(-1).cpu().tolist()}
                    )
                pbar.update(vecs3.size(0))

        pbar.close()
    with open(f'{args.pred_dir}/predictions1.json', "w") as ostr:
        json.dump(predictions1, ostr)

    with open(f'{args.pred_dir}/predictions2.json', "w") as ostr:
        json.dump(predictions2, ostr)
    if args.if_electra:
        with open(f'{args.pred_dir}/predictions3.json', "w") as ostr:
            json.dump(predictions3, ostr)

def main(args):
    if args.do_train:
        logger.debug("Performing revdict training")
        train(args)
    if args.do_pred:
        logger.debug("Performing revdict prediction")
        pred(args)


if __name__ == "__main__":
    args = get_parser().parse_args()
    args.if_electra = bool(args.if_electra)
    print(type(args.if_electra))
    print(args.if_electra)
    main(args)
