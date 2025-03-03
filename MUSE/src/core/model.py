import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import sys
import os
src_path = os.path.abspath('../..')
sys.path.append(src_path)
from adni_model import ADNIBackbone
from eicu_model import eICUBackbone
from mimic4_model import MIMIC4Backbone
from src.metrics import get_metrics_binary, get_metrics_multiclass


class MMLBackbone(nn.Module):
    def __init__(self, args, tokenizer=None):
        super(MMLBackbone, self).__init__()
        self.args = args

        if args.dataset == "mimic4":
            self.model = MIMIC4Backbone(
                tokenizer=tokenizer,
                embedding_size=args.embedding_size,
                code_pretrained_embedding=args.code_pretrained_embedding,
                code_layers=args.code_layers,
                code_heads=args.code_heads,
                bert_type=args.bert_type,
                dropout=args.dropout,
                rnn_layers=args.rnn_layers,
                rnn_type=args.rnn_type,
                rnn_bidirectional=args.rnn_bidirectional,
                gnn_layers=args.gnn_layers,
                gnn_norm=args.gnn_norm,
                device=args.device,
            )
        elif args.dataset == "eicu":
            self.model = eICUBackbone(
                tokenizer=tokenizer,
                embedding_size=args.embedding_size,
                code_pretrained_embedding=args.code_pretrained_embedding,
                code_layers=args.code_layers,
                code_heads=args.code_heads,
                dropout=args.dropout,
                rnn_layers=args.rnn_layers,
                rnn_type=args.rnn_type,
                rnn_bidirectional=args.rnn_bidirectional,
                ffn_layers=args.ffn_layers,
                gnn_layers=args.gnn_layers,
                gnn_norm=args.gnn_norm,
                device=args.device,
            )
        elif args.dataset == "adni":
            self.model = ADNIBackbone(
                embedding_size=args.embedding_size,
                dropout=args.dropout,
                ffn_layers=args.ffn_layers,
                gnn_layers=args.gnn_layers,
                gnn_norm=args.gnn_norm,
                device=args.device,
            )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    def train_epoch(self, data_loader):
        self.model.train()
        total_loss = []
        for i, batch in enumerate(tqdm(data_loader)):
            loss = self.model(**batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss.append(loss.item())
        return {"loss": np.mean(total_loss)}

    def eval_epoch(self, data_loader, bootstrap, output=False):
        self.model.eval()
        ids, ys, y_scores = [], [], []

        z_list = []
        x_flag_list = []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader)):
                ids.extend(batch["id"])
                y = batch["label"].to(self.args.device)
                y_score, logits, z, x_flag = self.model.inference(**batch)
                ys.append(y.cpu())
                y_scores.append(y_score.cpu())
                z_list.append(z.cpu())
                x_flag_list.append(x_flag.cpu())

        ids = np.array(ids)
        ys = torch.cat(ys, dim=0).numpy()
        y_scores = torch.cat(y_scores, dim=0).numpy()
        z_array = torch.cat(z_list, dim=0).numpy()
        x_flag_array = torch.cat(x_flag_list, dim=0).numpy()

        if self.args.num_classes == 1:
            #print("num class: 1")
            results = get_metrics_binary(ys, y_scores, bootstrap=bootstrap)
            predictions = np.stack([ids, ys, y_scores], axis=1)
        else:
            #print("num class:", self.args.num_classes)
            results = get_metrics_multiclass(ys, y_scores, bootstrap=bootstrap)
            predictions = np.concatenate([np.stack([ids, ys], axis=1), y_scores], axis=1)

        save_path = f"/root/autodl-tmp/result/eval_outputs_{self.args.task}.npz"  # 你可根据需要改成别的路径
        if output:
            np.savez(
            save_path,
            ids=ids,
            ys=ys,
            y_scores=y_scores,
            z=z_array,
            x_flag=x_flag_array
        )
        print(f"Saved inference results to {save_path}")
        return results, predictions


if __name__ == "__main__":
    from src.dataset.mimic4_dataset import MIMIC4Dataset
    from src.dataset.utils import mimic4_collate_fn
    from torch.utils.data import DataLoader
    import argparse


    def parse_arguments(parser):
        parser.add_argument("--dataset", type=str, default="mimic4")
        parser.add_argument("--num_classes", type=int, default=1)
        parser.add_argument("--embedding_size", type=int, default=128)
        parser.add_argument("--code_pretrained_embedding", type=bool, default=True)
        parser.add_argument("--code_layers", type=int, default=2)
        parser.add_argument("--code_heads", type=int, default=2)
        parser.add_argument("--bert_type", type=str, default="prajjwal1/bert-tiny")
        parser.add_argument("--rnn_layers", type=int, default=1)
        parser.add_argument("--rnn_type", type=str, default="GRU")
        parser.add_argument("--rnn_bidirectional", type=bool, default=True)
        parser.add_argument("--ffn_layers", type=int, default=2)
        parser.add_argument("--gnn_layers", type=int, default=3)
        parser.add_argument("--gnn_norm", type=str, default=None)
        parser.add_argument("--dropout", type=float, default=0.25)
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--batch_size", type=int, default=512)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-5)
        parser.add_argument("--device", type=str, default="cuda")
        args = parser.parse_args()
        return args


    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    dataset = MIMIC4Dataset(split="train", task="mortality", load_no_label=True)
    data_loader = DataLoader(dataset, batch_size=128, collate_fn=mimic4_collate_fn)
    batch = next(iter(data_loader))

    model = MMLBackbone(args=args, tokenizer=dataset.tokenizer)
    model.to("cuda")

    print(model)
    model.train_epoch(data_loader)
