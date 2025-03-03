import argparse
import sys

sys.path.append("..")

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import os
import os.path as osp
import timeit

from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

#################################
# 1) 导入你自定义的数据集和工具函数
#################################
from MUSE.src.dataset.mimic4_dataset import MIMIC4Dataset
from MUSE.src.dataset.utils import mimic4_collate_fn
# 如果你有自己的 engine 代码，可以保留
from engine import Engine

# 导入原先定义的损失（特别是 domain loss、distribution loss 等）
import loss_Dual as loss

# 这里是原先用于共享-特定特征学习的网络结构，可自行修改
# 如果你已修改 DualNet_SS.py，使之适配三模态 (text, code, lab)，可直接引用
from model_new import DualNet_SS

start = timeit.default_timer()

# 两个超参，可根据需要自行调整
alpha = 0.1  # shared domain loss weight
beta = 0.02  # specific domain loss weight


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def replace_zero_vectors(shared_info):
    """
    检查 shared_info 是否包含全 0 向量，并用其他模态的均值替换
    """
    shared_info = torch.stack(shared_info)  # [3, B, hidden_dim]
    zero_mask = (shared_info == 0).all(dim=(1, 2))  # [3]，检查每个模态是否全 0

    # 计算非零模态的均值
    non_zero_vectors = shared_info[~zero_mask]  # 选出非零模态
    if len(non_zero_vectors) > 0:
        mean_vector = non_zero_vectors.mean(dim=0)  # [B, hidden_dim]
    else:
        mean_vector = torch.zeros_like(shared_info[0])  # 如果所有模态都是 0，则保留全 0

    # 替换全 0 模态的向量
    for i in range(3):
        if zero_mask[i]:  # 如果该模态是全 0
            shared_info[i] = mean_vector  # 替换为均值

    return [shared_info[0], shared_info[1], shared_info[2]]  # 返回修正后的 shared_info

def get_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="Shared-Specific model for MIMIC-IV readmission/mortality prediction.")

    # 通用参数
    parser.add_argument("--snapshot_dir", type=str, default='snapshots/example/')
    parser.add_argument("--reload_path", type=str, default='snapshots/example/last.pth')
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=False)

    # 数据集与任务相关参数
    parser.add_argument("--task", type=str, default='mortality',
                        help="Task name: 'mortality' or 'readmission'")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--load_no_label", type=str2bool, default=False,
                        help="Whether to load no-label samples (usually not used).")

    # 训练超参数
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--start_iters", type=int, default=0)
    parser.add_argument("--val_pred_every", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--random_seed", type=int, default=999)

    # 模式设置（是否仅训练、不评估等）
    parser.add_argument("--train_only", action="store_true")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--mode", type=str, default='full',
                        help="If you want random missing modality, set mode='random', etc.")

    #TODO 来自MUSE
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--code_pretrained_embedding", type=bool, default=True)
    parser.add_argument("--code_layers", type=int, default=2)
    parser.add_argument("--code_heads", type=int, default=2)
    parser.add_argument("--bert_type", type=str, default="prajjwal1/bert-tiny")
    parser.add_argument("--rnn_layers", type=int, default=1)
    parser.add_argument("--rnn_type", type=str, default="GRU")
    parser.add_argument("--rnn_bidirectional", type=bool, default=True)
    parser.add_argument("--gnn_layers", type=int, default=2)
    parser.add_argument("--gnn_norm", type=str, default=None)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--dev", action="store_true", default=False)


    return parser


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, lr, num_steps, power):
    """多项式学习率衰减"""
    lr = lr_poly(lr, i_iter, num_steps, power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


########################################
# 2) 验证函数(二分类指标)
########################################
def evaluate_model(model, dataloader, device):
    """
    计算二分类的准确率、F1、PR AUC 和 ROC AUC。
    """
    model.eval()
    correct, total = 0, 0
    all_labels = []
    all_preds = []
    all_probs = []

    #x_flag_list = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["label"].to(device)  # shape: (B,)
            #flags = batch["label_flag"].to(device)

            # 三模态输入：对应于 DualNet_SS.forward() 中的调用
            logits, shared_info, spec_info = model(
                codes=batch["codes"].to(device),
                labvectors=batch["labvectors"].to(device),
                discharge=batch["discharge"],
            )
            # 根据输出维度判断使用哪种方式
            if logits.size(-1) == 1:
                # 二分类(以BCE形式输出)，计算 sigmoid 得到概率
                probs = torch.sigmoid(logits).view(-1)  # [B]
                preds = (probs > 0.5).long()
            else:
                # 如果输出为2类 softmax，此处取第二列的概率（正类）
                probs = F.softmax(logits, dim=-1)[:, 1]
                preds = torch.argmax(logits, dim=-1)

            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = 100.0 * correct / total
    f1 = f1_score(all_labels, all_preds)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = float('nan')
    pr_auc = average_precision_score(all_labels, all_probs)

    model.train()
    print(f"Accuracy: {acc:.2f}% | F1: {f1:.4f} | ROC AUC: {roc_auc:.4f} | PR AUC: {pr_auc:.4f}")
    return acc, f1, roc_auc, pr_auc


def main():
    parser = get_arguments()
    args = parser.parse_args()

    # 如果你有自定义的分布式引擎，可以保留
    with Engine(custom_parser=parser) as engine:
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda:{}'.format(args.local_rank))

        # 设置随机种子
        cudnn.benchmark = True
        seed = args.random_seed
        if engine.distributed:
            seed = args.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # TensorBoard 日志
        writer = SummaryWriter(args.snapshot_dir)
        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

        ########################################
        # 3) 初始化数据集 & DataLoader
        ########################################
        train_dataset = MIMIC4Dataset(split="train",
                                      task=args.task,
                                      dev=args.dev,
                                      load_no_label=args.load_no_label)
        val_dataset = MIMIC4Dataset(split="val",
                                    task=args.task)

        tokenizer = train_dataset.tokenizer

        trainloader, train_sampler = engine.get_train_loader(
            train_dataset,
            collate_fn=mimic4_collate_fn,
        )
        valloader, val_sampler = engine.get_test_loader(
            val_dataset
        )

        ########################################
        # 4) 初始化模型
        ########################################
        # 假设你已修改 DualNet_SS，可直接处理三模态输入并输出二分类
        model = DualNet_SS(args, tokenizer)
        model.to(device)

        # 如有分布式，多GPU
        if args.num_gpus > 1:
            model = engine.data_parallel(model)

        ########################################
        # 5) 如果需要从 checkpoint 恢复
        ########################################
        if args.reload_from_checkpoint:
            if os.path.exists(args.reload_path):
                checkpoint = torch.load(args.reload_path, map_location='cpu')
                model = checkpoint['model']
                args.start_iters = checkpoint['iter']
                print(f"[INFO] Loaded model from {args.reload_path}, trained for {args.start_iters} iters.")
            else:
                print(f"[ERROR] File not found: {args.reload_path}")
                exit(0)

        ########################################
        # 6) 定义损失函数 & 优化器
        ########################################
        # 二分类损失: BCEWithLogitsLoss or CrossEntropyLoss
        # 如果你的输出是 [B,1], 用 BCEWithLogitsLoss
        criterion_cls = nn.BCEWithLogitsLoss()

        # 域分类损失 + Distribution Alignment 损失 (保留原来的)
        loss_domain_cls = loss.DomainClsLoss().to(device)
        distribution_loss = nn.L1Loss()

        # 优化器
        optimizer = optim.SGD(model.parameters(),
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay,
                              nesterov=True)

        ########################################
        # 7) 主训练循环
        ########################################
        print("[INFO] Start training ...")
        model.train()
        best_acc = -1
        for i_iter, batch in enumerate(trainloader):
            i_iter += args.start_iters
            if i_iter >= args.num_steps:
                break

            # 学习率衰减
            lr = adjust_learning_rate(optimizer, i_iter, args.learning_rate, args.num_steps, args.power)

            # 取出数据
            labels = batch["label"].to(device)  # [B]
            #codes = batch["codes"].to(device)
            #labs = batch["labvectors"].to(device)
            #discharge = batch["discharge"].to(device)  # 可能是 list[str] 或张量，看你的 TextEncoder 实现

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播：返回分类输出、共享特征、模态特定特征等
            logits, shared_info, spec_info = model(
                **batch
            )
            # logits.shape = [B,1] or [B,2]

            # -------------------------
            # 计算核心任务的二分类损失
            # -------------------------
            if logits.size(-1) == 1:
                # [B,1] -> BCE
                task_loss = criterion_cls(logits.view(-1), labels.float())
            else:
                # [B,2] -> CE
                task_loss = nn.CrossEntropyLoss()(logits, labels.long())

            # -------------------------
            # Distribution Alignment
            # -------------------------
            # 如果你实现了三模态 "text, code, lab" 的共享特征，假设 shared_info = [text_shared, code_shared, lab_shared]
            # 可进行 pairwise 计算:
            """if len(shared_info) == 3:
                term_shared = (distribution_loss(shared_info[0], shared_info[1]) +
                               distribution_loss(shared_info[1], shared_info[2]) +
                               distribution_loss(shared_info[2], shared_info[0]))
            else:
                term_shared = 0.0"""

            assert len(shared_info)==3, f"########len(shared_info)=={len(shared_info)}"

            # 在计算 loss 之前，先修正 shared_info
            shared_info = replace_zero_vectors(shared_info)
            term_shared = (distribution_loss(shared_info[0], shared_info[1]) +
                           distribution_loss(shared_info[1], shared_info[2]) +
                           distribution_loss(shared_info[2], shared_info[0]))


            # -------------------------
            # Domain Classification
            # -------------------------
            # 3 个模态 -> label = 0/1/2
            # 参考原有做法，需要把 spec_info 拼到一起，并生成对应标签
            # 以下是个示例:
            if isinstance(spec_info, list) and len(spec_info) == 3:
                B = spec_info[0].size(0)
                # 拼接特征
                cat_spec = torch.cat(spec_info, dim=0)  # [3B, feat_dim]
                # 域标签
                spec_labels = torch.cat([
                    torch.zeros(B, dtype=torch.long),
                    torch.ones(B, dtype=torch.long),
                    2 * torch.ones(B, dtype=torch.long)
                ], dim=0).to(device)
                term_spec = loss_domain_cls(cat_spec, spec_labels)
            else:
                term_spec = 0.0

            # -------------------------
            # 总损失
            # -------------------------
            loss_all = task_loss + alpha * term_shared + beta * term_spec
            loss_all.backward()
            optimizer.step()

            # -------------------------
            # 打印 & 记录日志
            # -------------------------
            if i_iter % 100 == 0 and args.local_rank == 0:
                writer.add_scalar('learning_rate', lr, i_iter)
                writer.add_scalar('loss_all', loss_all.item(), i_iter)
                writer.add_scalar('task_loss', task_loss.item(), i_iter)
                writer.add_scalar('term_shared', term_shared if isinstance(term_shared, float) else term_shared.item(),
                                  i_iter)
                writer.add_scalar('term_spec', term_spec if isinstance(term_spec, float) else term_spec.item(), i_iter)

                print(f"Iter={i_iter}/{args.num_steps}, lr={lr:.6f}, task_loss={task_loss.item():.4f}, "
                      f"shared_loss={term_shared}, spec_loss={term_spec}, total={loss_all.item():.4f}")

            # -------------------------
            # 验证 & 保存模型
            # -------------------------
            if (not args.train_only) and (i_iter % args.val_pred_every == (args.val_pred_every - 1)):
                print("[INFO] Validating ...")
                val_acc, f1, roc_auc, pr_auc = evaluate_model(model, valloader, device)
                writer.add_scalar('Val_Accuracy', val_acc, i_iter)
                print(f"Validation @ iter {i_iter}: accuracy = {val_acc:.2f}, f1 = {f1}, roc_auc = {roc_auc}, pr_auc = {pr_auc}")

                # 保存 best
                if val_acc > best_acc and args.local_rank == 0:
                    best_acc = val_acc
                    print("[INFO] Saving best model ...")
                    checkpoint = {
                        'model': model,
                        'optimizer': optimizer.state_dict(),
                        'iter': i_iter
                    }
                    torch.save(checkpoint, osp.join(args.snapshot_dir, 'best.pth'))

                # 同时存一下 last
                if args.local_rank == 0:
                    checkpoint = {
                        'model': model,
                        'optimizer': optimizer.state_dict(),
                        'iter': i_iter
                    }
                    torch.save(checkpoint, osp.join(args.snapshot_dir, 'last.pth'))

        # 训练结束，保存 final
        if args.local_rank == 0:
            print("[INFO] Training finished, saving final model...")
            checkpoint = {
                'model': model,
                'optimizer': optimizer.state_dict(),
                'iter': args.num_steps
            }
            torch.save(checkpoint, osp.join(args.snapshot_dir, 'final.pth'))

    end = timeit.default_timer()
    print(f"Total training time: {end - start:.1f} seconds.")


if __name__ == '__main__':
    main()