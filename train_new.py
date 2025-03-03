import argparse
import sys
import os
import os.path as osp
import timeit
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

# 如果 utils 在上一级目录，需要如下添加:
sys.path.append("..")

#################################
# 1) 导入你自定义的数据集和工具函数
#################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MUSE/src')))

from xzc.MUSE.src.dataset.mimic4_dataset import MIMIC4Dataset
from xzc.MUSE.src.dataset.utils import mimic4_collate_fn

# 导入你的损失函数和模型
import loss_Dual as loss
from model_new import DualNet_SS


###############################################################################
# 1. 超参解析
###############################################################################
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--start_iters", type=int, default=0)
    parser.add_argument("--val_pred_every", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--random_seed", type=int, default=999)

    # 模式设置（是否仅训练、不评估等）
    parser.add_argument("--train_only", action="store_true")

    # 三模态相关参数（来自 MUSE）
    parser.add_argument("--embedding_size", type=int, default=256)
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


###############################################################################
# 2. 评估函数
###############################################################################
def evaluate_model(model, dataloader, device):
    """
    计算二分类的准确率、F1、PR AUC 和 ROC AUC。
    """
    model.eval()
    correct, total = 0, 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing Batches"):
            labels = batch["label"].to(device)
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


###############################################################################
# 3. 其他工具函数
###############################################################################
def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, lr, num_steps, power):
    """多项式学习率衰减"""
    new_lr = lr_poly(lr, i_iter, num_steps, power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr

def replace_zero_vectors(shared_info):
    """
    检查 shared_info 是否包含全 0 向量，并用其他模态的均值替换
    shared_info 是一个 list: [text_shared, code_shared, lab_shared]
    """
    shared_info = torch.stack(shared_info)  # [3, B, hidden_dim]
    zero_mask = (shared_info == 0).all(dim=(1, 2))  # [3]，检查每个模态是否全 0

    # 计算非零模态的均值
    non_zero_vectors = shared_info[~zero_mask]  # 选出非零模态
    if len(non_zero_vectors) > 0:
        mean_vector = non_zero_vectors.mean(dim=0)  # [B, hidden_dim]
    else:
        # 如果所有模态都是 0，则保留全 0
        mean_vector = torch.zeros_like(shared_info[0])

    # 替换全 0 模态的向量
    for i in range(3):
        if zero_mask[i]:  
            shared_info[i] = mean_vector

    return [shared_info[0], shared_info[1], shared_info[2]]  # 返回修正后的 shared_info


###############################################################################
# 4. 主函数：单 GPU 训练
###############################################################################
def main():
    parser = get_arguments()
    args = parser.parse_args()

    # 你可以选择性打印一下参数
    print("Training Arguments:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    # 设备：单 GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 设置随机种子
    cudnn.benchmark = True
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)

    # 创建日志目录 & TensorBoard 写入器
    writer = SummaryWriter(args.snapshot_dir)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    #########################################################
    # A. 数据集 & DataLoader
    #########################################################
    train_dataset = MIMIC4Dataset(
        split="train",
        task=args.task,
        dev=args.dev,
        load_no_label=args.load_no_label
    )
    val_dataset = MIMIC4Dataset(
        split="val",
        task=args.task
    )
    tokenizer = train_dataset.tokenizer

    # 训练集 DataLoader (单 GPU 不需要分布式 sampler)
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
        collate_fn=mimic4_collate_fn,
    )

    # 验证集 DataLoader
    valloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    #########################################################
    # B. 模型 & 优化器
    #########################################################
    model = DualNet_SS(args, tokenizer)
    model.to(device)

    # 如果需要从 checkpoint 恢复
    if args.reload_from_checkpoint:
        if os.path.exists(args.reload_path):
            checkpoint = torch.load(args.reload_path, map_location='cpu')
            # 这里取决于你在保存时如何做
            # 如果是 checkpoint['model'] = model，可能要直接 load_state_dict
            # 如果是直接存的整个 model，对象反序列化即可
            model = checkpoint['model']
            args.start_iters = checkpoint['iter']
            print(f"[INFO] Loaded model from {args.reload_path}, trained for {args.start_iters} iters.")
        else:
            print(f"[ERROR] File not found: {args.reload_path}")
            exit(0)

    # 定义损失函数
    criterion_cls = nn.BCEWithLogitsLoss()
    loss_domain_cls = loss.DomainClsLoss().to(device)
    distribution_loss = nn.L1Loss()

    # 优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    #########################################################
    # C. 训练循环
    #########################################################
    print("[INFO] Start Training ...")
    model.train()

    alpha = 0.1  # shared domain loss weight
    beta = 0.02  # specific domain loss weight
    best_acc = -1

    start_time = timeit.default_timer()

    for i_iter, batch in enumerate(tqdm(trainloader, desc="Training Progress")):
        i_iter += args.start_iters
        if i_iter >= args.num_steps:
            break

        # 学习率衰减
        lr = adjust_learning_rate(optimizer, i_iter, args.learning_rate, args.num_steps, args.power)

        labels = batch["label"].to(device)
        optimizer.zero_grad()

        # 前向传播
        logits, shared_info, spec_info = model(
            **batch
        )

        # 计算任务损失
        if logits.size(-1) == 1:
            task_loss = criterion_cls(logits.view(-1), labels.float())
        else:
            task_loss = nn.CrossEntropyLoss()(logits, labels.long())

        # Distribution Alignment (三模态)
        shared_info = replace_zero_vectors(shared_info)
        term_shared = (distribution_loss(shared_info[0], shared_info[1]) +
                       distribution_loss(shared_info[1], shared_info[2]) +
                       distribution_loss(shared_info[2], shared_info[0]))

        # Domain Classification (三模态)
        if isinstance(spec_info, list) and len(spec_info) == 3:
            B = spec_info[0].size(0)
            cat_spec = torch.cat(spec_info, dim=0)  # [3B, feat_dim]
            spec_labels = torch.cat([
                torch.zeros(B, dtype=torch.long),
                torch.ones(B, dtype=torch.long),
                2 * torch.ones(B, dtype=torch.long)
            ], dim=0).to(device)
            term_spec = loss_domain_cls(cat_spec, spec_labels)
        else:
            term_spec = 0.0

        # 总损失
        loss_all = task_loss + alpha * term_shared + beta * term_spec
        loss_all.backward()
        optimizer.step()

        # 打印日志
        if i_iter % 100 == 0:
            writer.add_scalar('learning_rate', lr, i_iter)
            writer.add_scalar('loss_all', loss_all.item(), i_iter)
            writer.add_scalar('task_loss', task_loss.item(), i_iter)
            writer.add_scalar('term_shared', term_shared.item(), i_iter)
            if not isinstance(term_spec, float):
                writer.add_scalar('term_spec', term_spec.item(), i_iter)

            print(f"Iter={i_iter}/{args.num_steps}, lr={lr:.6f}, task_loss={task_loss.item():.4f}, "
                  f"shared_loss={term_shared.item():.4f}, spec_loss={term_spec}, total={loss_all.item():.4f}")

        # 验证 & 保存
        if (not args.train_only) and (i_iter % args.val_pred_every == args.val_pred_every - 1):
            print("[INFO] Validating ...")
            val_acc, f1, roc_auc, pr_auc = evaluate_model(model, valloader, device)
            writer.add_scalar('Val_Accuracy', val_acc, i_iter)
            print(f"Validation @ iter {i_iter}: accuracy = {val_acc:.2f}, f1 = {f1}, roc_auc = {roc_auc}, pr_auc = {pr_auc}")

            # 保存 best
            if val_acc > best_acc:
                best_acc = val_acc
                print("[INFO] Saving best model ...")
                checkpoint = {
                    'model': model,
                    'optimizer': optimizer.state_dict(),
                    'iter': i_iter
                }
                torch.save(checkpoint, osp.join(args.snapshot_dir, 'best.pth'))

            # 保存 last
            checkpoint = {
                'model': model,
                'optimizer': optimizer.state_dict(),
                'iter': i_iter
            }
            torch.save(checkpoint, osp.join(args.snapshot_dir, 'last.pth'))

    # 训练结束，保存 final
    print("[INFO] Training finished, saving final model...")
    checkpoint = {
        'model': model,
        'optimizer': optimizer.state_dict(),
        'iter': args.num_steps
    }
    torch.save(checkpoint, osp.join(args.snapshot_dir, 'final.pth'))

    end_time = timeit.default_timer()
    print(f"Total training time: {end_time - start_time:.1f} seconds.")


if __name__ == '__main__':
    main()