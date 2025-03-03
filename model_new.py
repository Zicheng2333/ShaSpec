import torch
import torch.nn as nn
import torch.nn.functional as F
import random

########################################
# 示例：引用你已有的 Encoder
# 假设你已经有以下类可以直接使用:
#   TextEncoder   -> 提取 discharge summaries 的文本特征
#   CodeEncoder   -> 提取诊断/手术代码特征
#   RNNEncoder    -> 提取 lab events 序列特征
#
# 在实际项目中，请在此导入或直接定义它们
########################################
from MUSE.src.encoder.code_encoder import CodeEncoder
from MUSE.src.encoder.rnn_encoder import RNNEncoder
from MUSE.src.encoder.text_encoder import TextEncoder

########################################
# 保留的 CompositionalLayer (可选)
# 如果你仍然需要做共享 + 特定特征的组合
########################################
class CompositionalLayer(nn.Module):
    """
    用于将共享特征和特定特征拼接后，再做一次卷积或线性映射。
    这里改成简单的线性或 MLP 也可。
    """

    def __init__(self, in_dim=256, out_dim=256):
        super().__init__()
        self.fc = nn.Linear(in_dim * 2, out_dim, bias=False)
        # 也可改成 conv, BN, etc.

    def forward(self, shared_ft, spec_ft):
        # 简单拼接 -> FC -> residual
        cat = torch.cat([shared_ft, spec_ft], dim=-1)
        out = self.fc(cat)
        return shared_ft + out


########################################
# MIMIC encoder
########################################
class MIMICEncoder(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=256):
        """
        构造一个用于 MIMIC-IV 数据的 encoder 模块，
        模拟 U_Res3D_enc 的架构思路：
          - fc_reduce: 将输入特征降维至 hidden_dim，相当于 asppreduce 部分
          - fc_aspp: 进一步提取特征，相当于 ASPP 模块（这里用简单的全连接层模拟）
          - fc_out: 输出最终特征，与全局特征进行结合

        参数:
            input_dim: 输入特征维度（例如各模态 encoder 输出的特征维度）
            hidden_dim: 隐藏层维度，默认 256
            output_dim: 最终输出特征维度，默认 256
        """
        super(MIMICEncoder, self).__init__()

        # 类似于 asppreduce，将高维输入先映射到 hidden_dim
        self.fc_reduce = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        # 类似于 ASPP 模块，这里简单用一个全连接层来进行非线性映射
        self.fc_aspp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        # 最终映射到输出特征空间
        self.fc_out = nn.Linear(hidden_dim, output_dim, bias=False)

        # 初始化权重（类似于 U_Res3D_enc 中的初始化方式）
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        输入:
          x: [B, input_dim] 形状的张量

        输出:
          feature_out: 最终输出特征（用于下游任务或分类头），形状为 [B, output_dim]
          global_feature: 作为全局特征的中间输出（来自 fc_reduce），形状为 [B, hidden_dim]
        """
        global_feature = self.fc_reduce(x)  # 降维后的全局特征
        feature_aspp = self.fc_aspp(global_feature)  # 进一步提取的特征
        feature_out = self.fc_out(feature_aspp)  # 映射到输出空间

        return feature_out, global_feature

########################################
# Domain-Specific / Domain Classification
# 参考原先 dom_classifier 用于区分模态 0/1/2
########################################
class DualNet_SS(nn.Module):
    """
    新的三模态网络结构:
      - text_enc:    处理 discharge summaries
      - code_enc:    处理诊断/手术代码
      - lab_enc:     处理 lab events
      - dom_classifier: 预测当前特征属于哪个模态 (0=text,1=code,2=lab)
      - cls_classifier: 用于最终二分类 (mortality/readmission)
      - compos_layer: (可选) 用于将共享特征 + 特定特征 组合
    保留 Distribution Alignment 和 Domain Classification 的思路：
      - Distribution Alignment: 需要在训练脚本中，用 shared_feats 之间做 pairwise 距离
      - Domain Classification: 用 spec_feats 或最终特征 做 domain cls
    """

    def __init__(self, args,
                 tokenizer,
                 hidden_dim=256,
                 out_dim=1,
                 use_compos=True):
        super().__init__()
        self.device = "cuda"
        self.args = args
        self.hidden_dim = hidden_dim
        self.mimic_enc = MIMICEncoder()
        # 1) 三个模态的特征提取器
        self.text_enc = TextEncoder(
            bert_type=args.bert_type,
            device="cuda"
        )

        self.code_enc = CodeEncoder(
            tokenizer=tokenizer,
            embedding_size=args.embedding_size,
            pretrained_embedding=args.code_pretrained_embedding,
            dropout=args.dropout,
            layers=args.code_layers,
            heads=args.code_heads,
            device="cuda"
        )

        self.lab_enc = RNNEncoder(
            input_size=116,
            hidden_size=args.embedding_size,
            num_layers=args.rnn_layers,
            rnn_type = args.rnn_type,
            dropout=args.dropout,
            bidirectional=args.rnn_bidirectional,
            device="cuda")
        # 如果你不在GPU上运行，去掉 device="cuda" 或改成当前设备

        # 2) 是否使用 compos_layer，把共享 + 特定特征结合 (可选)
        self.use_compos = use_compos
        if self.use_compos:
            self.compos_layer = CompositionalLayer(in_dim=hidden_dim, out_dim=hidden_dim)

        # 3) 用于域分类(3个模态)
        # 注意：在训练时，需要你构造对应的标签 0=text,1=code,2=lab
        self.dom_classifier = nn.Linear(hidden_dim, 3, bias=True)

        # 4) 最终二分类输出
        # 如果 out_dim=1，表示后续用 BCEWithLogitsLoss；若 out_dim=2，则用 CrossEntropy
        self.cls_classifier = nn.Linear(hidden_dim, out_dim, bias=True)

    def forward(self,
                age,
                gender,
                ethnicity,
                types,
                codes,
                codes_flag,
                labvectors,
                labvectors_flag,
                discharge,
                discharge_flag,
                **kwargs,):
        """
        参数命名示例:
          - codes:     batch["codes"] -> [B, 400] (仅举例)
          - labvectors: batch["labvectors"] -> [B, T, 116]
          - discharge:  batch["discharge"] -> [B, 300] (embedding后)

        返回:
          - logits:      最终二分类预测 (shape=[B,1] or [B,2])
          - shared_feats:   用于 distribution alignment (list of shared feats)
          - spec_feats:     用于 domain classification (list of specific feats)
        """
        # 1) 提取模态特征

        codes_flag = codes_flag.to(self.device)
        labvectors_flag = labvectors_flag.to(self.device)
        discharge_flag = discharge_flag.to(self.device)

        text_ft = self.text_enc(discharge)
        code_ft = self.code_enc(codes, types, age, gender, ethnicity)
        lab_ft = self.lab_enc(labvectors)

        text_ft[discharge_flag==0]=0
        code_ft[codes_flag==0]=0
        lab_ft[labvectors_flag==0]=0

        # 2) 定义 "shared" 与 "specific" 特征
        shared_text = self.mimic_enc(text_ft)
        shared_code = self.mimic_enc(code_ft)
        shared_lab = self.mimic_enc(lab_ft)

        # 3) compos_layer 用于将 shared + specific 组合
        #    若不需要复合，可以直接 shared_xx=text_ft, spec_xx=text_ft
        if self.use_compos:
            text_fused = self.compos_layer(shared_text, text_ft)
            code_fused = self.compos_layer(shared_code, code_ft)
            lab_fused = self.compos_layer(shared_lab, lab_ft)
        else:
            text_fused = text_ft
            code_fused = code_ft
            lab_fused = lab_ft

        # 4) 做域分类 (domain classification)
        #    这里需要把 3 个模态的特征拼起来，再输入 self.dom_classifier
        #    并在训练时给出对应标签 0,1,2
        spec_feats = [text_fused, code_fused, lab_fused]  # list[Tensor], shape=[B, hidden_dim]

        # 5) 最终做二分类输出
        #    你需要决定是拼接三模态再分类，还是只用某个模态？
        #    这里示例：直接 "加和 / 平均" 三个模态特征后分类
        fused_all = text_fused + code_fused + lab_fused  # [B, hidden_dim]
        fused_all = fused_all / 3.0  # 均值
        logits = self.cls_classifier(fused_all)  # [B, out_dim]

        # 6) 返回的 shared_feats
        #    用于做 distribution alignment, 你可以返回 [shared_text, shared_code, shared_lab]
        shared_feats = [shared_text, shared_code, shared_lab]

        return logits, shared_feats, spec_feats