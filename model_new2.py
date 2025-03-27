import torch
import torch.nn as nn
import torch.nn.functional as F

# 示例保留
from MUSE.src.encoder.code_encoder import CodeEncoder
from MUSE.src.encoder.rnn_encoder import RNNEncoder
from MUSE.src.encoder.text_encoder import TextEncoder

class CompositionalLayer(nn.Module):
    """
    用于将共享特征和特定特征拼接后，再经过1×1卷积、BN和激活函数映射，
    最后通过残差连接融合到共享特征上。
    
    输入:
        shared_ft: [B, in_dim]
        spec_ft:   [B, in_dim]
    输出:
        融合后的特征: [B, out_dim]
    默认 in_dim = out_dim = 256
    """
    def __init__(self, in_dim=256, out_dim=256):
        super().__init__()
        # 用1×1卷积来替换原来的线性层，输入通道为 in_dim*2，输出通道为 out_dim
        self.conv = nn.Conv2d(in_dim * 2, out_dim, kernel_size=1, bias=False)
        self.bn = nn.GroupNorm(num_groups=32, num_channels=out_dim)
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, shared_ft, spec_ft):
        # shared_ft, spec_ft 的形状为 [B, in_dim]
        # 拼接得到 [B, 2*in_dim]
        cat = torch.cat([shared_ft, spec_ft], dim=-1)
        # 将其扩展为 [B, 2*in_dim, 1, 1] 以适用于卷积操作
        cat = cat.unsqueeze(-1).unsqueeze(-1)
        out = self.conv(cat)      # [B, out_dim, 1, 1]
        out = self.bn(out)
        out = self.activation(out)
        out = out.squeeze(-1).squeeze(-1)  # [B, out_dim]
        return shared_ft + out

class MIMICEncoder(nn.Module):
    """
    用于将模态特征投影到共享空间 (隐藏维 256)
    """
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=128):
        super().__init__()
        self.fc_reduce = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.fc_aspp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        # x: [B, input_dim]
        g = self.fc_reduce(x)
        a = self.fc_aspp(g)
        out = self.fc_out(a)  # [B, output_dim]
        return out

class DualNet_SS(nn.Module):
    """
    目标:
      1) 三模态 => text, code, lab
      2) 有 shared & specific 特征: shared_x = MIMICEnc(specific_x)
      3) 缺失模态 => 用其他模态的共享特征 average 替换, 并直接跳过 compos_layer
      4) 存在模态 => compos_layer(shared_x, specific_x) => fused_x
      5) 最终 [fused_text, fused_code, fused_lab] 加和/平均 => logits
    """
    def __init__(self, args, tokenizer,
                 input_dim=128,  # 假定 text/code/lab 均输出 256 维
                 hidden_dim=128,
                 out_dim=1):
        super().__init__()
        self.device = "cuda"
        self.args = args
        self.hidden_dim = hidden_dim
        
        # 1) 三个基础编码器 => specific feats
        self.text_enc = TextEncoder(
            bert_type=args.bert_type,
            device=self.device
        )
        self.code_enc = CodeEncoder(
            tokenizer=tokenizer,
            embedding_size=args.embedding_size,
            pretrained_embedding=args.code_pretrained_embedding,
            dropout=args.dropout,
            layers=args.code_layers,
            heads=args.code_heads,
            device=self.device
        )
        self.lab_enc = RNNEncoder(
            input_size=116,
            hidden_size=args.embedding_size,
            num_layers=args.rnn_layers,
            rnn_type=args.rnn_type,
            dropout=args.dropout,
            bidirectional=args.rnn_bidirectional,
            device=self.device
        )

        # 2) 公共 encoder => 生成 shared feats
        #    这里假设 specific feats 也都是 256 维 => input_dim=256
        self.mimic_enc = MIMICEncoder(input_dim, hidden_dim, hidden_dim)

        # 3) compos_layer => compos shared & specific
        self.compos_layer = CompositionalLayer(in_dim=hidden_dim, out_dim=hidden_dim)

        # 4) classifier => 二分类
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
                **kwargs):
        
        device = self.device
        #B = discharge.size(0)
        B = len(discharge)

        # ============ 1) 获得各模态特定特征 (specific feats) ============
        text_ft = self.text_enc(discharge)      # [B, embed_dim_text], 例如 768
        code_ft = self.code_enc(codes, types, age, gender, ethnicity)  # [B, input_dim]
        lab_ft  = self.lab_enc(labvectors)      # [B, input_dim]

        # 你需要确保 code_ft, lab_ft 输出形状是 [B, 256]；text_ft 也需要投影到 256, 见下:
        # 如果 text_ft.shape[-1] != 256, 需要多加一层线性投影 => e.g. text_ft = self.text_proj(text_ft)

        # Mask => 缺失模态 => spec = 0
        codes_flag = codes_flag.to(device)
        labvectors_flag = labvectors_flag.to(device)
        discharge_flag = discharge_flag.to(device)

        text_ft  = text_ft  * discharge_flag.unsqueeze(-1).float()
        code_ft  = code_ft  * codes_flag.unsqueeze(-1).float()
        lab_ft   = lab_ft   * labvectors_flag.unsqueeze(-1).float()

        # ============ 2) 获得各模态共享特征 (shared) ============
        shared_text = self.mimic_enc(text_ft)  # [B, hidden_dim]
        shared_code = self.mimic_enc(code_ft)
        shared_lab  = self.mimic_enc(lab_ft)

        # ============ 3) 对于缺失模态 => 用其他共享特征平均替换 ============
        # 按 batch 逐条处理
        shared_text_fixed = shared_text.clone()
        shared_code_fixed = shared_code.clone()
        shared_lab_fixed  = shared_lab.clone()

        # 记录: discharge_flag[i], codes_flag[i], labvectors_flag[i] => 0/1
        # 如果 =0 => 该模态缺失 => shared_x_fixed[i] = others' average
        for i in range(B):
            exist_shared = []
            if discharge_flag[i] == 1:
                exist_shared.append(shared_text[i])
            if codes_flag[i] == 1:
                exist_shared.append(shared_code[i])
            if labvectors_flag[i] == 1:
                exist_shared.append(shared_lab[i])
            
            if len(exist_shared) == 0:
                # 三模态都缺 => 留 0
                continue
            else:
                mean_ft = torch.stack(exist_shared, dim=0).mean(dim=0)
                if discharge_flag[i] == 0:
                    shared_text_fixed[i] = mean_ft
                if codes_flag[i] == 0:
                    shared_code_fixed[i] = mean_ft
                if labvectors_flag[i] == 0:
                    shared_lab_fixed[i] = mean_ft

        # ============ 4) compos_layer (仅对存在的模态进行) ============
        # 根据flag决定: 如果 flag=1 => compos => fused_x; 如果 flag=0 => fused_x=shared_x_fixed
        fused_text = []
        fused_code = []
        fused_lab  = []
        for i in range(B):
            # text
            if discharge_flag[i] == 1:
                # 进行 compos
                comp_x = self.compos_layer(shared_text_fixed[i].unsqueeze(0), 
                                           text_ft[i].unsqueeze(0))
                fused_text.append(comp_x.squeeze(0))
            else:
                # 不经过 compos_layer, 直接使用替换后的 shared
                fused_text.append(shared_text_fixed[i])
            
            # code
            if codes_flag[i] == 1:
                comp_x = self.compos_layer(shared_code_fixed[i].unsqueeze(0),
                                           code_ft[i].unsqueeze(0))
                fused_code.append(comp_x.squeeze(0))
            else:
                fused_code.append(shared_code_fixed[i])

            # lab
            if labvectors_flag[i] == 1:
                comp_x = self.compos_layer(shared_lab_fixed[i].unsqueeze(0),
                                           lab_ft[i].unsqueeze(0))
                fused_lab.append(comp_x.squeeze(0))
            else:
                fused_lab.append(shared_lab_fixed[i])
        
        # 重新拼成 batch 张量: [B, hidden_dim]
        fused_text = torch.stack(fused_text, dim=0)
        fused_code = torch.stack(fused_code, dim=0)
        fused_lab  = torch.stack(fused_lab, dim=0)

        # ============ 5) 融合三模态 & 分类 ============
        fused_all = (fused_text + fused_code + fused_lab) / 3.0
        logits = self.cls_classifier(fused_all)  # [B, out_dim]

        # ============ 6) 如果还要 domain cls / distribution alignment ============
        # domain cls => 3B => label 0,1,2
        # distribution alignment => [shared_text_fixed, shared_code_fixed, shared_lab_fixed]
        shared_feats = [shared_text_fixed, shared_code_fixed, shared_lab_fixed]
        spec_feats   = [fused_text, fused_code, fused_lab]  # 这里把 fused 也当成 'spec' or final feats

        return logits, shared_feats, spec_feats