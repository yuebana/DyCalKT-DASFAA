import random
import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from .utils import transformer_FFN, ut_mask, pos_encode, get_clones
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, \
        MultiLabelMarginLoss, MultiLabelSoftMarginLoss, CrossEntropyLoss, BCELoss, MultiheadAttention
from torch.nn.functional import one_hot, cross_entropy, multilabel_margin_loss, binary_cross_entropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SEBlock(nn.Module):
    """
    通道重标定：输入 [B, T, H]，输出同形状
    reduction 默认为 16，可通过 --se_ratio 调整
    """
    def __init__(self, hidden_dim: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // reduction, hidden_dim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=1)                  # [B, H]  (Squeeze)
        s = self.fc2(self.relu(self.fc1(s))).unsqueeze(1)  # [B,1,H]
        s = self.sigmoid(s)                # Gate
        return x * s                       # [B,T,H]  (Excitation)
    

class CosinePositionalEmbedding(nn.Module):
    """
    简单示例：用 sin / cos 函数构造的位置编码，常见于Transformer。
    这里叫 “CosinePositionalEmbedding” 只是个命名示例，也包含了 sin 分量。
    你也可以根据需求改成纯 cos 或其他自定义函数。
    """
    def __init__(self, d_model, max_len=5000):
        super(CosinePositionalEmbedding, self).__init__()
        # 创建一个 [max_len, d_model] 的位置编码矩阵 pe
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # shape=[max_len,1]
        # 计算 dim=2 的 div_term: 频率递增，常规Transformer做法
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # 偶数维用 sin，奇数维用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # shape: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        # 注册为buffer，意味着在模型保存/加载时自动处理，但不会成为可训练参数
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model] 或 [batch_size, seq_len]
        这里仅用 seq_len 来截取相应长度的编码并返回。
        """
        seq_len = x.size(1)
        # 注意 self.pe[:, :seq_len, :] 只能取到序列长度之内的位置编码
        # shape=[1, seq_len, d_model]
        return self.pe[:, :seq_len, :]


class Architecture(nn.Module):
    def __init__(self, n_question, n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, seq_len):
        super().__init__()
        self.d_model = d_model
        self.model_type = model_type

        self.position_embeddings = CosinePositionalEmbedding(
            d_model=self.d_model,
            max_len=seq_len
        )

        if model_type in {'DyCalKT'}:
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(
                    d_model=d_model,
                    d_feature=d_model // n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    n_heads=n_heads,
                    kq_same=kq_same
                )
                for _ in range(n_blocks)
            ])

    def forward(self, q_embed_data, qa_embed_data, atten_type='dp'):
        q_embed_data = q_embed_data + self.position_embeddings(q_embed_data)
        qa_embed_data = qa_embed_data + self.position_embeddings(qa_embed_data)

        x = q_embed_data
        y = qa_embed_data

        for block in self.blocks_2:
            x = block(
                mask=0,
                query=x,
                key=x,
                values=y,
                atten_type=atten_type,
                apply_pos=True
            )

        return x


        
class DyCalKT(nn.Module):
    def __init__(self, n_question, n_pid, emb_type,   
                 n_blocks, dropout, d_model, d_ff,
                 final_fc_dim, final_fc_dim2, seq_len, model_type='DyCalKT',
                 noise_level=0.2,
                 num_attn_heads=8, 
            loss1=0.5, loss2=0.5, loss3=0.5, start=50, num_layers=2, nheads=4,
            kq_same=1, atten_type='dp', emb_path="", pretrain_dim=768,separate_qa=False, n_heads=8, se_ratio: int = 16,
            # ===== 新增：动态模态相关参数 =====
             num_skill=0,
             num_diff=0,
             num_time=0,
             num_cluster=0,
             num_behavior=0,
             proj_dim=128,
             anchor_weight=0.02,
            amp_weight=0.3,
            dynamic_weight=0.08,
            threshold=0.15,
            text_input_dim=384,
             use_dynamic_modal=True,
             use_anchor_alignment=True,
                # ===== 消融实验开关 =====
            ablation_mode="full",
            use_cal_loss=True,
            use_con_loss=True):
        super(DyCalKT, self).__init__()
        self.model_name = "DyCalKT"
        self.n_question = n_question
        self.n_pid = n_pid
        self.emb_type = emb_type
        self.separate_qa = separate_qa
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.d_model = d_model
        self.model_type = model_type
        self.threshold = threshold
        self.dynamic_weight = dynamic_weight
        self.kq_same = 1  # query and key use same projection

        self.ablation_mode = ablation_mode
        self.use_cal_loss = use_cal_loss
        self.use_con_loss = use_con_loss
        
        if self.n_pid > 0 and "norasch" not in emb_type:
            self.q_embed_diff = nn.Embedding(n_question+1, d_model)
            self.qa_embed_diff = nn.Embedding(2, d_model)
            self.difficult_param = nn.Embedding(n_pid+1, d_model)
        self.atten_type = atten_type
        
        # ===== 新增：动态模态配置 =====
        self.use_dynamic_modal = use_dynamic_modal
        self.use_anchor_align = use_anchor_alignment
        self.anchor_weight = anchor_weight
        self.proj_dim = proj_dim
        self.amp_weight = amp_weight
        self.text_input_dim = text_input_dim

        # 调试打印计数器：只打印前几个 batch
        self.debug_print_count = 0

        # 各模态词表大小（由配置文件传入；为 0 表示该模态不用 embedding）
        self.num_skill = num_skill
        self.num_diff = num_diff
        self.num_time = num_time
        self.num_cluster = num_cluster
        self.num_behavior = num_behavior

        valid_modes = [
            "backbone",
            "static_evidence",
            "selection_only",
            "calibration_only",
            "full"
        ]
        self.evidence_names = [
            "skill",
            "difficulty",
            "time",
            "behavior",
            "cluster"
        ]

        assert self.ablation_mode in valid_modes, (f"Unknown ablation mode: {self.ablation_mode}")


        embed_l = d_model

        if emb_type.startswith("qid") or emb_type.startswith("stoc"):
            self.q_embed = nn.Embedding(self.n_question + 1, embed_l)

            if self.separate_qa:
                self.qa_embed = nn.Embedding(2 * self.n_question + 1, embed_l)
            else:
                self.qa_embed = nn.Embedding(2, embed_l)
            # question 模态使用真实题目 ID，因此单独定义题目 embedding
            # n_pid > 0 时，pid_data 对应真实题目编号
            if self.n_pid > 0:
                self.question_embed = nn.Embedding(self.n_pid + 1, embed_l)
            else:
                self.question_embed = self.q_embed

        # Architecture 对象，包含多层注意力块
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=n_heads, dropout=dropout,
                                  d_model=d_model, d_feature=d_model / n_heads, d_ff=d_ff,
                                  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len)
        
        self.se_gate = SEBlock(d_model, reduction=se_ratio)

        # ===== 新增：动态模态编码器 =====
        # question 模态直接复用原有 q 表示，因此这里不单独新建 question_emb
        if self.num_skill > 0:
            self.skill_embed = nn.Embedding(self.num_skill + 1, d_model)
        else:
            self.skill_embed = None

        if self.num_diff > 0:
            self.diff_embed = nn.Embedding(self.num_diff + 1, d_model)
        else:
            self.diff_embed = None

        if self.num_time > 0:
            self.time_embed = nn.Embedding(self.num_time + 1, d_model)
        else:
            self.time_embed = None

        if self.num_cluster > 0:
            self.cluster_embed = nn.Embedding(self.num_cluster + 1, d_model)
        else:
            self.cluster_embed = None

        # behavior 往往不是离散 ID，因此这里用一个线性层把连续行为特征映射到 d_model
        if self.num_behavior > 0:
            self.behavior_embed = nn.Embedding(self.num_behavior + 1, d_model)
        else:
            self.behavior_embed = None

        # ===== 新增：Anchor 投影头 =====
        self.student_proj = nn.Linear(d_model, proj_dim)
        self.modality_proj = nn.ModuleDict({
            "question": nn.Linear(d_model, proj_dim),
            "skill": nn.Linear(d_model, proj_dim),
            "difficulty": nn.Linear(d_model, proj_dim),
            "time": nn.Linear(d_model, proj_dim),
            "cluster": nn.Linear(d_model, proj_dim),
            "behavior": nn.Linear(d_model, proj_dim),
        })

        # ===== 新增：状态引导的模态门控层 =====
        # # 输入为 [modal_emb ; student_state]，输出该模态的打分
        # self.modality_gate = nn.ModuleDict({
        #     "question": nn.Linear(d_model * 2, 1),
        #     "skill": nn.Linear(d_model * 2, 1),
        #     "difficulty": nn.Linear(d_model * 2, 1),
        #     "time": nn.Linear(d_model * 2, 1),
        #     "cluster": nn.Linear(d_model * 2, 1),
        #     "behavior": nn.Linear(d_model * 2, 1),
        # })

        # ===== 状态-证据关系评分层 =====
        # 输入 [h, z, |h-z|, h*z]，维度为 4D
        self.modality_gate = nn.ModuleDict({
            "question": nn.Linear(d_model * 4, 1),
            "skill": nn.Linear(d_model * 4, 1),
            "difficulty": nn.Linear(d_model * 4, 1),
            "time": nn.Linear(d_model * 4, 1),
            "cluster": nn.Linear(d_model * 4, 1),
            "behavior": nn.Linear(d_model * 4, 1),
        })

        # ===== Anchor Calibration =====
        self.anchor_gate = nn.Linear(
            self.d_model * 2,
            self.d_model
        )

        self.anchor_proj = nn.Linear(
            self.d_model,
            self.d_model
        )
        

        # 输出层：将四部分拼接后经过全连接得到最终预测
        self.out = nn.Sequential(
            nn.Linear(embed_l * 4, final_fc_dim),
            nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2),
            nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1)
        )
        self.reset()


    def reset(self):
        if hasattr(self, "difficult_param"):
            torch.nn.init.constant_(self.difficult_param.weight, 0.)

    # =========================================================
    # 新增辅助函数 1：安全读取模态,兼容有些数据集没有该模态
    # =========================================================
    def _get_optional_seq(self, dcur, key, shift_key=None):
        """
        读取可选模态。
        若该模态不存在，或整条序列全为 pad_val(-1)，则返回 None。
        """
        if key not in dcur:
            # if self.debug_print_count < 3:
            #     print(f"[DEBUG] {key} not in dcur")
            return None

        cur = dcur[key]
        if shift_key is not None and shift_key in dcur:
            shft = dcur[shift_key]
            seq = torch.cat((cur[:, 0:1], shft), dim=1)
        else:
            seq = cur

        # 如果整批都是 pad(-1)，说明该模态在当前数据集中实际缺失
        if torch.all(seq == -1):
            return None
        return seq


    # =========================================================
    # 新增辅助函数 2：把原始模态值编码skill/difficulty/time/cluster成统一维度
    # =========================================================
    def _encode_modalities(self, dcur, question_data):
        """
        返回一个 modality_dict:
        {
            "question": [B, T, D],
            "skill": [B, T, D] or None,
            "difficulty": [B, T, D] or None,
            "time":[B, T, D] or None,
            "cluster": [B, T, D] or None,
            "mult-cluster": [B, T, D] or None,
            "behavior": [B, T, D] or None,
            ...
        }
        """
        modality_dict = {}

        # ---------- question ----------
        # 如果当前数据集没有有效的 question 序列，就把 question 模态置空
        # if question_data is None or question_data.size(1) == 0:
        #     modality_dict["question"] = None
        # else:
        #     question_data_safe = question_data.clone()

        #     if self.n_pid > 0:
        #         question_data_safe = torch.clamp(question_data_safe, min=0, max=self.n_pid)
        #         modality_dict["question"] = self.question_embed(question_data_safe.long())
        #     else:
        #         question_data_safe = torch.clamp(question_data_safe, min=0, max=self.n_question)
        #         modality_dict["question"] = self.q_embed(question_data_safe.long())


        # ---------- skill ----------
        # 先尝试读取显式 skill 字段；如果没有，就退回使用 concept 序列 cseqs
        skill_data = self._get_optional_seq(dcur, "skillseqs", "shft_skillseqs")
        if skill_data is None:
            skill_data = self._get_optional_seq(dcur, "cseqs", "shft_cseqs")

        if skill_data is not None and self.skill_embed is not None:
            # 防止越界
            skill_data_safe = torch.clamp(skill_data.clone(), min=0, max=self.num_skill)
            modality_dict["skill"] = self.skill_embed(skill_data_safe.long())
        else:
            modality_dict["skill"] = None

        # ---------- difficulty ----------
        diff_data = self._get_optional_seq(dcur, "diffseqs", "shft_diffseqs")
        if diff_data is not None and self.diff_embed is not None:
            diff_data_safe = torch.clamp(diff_data.clone(), min=0, max=self.num_diff)
            modality_dict["difficulty"] = self.diff_embed(diff_data_safe.long())
        else:
            modality_dict["difficulty"] = None

        # ---------- time ----------
        # 先尝试显式 time 字段；如果没有，就退回到 tseqs；
        time_data = self._get_optional_seq(dcur, "timeseqs", "shft_timeseqs")
        if time_data is None:
            time_data = self._get_optional_seq(dcur, "tseqs", "shft_tseqs")
        if time_data is None:
            time_data = self._get_optional_seq(dcur, "utseqs", "shft_utseqs")

        if time_data is not None and self.time_embed is not None:
            time_data_safe = torch.clamp(time_data.clone(), min=0, max=self.num_time)
            modality_dict["time"] = self.time_embed(time_data_safe.long())
        else:
            modality_dict["time"] = None
            
        # ---------- cluster ----------
        cluster_data = self._get_optional_seq(dcur, "clusterseqs", "shft_clusterseqs")
        if cluster_data is not None and self.cluster_embed is not None:

            # 防止 cluster 索引越界
            cluster_data_safe = torch.clamp(cluster_data.clone(), min=0, max=self.num_cluster)
            modality_dict["cluster"] = self.cluster_embed(cluster_data_safe.long())
        else:
            modality_dict["cluster"] = None

        # ---------- behavior ----------
        # behavior 默认按连续向量处理: [B, T, F]
        behavior_data = self._get_optional_seq(dcur, "behaviorseqs", "shft_behaviorseqs")
        if behavior_data is not None and self.behavior_embed is not None:
            behavior_data_safe = torch.clamp(behavior_data.clone(), min=0, max=self.num_behavior)
            modality_dict["behavior"] = self.behavior_embed(behavior_data_safe.long())
        else:
            modality_dict["behavior"] = None

        return modality_dict

    # =========================================================
    # 新增辅助函数 3：状态引导的动态模态融合，根据学生状态做动态门控融合
    # =========================================================
    def _fuse_modalities(self, modality_dict, student_state):
        """
        状态引导的动态模态融合。

        student_state: [B, T, D]
        modality_dict 中每个模态也是 [B, T, D]

        返回:
            fused_modal: [B, T, D]
            gate_scores: dict，保存每个模态的原始打分
            alpha: [B, T, M]，动态模态权重
            valid_names: list[str]，当前 batch 中实际参与融合的模态名称
        """
        valid_modalities = []
        valid_names = []
        gate_scores = {}

        # for name, emb in modality_dict.items():
        #修改让question不参与额外证据选择
        for name in self.evidence_names:
            emb = modality_dict.get(name, None)
            
            if emb is None:
                continue

            # gate_input = torch.cat([emb, student_state], dim=-1)   # [B, T, 2D]
            # score = self.modality_gate[name](gate_input)           # [B, T, 1]

            #修改维度为4⭐
            # 状态-证据关系表示：[h, z, |h-z|, h*z]
            relation = torch.cat([
                student_state,
                emb,
                torch.abs(student_state - emb),
                student_state * emb
                ], dim=-1)  # [B, T, 4D]
            # 计算证据效用分数
            score = self.modality_gate[name](relation)

            valid_modalities.append(emb)
            valid_names.append(name)
            gate_scores[name] = score

        if len(valid_modalities) == 0:
            return torch.zeros_like(student_state), gate_scores, None, valid_names

        score_tensor = torch.cat(
            [gate_scores[name] for name in valid_names],
            dim=-1
        )  # [B, T, M]

        alpha = torch.softmax(score_tensor, dim=-1)  # [B, T, M]
        # ===== High-confidence modality selection =====

        mask_alpha = (alpha > self.threshold).float()

        alpha = alpha * mask_alpha

        alpha = alpha / (alpha.sum(dim=-1, keepdim=True) + 1e-8)

        fused_modal = 0.0
        for idx, name in enumerate(valid_names):
            fused_modal = fused_modal + alpha[..., idx:idx + 1] * modality_dict[name]

        return fused_modal, gate_scores, alpha, valid_names

 #静态证据融合函数
    def _static_fuse_modalities(
        self,
        modality_dict,
        student_state
    ):
        """
        使用与 DyCalKT 完全相同的外部证据，
        但不使用动态打分、不使用 threshold。
        所有有效证据等权平均。
        """

        valid_modalities = []
        valid_names = []

        for name in self.evidence_names:
            emb = modality_dict.get(name, None)

            if emb is None:
                continue

            valid_modalities.append(emb)
            valid_names.append(name)

        if len(valid_modalities) == 0:
            return (
                torch.zeros_like(student_state),
                None,
                valid_names
            )

        # [B,T,M,D]
        modal_stack = torch.stack(
            valid_modalities,
            dim=2
        )

        # 所有证据等权平均
        fused_modal = modal_stack.mean(dim=2)

        B, T, M, _ = modal_stack.shape

        alpha = torch.full(
            (B, T, M),
            1.0 / M,
            device=student_state.device,
            dtype=student_state.dtype
        )

        return fused_modal, alpha, valid_names
    

    def _anchor_calibration(
        self,
        student_state,
        modality_anchor
        ):

        """
        student_state:   [B,T,D]
        modality_anchor:[B,T,D]
        """

        gate_input = torch.cat(
            [student_state, modality_anchor],
            dim=-1
        )

        gate = torch.sigmoid(
            self.anchor_gate(gate_input)
        )

        anchor_info = self.anchor_proj(
            modality_anchor
        )

        calibrated_state = (
            student_state
            + gate * anchor_info
        )

        return calibrated_state

    def _time_step_anchor_loss(
            self,
            student_state,
            modality_anchor,
            mask=None
        ):

        """
        student_state:   [B,T,D]
        modality_anchor:[B,T,D]
        """

        student_norm = F.normalize(
            student_state,
            dim=-1
        )

        anchor_norm = F.normalize(
            modality_anchor,
            dim=-1
        )

        sim = (
            student_norm * anchor_norm
        ).sum(dim=-1)

        loss = 1.0 - sim

        if mask is not None:
            true_tensor = torch.ones(
                mask.size(0), 1,
                dtype=torch.bool,
                device=mask.device
            )
            mask_full = torch.cat((true_tensor, mask), dim=1)  # [B, T]

            loss = loss * mask_full.float()

            return loss.sum() / (mask_full.float().sum() + 1e-8)

        return loss.mean()

    # =========================================================
    # 新增辅助函数 4：池化序列，得到序列级 Anchor 表示，把序列变成序列级表示，用于对齐
    # =========================================================
    def _masked_mean_pool(self, x, mask):
        """
        x: [B, T, D]
        mask: [B, T-1] (原实现里是 shift 后的 mask)
        这里与原代码保持一致，前面补一个 True。
        """
        true_tensor = torch.ones(mask.size(0), 1, dtype=torch.bool).to(mask.device)
        mask_full = torch.cat((true_tensor, mask), dim=1).unsqueeze(-1).float()  # [B, T, 1]
        x_sum = torch.sum(x * mask_full, dim=1)
        x_cnt = torch.clamp(mask_full.sum(dim=1), min=1.0)
        return x_sum / x_cnt

    # =========================================================
    # 新增辅助函数 5：Anchor-InfoNCE
    # =========================================================
    def _anchor_nce_loss(self, student_state, modality_dict, mask, temperature=0.2):
        """
        student_state: [B, T, D]，作为 Anchor
        对每个存在的模态做序列级 InfoNCE
        """
        if not self.use_anchor_align:
            return torch.tensor(0.0, device=student_state.device)

        anchor_seq = self._masked_mean_pool(student_state, mask)            # [B, D]
        anchor_proj = F.normalize(self.student_proj(anchor_seq), dim=-1)    # [B, P]

        total_loss = 0.0
        valid_cnt = 0

        for name, emb in modality_dict.items():
            if emb is None:
                continue
            modal_seq = self._masked_mean_pool(emb, mask)                   # [B, D]
            modal_proj = F.normalize(self.modality_proj[name](modal_seq), dim=-1)  # [B, P]

            logits = torch.matmul(anchor_proj, modal_proj.transpose(0, 1)) / temperature
            labels = torch.arange(logits.size(0), device=logits.device)
            total_loss = total_loss + F.cross_entropy(logits, labels)
            valid_cnt += 1

        if valid_cnt == 0:
            print("valid_cnt =", valid_cnt)
            return torch.tensor(0.0, device=student_state.device)

        return total_loss / valid_cnt

    # =========================================================
    # 新增辅助函数 6：动态权重感知 Anchor-InfoNCE
    # =========================================================
    def _dynamic_anchor_nce_loss(
        self,
        student_state,
        modality_dict,
        mask,
        modal_alpha,
        valid_names,
        temperature=0.2
        ):
        """
        Dynamic-aware Anchor-InfoNCE.

        student_state: [B, T, D]
        modality_dict: 各模态表示
        mask: [B, T-1]
        modal_alpha: [B, T, M]，来自动态模态融合的权重
        valid_names: 长度为 M 的有效模态名称列表

        核心思想：
        动态融合中权重越高的模态，在 Anchor 对齐中贡献越大。
        """
        if not self.use_anchor_align:
            return torch.tensor(0.0, device=student_state.device)

        if modal_alpha is None or valid_names is None or len(valid_names) == 0:
            return torch.tensor(0.0, device=student_state.device)

        # mask_full: [B, T, 1]
        true_tensor = torch.ones(
            mask.size(0),
            1,
            dtype=torch.bool,
            device=mask.device
        )
        mask_full = torch.cat((true_tensor, mask), dim=1).unsqueeze(-1).float()

        # 学生状态作为 Anchor
        anchor_seq = self._masked_mean_pool(student_state, mask)             # [B, D]
        anchor_proj = F.normalize(self.student_proj(anchor_seq), dim=-1)     # [B, P]

        total_loss = torch.tensor(0.0, device=student_state.device)
        total_weight = torch.tensor(0.0, device=student_state.device)

        for idx, name in enumerate(valid_names):
            emb = modality_dict.get(name, None)
            if emb is None:
                continue

            # 当前模态在每个样本上的平均动态权重
            # alpha_m: [B, T, 1]
            alpha_m = modal_alpha[..., idx:idx + 1]

            alpha_m_seq = torch.sum(alpha_m * mask_full, dim=1) / torch.clamp(
                mask_full.sum(dim=1),
                min=1.0
            )  # [B, 1]

            # 当前模态在 batch 上的平均重要性，作为该模态 Anchor loss 权重
            modal_weight = alpha_m_seq.mean()  # scalar

            modal_seq = self._masked_mean_pool(emb, mask)                   # [B, D]
            modal_proj = F.normalize(self.modality_proj[name](modal_seq), dim=-1)

            logits = torch.matmul(anchor_proj, modal_proj.transpose(0, 1)) / temperature
            labels = torch.arange(logits.size(0), device=logits.device)

            modal_loss = F.cross_entropy(logits, labels)

            total_loss = total_loss + modal_weight * modal_loss
            total_weight = total_weight + modal_weight.detach()

        if total_weight.item() <= 1e-8:
            return torch.tensor(0.0, device=student_state.device)

        return total_loss / torch.clamp(total_weight, min=1e-8)

    

    def base_emb(self, q_data, target):
        """
        普通序列 embedding。
        q_data:  [B, T]，概念/题目序列
        target:  [B, T]，作答结果序列
        return:
            q_embed_data:  [B, T, D]
            qa_embed_data: [B, T, D]
        """
        q_embed_data = self.q_embed(q_data.long())

        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_data = torch.clamp(qa_data, min=0, max=2 * self.n_question)
            qa_embed_data = self.qa_embed(qa_data.long())
        else:
            qa_embed_data = self.qa_embed(target.long()) + q_embed_data

        return q_embed_data, qa_embed_data


    def forward(self, dcur, qtest=False, train=False):
        # 1. Input Data
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()
        pid_data = torch.cat((q[:, 0:1], qshft), dim=1)
        q_data = torch.cat((c[:, 0:1], cshft), dim=1)
        target = torch.cat((r[:, 0:1], rshft), dim=1)
        mask = dcur["masks"]

        # 2. Base Embedding
        q_embed_data, qa_embed_data = self.base_emb(q_data, target)
        zero_qa = torch.zeros_like(qa_embed_data[:, 0:1, :])
        qa_embed_data = torch.cat([zero_qa, qa_embed_data[:, :-1, :]], dim=1)

        # 3. Dynamic Modality Encoding
        modality_dict = None
        if self.use_dynamic_modal:
            if pid_data.size(1) > 0:
                modality_dict = self._encode_modalities(dcur, pid_data)
            else:
                modality_dict = self._encode_modalities(dcur, q_data)

        # 4. Difficulty Injection
        if self.n_pid > 0 and self.emb_type.find("norasch") == -1:
            q_embed_diff_data = self.q_embed_diff(q_data)
            pid_embed_data = self.difficult_param(pid_data)
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data
            
            if self.emb_type.find("aktrasch") != -1:
                qa_embed_diff_data = self.qa_embed_diff(target)
                qa_embed_data = qa_embed_data + pid_embed_data * (qa_embed_diff_data + q_embed_diff_data)

        # 5. Primary State Estimation
        d_output = self.model(q_embed_data, qa_embed_data, self.atten_type)
        d_output = self.se_gate(d_output)


        # =========================================================
        # 6. Evidence Ablation / Mechanism Attribution
        # =========================================================

        modal_alpha = None
        valid_modal_names = []
        modality_anchor = None

        anchor_loss = torch.tensor(
            0.0,
            device=d_output.device
        )

        L_dynamic = torch.tensor(
            0.0,
            device=d_output.device
        )

        # 保存未使用 evidence 前的 preliminary state
        preliminary_state = d_output


        # ---------------------------------------------------------
        # A. Backbone only
        # ---------------------------------------------------------
        if self.ablation_mode == "backbone":
            # 完全不使用外部 evidence
            d_output = preliminary_state


        # ---------------------------------------------------------
        # 其余模式都使用完全相同的 external evidence
        # ---------------------------------------------------------
        elif (
            self.use_dynamic_modal
            and modality_dict is not None
        ):

            # =====================================================
            # B. Static evidence
            # C. Calibration only
            # -> 不使用动态 selection
            # =====================================================
            if self.ablation_mode in [
                "static_evidence",
                "calibration_only"
            ]:
                modality_anchor, modal_alpha, valid_modal_names = \
                    self._static_fuse_modalities(
                        modality_dict,
                        preliminary_state
                    )


            # =====================================================
            # D. Selection only
            # E. Full DyCalKT
            # -> 使用动态 selection
            # =====================================================
            elif self.ablation_mode in [
                "selection_only",
                "full"
            ]:
                modality_anchor, gate_scores, modal_alpha, valid_modal_names = \
                    self._fuse_modalities(
                        modality_dict,
                        preliminary_state
                    )


            # =====================================================
            # Simple evidence fusion
            # static_evidence / selection_only
                #
            # 没有 proposed calibration gate
            # 只做简单残差融合
            # =====================================================
            if self.ablation_mode in [
                "static_evidence",
                "selection_only"
            ]:
                anchor_info = self.anchor_proj(
                    self.amp_weight * modality_anchor
                )

                d_output = (
                    preliminary_state
                    + anchor_info
                )


            # =====================================================
            # Proposed gated calibration
            # calibration_only / full
            # =====================================================
            elif self.ablation_mode in [
                "calibration_only",
                "full"
            ]:
                d_output = self._anchor_calibration(
                    preliminary_state,
                    self.amp_weight * modality_anchor
                )

        # 6. Dynamic Evidence Selector & Anchor Calibration
        # modal_alpha = None
        # valid_modal_names = []
        # anchor_loss = torch.tensor(0.0, device=d_output.device)
        # L_dynamic = torch.tensor(0.0, device=d_output.device)

        # if self.use_dynamic_modal and modality_dict is not None:
        #     modality_anchor, gate_scores, modal_alpha, valid_modal_names = \
        #             self._fuse_modalities(modality_dict, d_output)
                    
        #     # 注意这里使用了 amp_weight 控制特征强弱
        #     d_output = self._anchor_calibration(
        #         d_output,
        #         self.amp_weight * modality_anchor
        #     )

        #     if train:
        #         # 几何余弦距离约束
        #         anchor_loss = self._time_step_anchor_loss(
        #             student_state=d_output,
        #             modality_anchor=modality_anchor,
        #             mask=mask
        #         )
        #         # 动态感知 InfoNCE 约束
        #         L_dynamic = self._dynamic_anchor_nce_loss(
        #             student_state=d_output,
        #             modality_dict=modality_dict,
        #             mask=mask,
        #             modal_alpha=modal_alpha,
        #             valid_names=valid_modal_names
        #         )

        # # 7. Knowledge Mastery Prediction
        # concat_q = torch.cat([d_output, d_output, q_embed_data, q_embed_data], dim=-1)
        # output = self.out(concat_q).squeeze(-1)
        # preds = torch.sigmoid(output)

        # temp = torch.ones(d_output.size(0), device=d_output.device)


        # if train:
        #     total_loss = torch.tensor(
        #         0.0,
        #         device=d_output.device
        #     )

        #     if self.use_cal_loss:
        #         total_loss = total_loss + \
        #             self.anchor_weight * anchor_loss

        #     if self.use_con_loss:
        #         total_loss = total_loss + \
        #             self.dynamic_weight * L_dynamic

        #     return preds, total_loss, temp

        # # if train:
        # #     # 返回加权后的辅助特征损失 (主损失在外部计算)
        # #     total_loss = self.anchor_weight * anchor_loss + self.dynamic_weight * L_dynamic
        # #     return preds, total_loss, temp
        # # else:
        # #     if qtest:
        # #         return preds, concat_q
        # #     else:
        # #         return preds

        # =========================================================
        # 7. Knowledge Mastery Prediction
        # =========================================================

        concat_q = torch.cat(
            [d_output, d_output, q_embed_data, q_embed_data],
            dim=-1
        )

        output = self.out(concat_q).squeeze(-1)
        preds = torch.sigmoid(output)

        temp = torch.ones(
            d_output.size(0),
            device=d_output.device
        )


        # =========================================================
        # 8. Return
        # =========================================================

        if train:

            total_loss = torch.tensor(
                0.0,
                device=d_output.device
            )

            if self.use_cal_loss:
                total_loss = (
                    total_loss
                    + self.anchor_weight * anchor_loss
                )

            if self.use_con_loss:
                total_loss = (
                    total_loss
                    + self.dynamic_weight * L_dynamic
                )

            return preds, total_loss, temp


        # Evaluation / Test
        if qtest:
            return preds, concat_q

        return preds


       
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(
            d_model,
            d_feature,
            n_heads,
            dropout,
            kq_same=(kq_same == 1)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, atten_type='dp', apply_pos=True):
        seqlen = query.size(1)

        diagonal = 1
        attn_mask = torch.triu(
            torch.ones(seqlen, seqlen, dtype=torch.bool, device=query.device),
            diagonal=diagonal
        )

        query2 = self.masked_attn_head(query, key, values, mask=attn_mask)
        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)

        if apply_pos:
            query2 = self.linear2(
                self.dropout(
                    self.activation(
                        self.linear1(query)
                    )
                )
            )
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)

        return query

        
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True
        )

    def forward(self, query, key, values, mask=None):
        output, _ = self.attn(
            query=query,
            key=key,
            value=values,
            attn_mask=mask,
            need_weights=False
        )
        return output