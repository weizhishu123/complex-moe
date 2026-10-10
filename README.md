# ComplexMoE: 复数混合专家语言模型

**48.62M 参数 | 单张 GTX 1080 Ti 训练 | 通用架构 + 中文公文/新闻语料**

## 简介

ComplexMoE 是一个 48.62M 参数的复数混合专家（Mixture-of-Experts）语言模型。

**架构是领域无关的通用设计**，训练语料为中文公文与新闻文本（约 2 亿 token）。

## 架构 vs 数据

| 层 | 说明 |
|----|------|
| **架构** | 复数 Attention + 复数 MoE + RoPE++ + 多 K 训练。**领域无关**，可迁移到任意语言/领域 |
| **训练数据** | 中文公文/新闻，约 2 亿 token |
| **模型能力** | 仅在训练数据分布内验证，其他领域未测 |

架构创新点：
- **复数 Attention**：Q/K/V 在复数域运算，位置通过相位旋转注入
- **复数 MoE**：N 个复数专家 + top-k 路由
- **RoPE++**：每层 Q/K 相对位置注入
- **多 K 训练**：训练时 K 在 [32, 256] 随机采样，全范围可用
- **base=10000**：K=5 到 K=256 范围内稳定

## 快速开始

安装：

pip install -r requirements.txt

生成：

import torch
import sentencepiece as spm
from model.complex_lm import ComplexMoELM

sp = spm.SentencePieceProcessor()
sp.load("bpe_v72_8000.model")

ckpt = torch.load("checkpoints/best.pt", map_location="cuda", weights_only=False)
cfg = ckpt["config"]
model = ComplexMoELM(
    sp.vocab_size(), cfg["D"], cfg["num_layers"], cfg["num_experts"],
    cfg["top_k"], cfg["ffn_ratio"], cfg["MAX_K"], 0.0, cfg["base"]
).cuda()
model.load_state_dict(ckpt["model"])
model.eval()

prompt = "各县(市、区)人民政府，市直各部门："
ids = [i for i in sp.encode(prompt, out_type=int) if i >= 4]
ctx = list(ids)
for _ in range(150):
    ct = torch.tensor([ctx[-256:]], dtype=torch.long, device="cuda")
    with torch.no_grad():
        lg, _ = model(ct)
    ctx.append(int(lg[0].argmax()))
print(sp.decode(ctx))

## 评测结果

| 指标 | ComplexMoE-48M | GPT-2 small (117M) |
|------|----------------|---------------------|
| 参数 | 48.62M | 117M |
| 架构 | 复数 MoE（通用） | 实数 Transformer（通用） |
| 训练语料 | 中文公文/新闻 2 亿 token | 英文 WebText 40 GB |
| 中文公文续写 4-gram 重复率 | 1.60% | N/A（无中文能力） |
| K=5 NLL | 6.52 | - |
| K=256 top1 | 0.19 | - |
| 训练硬件 | 单张 GTX 1080 Ti | - |

## 限制

- 训练语料仅中文公文/新闻，未覆盖其他领域
- 通用能力（对话、百科、代码等）未验证
- 上下文窗口 256 token

## 架构

Input tokens
    |
Complex Embedding (a, b 归一化为单位圆)
    |
ComplexBlock x 4          <- 通用组件, 与领域无关
  - Complex Attention (RoPE 旋转)
  - ComplexMoE (4 专家, top-2)
  - 残差连接 + LayerNorm
    |
Final Head -> logits (8000 词表)

**架构本身可用于**: 任何语言、任何领域、任何规模的 MoE 变体。
本仓库的权重只是在中文公文/新闻上训练的**一个实例**。

## 引用

@misc{yang2025complexmoe,
  title={ComplexMoE: A Complex-Valued Mixture-of-Experts Language Model Architecture},
  author={Yang, Peng},
  year={2025},
  url={https://github.com/weizhishu123/complex-moe}
}

## 许可证

Apache License 2.0
