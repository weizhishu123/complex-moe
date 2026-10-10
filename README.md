# ComplexMoE: 复数混合专家中文语言模型

**48.62M 参数 | 单张 GTX 1080 Ti 训练 | 中文公文/新闻专用**

## 简介

ComplexMoE 是一个 48.62M 参数的复数混合专家（Mixture-of-Experts）语言模型，专为中文公文/新闻场景设计。在单张消费级 GPU 上完成 2 亿 token 训练。

## 核心特性

- 复数架构：注意力与 MoE 在复数域运算，位置信息通过相位旋转注入
- RoPE++ 位置编码：每层 Q/K 相对注入，非输入端一次性注入
- 多 K 训练：训练时 K 在 [32, 256] 随机采样，全范围可用
- 动态 base：base=10000，K=5 到 K=256 都稳定
- 短 prompt 友好：多 K 训练让模型适应短输入+填充模式，不锁死

## 快速开始

### 安装

pip install -r requirements.txt

### 生成

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

### 训练

python train/train.py --config configs/v15.yaml

## 评测结果

| 指标 | ComplexMoE-48M | GPT-2 small |
|------|----------------|-------------|
| 参数 | 48.62M | 117M |
| 中文公文 4-gram 重复率 | 1.60% | N/A |
| K=5 NLL | 6.52 | - |
| K=256 top1 | 0.19 | - |
| 训练硬件 | 单张 GTX 1080 Ti | - |

## 架构

Input tokens
    |
Complex Embedding
    |
ComplexBlock x 4
  - Complex Attention (RoPE)
  - ComplexMoE (4 experts, top-2)
  - Residual + LayerNorm
    |
Final Head -> logits (8000 vocab)

## 引用

@misc{yang2025complexmoe,
  title={ComplexMoE: A 48M-Parameter Complex-Valued Mixture-of-Experts Language Model for Chinese},
  author={Yang, Peng},
  year={2025}
}

## 许可证

Apache License 2.0
