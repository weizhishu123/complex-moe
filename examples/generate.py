# -*- coding: utf-8 -*-
"""生成示例"""
import os
import sys
import torch
import sentencepiece as spm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.complex_lm import ComplexMoELM


def generate(model, sp, prompt, length=150, temperature=0.8, top_k=20,
             penalty=1.3, window=20, K=256, device="cuda", seed=42):
    torch.manual_seed(seed)
    ids = [i for i in sp.encode(prompt, out_type=int) if i >= 4]
    if not ids:
        return ""
    if len(ids) < K:
        ctx = [ids[0]] * (K - len(ids)) + ids
    else:
        ctx = ids[-K:]
    out = list(ids)
    for _ in range(length):
        ct = torch.tensor([ctx[-K:]], dtype=torch.long, device=device)
        with torch.no_grad():
            lg, _ = model(ct)
        lg = lg[0].float() / temperature
        if penalty > 1.0:
            for t in set(ctx[-window:]):
                lg[t] = lg[t] / penalty if lg[t] > 0 else lg[t] * penalty
        if top_k:
            v, _ = torch.topk(lg, top_k)
            lg = torch.where(lg < v[-1], torch.full_like(lg, -1e9), lg)
        nxt = int(torch.multinomial(torch.softmax(lg, -1), 1))
        out.append(nxt)
        ctx.append(nxt)
    return sp.decode(out)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor()
    sp.load("bpe_v72_8000.model")
    n_vocab = sp.vocab_size()

    ckpt = torch.load("checkpoints/best.pt", map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = ComplexMoELM(
        n_vocab, cfg["D"], cfg["num_layers"], cfg["num_experts"],
        cfg["top_k"], cfg["ffn_ratio"], cfg["MAX_K"], 0.0, cfg["base"]
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    prompts = [
        "各县(市、区)人民政府，市直各部门：",
        "本报讯 昨日下午3时许，位于市区中心的",
        "新华社北京电 中共中央总书记、国家主席习近平",
    ]
    for p in prompts:
        out = generate(model, sp, p, device=device, K=cfg["MAX_K"])
        print(f"\n[prompt] {p}")
        print(f"[out] {out[:250]}")


if __name__ == "__main__":
    main()
