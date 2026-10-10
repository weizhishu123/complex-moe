# -*- coding: utf-8 -*-
"""事实问答评测"""
import os
import sys
import torch
import torch.nn.functional as F
import sentencepiece as spm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.complex_lm import ComplexMoELM


CASES = [
    ("中国的首都是", ["北京", "上海"]),
    ("水的化学式是", ["H2O", "CO2"]),
    ("中华人民共和国的首任主席是", ["毛泽东", "邓小平"]),
    ("光速约等于", ["30万公里每秒", "3万公里每秒"]),
    ("氢的化学符号是", ["H", "He"]),
    ("地球上最大的洋是", ["太平洋", "印度洋"]),
    ("地球的卫星是", ["月球", "太阳"]),
    ("圆周率约等于", ["3.14", "2.71"]),
    ("一年有", ["十二个月", "四季"]),
    ("中华人民共和国的成立时间是", ["1949年", "1921年"]),
]


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

    MAX_K = cfg["MAX_K"]

    @torch.no_grad()
    def phrase_total_logp(ctx_text, phrase_text):
        ctx = [i for i in sp.encode(ctx_text, out_type=int) if i >= 4]
        phr = [i for i in sp.encode(phrase_text, out_type=int) if i >= 4]
        if not ctx or not phr:
            return 0.0
        if len(ctx) < MAX_K:
            cur = [ctx[0]] * (MAX_K - len(ctx)) + ctx
        else:
            cur = ctx[-MAX_K:]
        total = 0.0
        for pid in phr:
            ct = torch.tensor([cur[-MAX_K:]], dtype=torch.long, device=device)
            lg, _ = model(ct)
            total += F.log_softmax(lg[0].float(), dim=-1)[pid].item()
            cur = (cur + [pid])[-MAX_K:]
        return total

    n_ok = 0
    for ctx, cands in CASES:
        scores = [(c, phrase_total_logp(ctx, c)) for c in cands]
        scores.sort(key=lambda x: -x[1])
        ok = scores[0][0] == cands[0]
        if ok:
            n_ok += 1
        print(f"{'OK ' if ok else 'XX '} {ctx!r:<28} -> {scores[0][0]}")

    print(f"事实问答: {n_ok}/{len(CASES)}")


if __name__ == "__main__":
    main()
