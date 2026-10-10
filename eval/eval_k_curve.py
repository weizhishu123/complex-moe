# -*- coding: utf-8 -*-
"""K 曲线评测"""
import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.complex_lm import ComplexMoELM


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

    test_texts = []
    with open("test_data.jsonl", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if len(test_texts) >= 500:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                t = obj.get("Content", "")
                if isinstance(t, str) and len(t) > 100:
                    test_texts.append(t)
            except Exception:
                pass

    @torch.no_grad()
    def eval_at_K(K_test, n_samples=2000, batch=8):
        rng = random.Random(999)
        ctx_list, tgt_list = [], []
        for _ in range(n_samples):
            t = test_texts[rng.randrange(len(test_texts))]
            ids = [i for i in sp.encode(t, out_type=int) if i >= 4]
            if len(ids) <= K_test:
                continue
            p = rng.randrange(K_test, len(ids))
            ctx = ids[p - K_test:p]
            if K_test < cfg["MAX_K"]:
                padded = np.full(cfg["MAX_K"], ctx[0], dtype=np.int64)
                padded[cfg["MAX_K"] - K_test:] = ctx
                ctx_list.append(padded)
            else:
                ctx_list.append(ctx)
            tgt_list.append(ids[p])
        if not ctx_list:
            return 0.0, 0.0
        all_ct = np.array(ctx_list, dtype=np.int64)
        all_tt = np.array(tgt_list, dtype=np.int64)
        total_nll, total_top1, total = 0.0, 0, 0
        for i in range(0, len(all_ct), batch):
            cb = torch.from_numpy(all_ct[i:i+batch]).to(device)
            lg, _ = model(cb)
            lg = lg.float().cpu()
            tb = torch.from_numpy(all_tt[i:i+batch])
            lp = F.log_softmax(lg, dim=-1)
            total_nll += -lp[torch.arange(len(tb)), tb].sum().item()
            total_top1 += (lg.argmax(dim=-1) == tb).sum().item()
            total += len(tb)
        return total_nll / total, total_top1 / total

    print(f"{'K':>5} | {'NLL':>8} | {'top1':>8}")
    print("-" * 28)
    for K_test in [5, 20, 64, 128, 256]:
        if K_test > cfg["MAX_K"]:
            continue
        nll, top1 = eval_at_K(K_test)
        print(f"{K_test:>5} | {nll:>8.4f} | {top1:>8.4f}")


if __name__ == "__main__":
    main()
