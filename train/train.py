# -*- coding: utf-8 -*-
"""ComplexMoE 训练脚本"""
import os
import sys
import json
import random
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import sentencepiece as spm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.complex_lm import ComplexMoELM


def load_config(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def sample_batch_multik(seqs, bs, K_min, K_max, rng):
    ct = np.empty((bs, K_max), dtype=np.int64)
    tt = np.empty((bs,), dtype=np.int64)
    n = len(seqs)
    b = 0
    while b < bs:
        i = int(rng.integers(0, n))
        s = seqs[i]
        if len(s) <= K_max:
            continue
        K_actual = int(rng.integers(K_min, K_max + 1))
        p = int(rng.integers(K_actual, len(s)))
        ctx = s[p - K_actual:p]
        if K_actual < K_max:
            padded = np.full(K_max, ctx[0], dtype=np.int64)
            padded[K_max - K_actual:] = ctx
            ct[b] = padded
        else:
            ct[b] = ctx
        tt[b] = s[p]
        b += 1
    return torch.from_numpy(ct), torch.from_numpy(tt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v15.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.get("use_amp", False) and torch.cuda.is_available()
    print(f"Device: {device}, use_amp: {use_amp}")

    sp = spm.SentencePieceProcessor()
    sp.load(cfg["bpe_path"])
    n_vocab = sp.vocab_size()

    print("加载数据...")
    all_texts = []
    for path in cfg["data_files"]:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                t = obj.get("Content", "") or obj.get("content", "") or obj.get("text", "")
                if isinstance(t, str) and len(t.strip()) >= 20:
                    all_texts.append(t.strip())
        if cfg.get("n_gongwen") and len(all_texts) >= cfg["n_gongwen"]:
            break

    wiki_path = cfg.get("wiki_path")
    if wiki_path and os.path.exists(wiki_path):
        wiki_texts = []
        with open(wiki_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if len(wiki_texts) >= cfg.get("n_wiki", 0):
                    break
                line = line.strip()
                if len(line) >= 50:
                    wiki_texts.append(line)
        all_texts.extend(wiki_texts)
        print(f"  wiki: {len(wiki_texts):,}")

    random.Random(42).shuffle(all_texts)

    print("编码...")
    all_seqs = []
    for t in all_texts:
        ids = [i for i in sp.encode(t, out_type=int) if i >= 4]
        if len(ids) > cfg["MAX_K"]:
            all_seqs.append(np.asarray(ids[:cfg["seq_maxlen"]], dtype=np.int64))
    print(f"  {len(all_seqs):,} seqs")

    random.Random(0).shuffle(all_seqs)
    split = int(len(all_seqs) * 0.95)
    train_seqs = all_seqs[:split]
    val_seqs = all_seqs[split:]
    print(f"  train={len(train_seqs):,}, val={len(val_seqs):,}")

    print("构建模型...")
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    model = ComplexMoELM(
        n_vocab=n_vocab,
        D=cfg["D"],
        num_layers=cfg["num_layers"],
        num_experts=cfg["num_experts"],
        top_k=cfg["top_k"],
        ffn_ratio=cfg["ffn_ratio"],
        max_K=cfg["MAX_K"],
        dropout=cfg["dropout"],
        base=cfg["base"],
    ).to(device)
    print(f"  params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    opt = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    steps_total = cfg["samples_per_epoch"] // (cfg["batch_size"] * cfg["grad_accum"])
    warmup = optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=cfg["warmup_steps"])
    cosine = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps_total - cfg["warmup_steps"]))
    sched = optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[cfg["warmup_steps"]])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    print(f"训练 {steps_total:,} 步...")
    model.train()
    rng_np = np.random.default_rng(2025)
    t_start = time.time()

    for step in range(steps_total):
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(cfg["grad_accum"]):
            ct, tt = sample_batch_multik(train_seqs, cfg["batch_size"],
                                          cfg["K_MIN"], cfg["K_MAX"], rng_np)
            ct = ct.to(device)
            tt = tt.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                logits, aux = model(ct)
                loss = F.cross_entropy(logits, tt, label_smoothing=0.05) + cfg["aux_loss_weight"] * aux
            loss = loss / cfg["grad_accum"]
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += loss.item() * cfg["grad_accum"]

        if use_amp:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
        try:
            sched.step()
        except Exception:
            pass

        if (step + 1) % 500 == 0:
            torch.cuda.empty_cache()

        if (step + 1) % 2000 == 0:
            el = time.time() - t_start
            eta = el / (step + 1) * (steps_total - step - 1)
            print(f"  step {step+1}/{steps_total} | loss={accum_loss/cfg['grad_accum']:.4f} | ETA {eta/3600:.1f}h", flush=True)

        if (step + 1) % 1000 == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "global_step": step + 1,
                "config": cfg,
            }, os.path.join(cfg["ckpt_dir"], "latest.pt"))

    print(f"训练完成: {(time.time()-t_start)/3600:.1f}h")
    torch.save({"model": model.state_dict(), "config": cfg},
               os.path.join(cfg["ckpt_dir"], "best.pt"))


if __name__ == "__main__":
    main()
