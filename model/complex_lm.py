# -*- coding: utf-8 -*-
"""ComplexMoE: 复数混合专家语言模型"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def complex_ln(xr, xi, eps=1e-6):
    c = torch.cat([xr, xi], dim=-1)
    c = F.layer_norm(c, [c.size(-1)], eps=eps)
    D_ = xr.size(-1)
    return c[..., :D_], c[..., D_:]


class ComplexRoPE(nn.Module):
    def __init__(self, D, max_seq_len=256, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, D).float() / D))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def forward(self, xr, xi):
        S = xr.size(-2)
        cos = self.cos_cached[:S]
        sin = self.sin_cached[:S]
        return xr * cos - xi * sin, xr * sin + xi * cos


class ComplexExpert(nn.Module):
    def __init__(self, D, ffn_ratio=4):
        super().__init__()
        H = D * ffn_ratio
        self.W1_re = nn.Linear(D, H, bias=False)
        self.W1_im = nn.Linear(D, H, bias=False)
        self.W2_re = nn.Linear(H, D, bias=False)
        self.W2_im = nn.Linear(H, D, bias=False)

    def _c(self, xr, xi, Wr, Wi):
        return Wr(xr) - Wi(xi), Wr(xi) + Wi(xr)

    def forward(self, xr, xi):
        hr, hi = self._c(xr, xi, self.W1_re, self.W1_im)
        hr, hi = F.gelu(hr), F.gelu(hi)
        return self._c(hr, hi, self.W2_re, self.W2_im)


class ComplexMoE(nn.Module):
    def __init__(self, D, num_experts=4, top_k=2, ffn_ratio=4):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.D = D
        self.router = nn.Linear(2 * D, num_experts, bias=False)
        self.experts = nn.ModuleList([
            ComplexExpert(D, ffn_ratio) for _ in range(num_experts)
        ])

    def forward(self, xr, xi):
        B, Kk, D = xr.shape
        xrf = xr.reshape(-1, D)
        xif = xi.reshape(-1, D)
        xf = torch.cat([xrf, xif], dim=-1)
        rp = F.softmax(self.router(xf), dim=-1)
        tp, ti = rp.topk(self.top_k, dim=-1)
        tp = tp / (tp.sum(-1, keepdim=True) + 1e-8)
        er, ei = [], []
        for e in self.experts:
            a, b = e(xrf, xif)
            er.append(a)
            ei.append(b)
        ar = torch.stack(er, dim=1)
        ai = torch.stack(ei, dim=1)
        w = torch.zeros(xf.size(0), self.num_experts, device=xr.device)
        for k in range(self.top_k):
            w.scatter_add_(1, ti[:, k:k+1], tp[:, k:k+1])
        or_ = (ar * w.unsqueeze(-1)).sum(1)
        oi = (ai * w.unsqueeze(-1)).sum(1)
        mp = rp.mean(0)
        sc = torch.bincount(ti.flatten(), minlength=self.num_experts).float()
        aux = (mp * (sc / ti.numel())).sum() * self.num_experts
        return or_.reshape(B, Kk, D), oi.reshape(B, Kk, D), aux


class ComplexAttentionRoPEPP(nn.Module):
    def __init__(self, D, dropout=0.1, max_seq_len=256, base=10000.0):
        super().__init__()
        self.D = D
        self.Wq_re = nn.Linear(D, D, bias=False)
        self.Wq_im = nn.Linear(D, D, bias=False)
        self.Wk_re = nn.Linear(D, D, bias=False)
        self.Wk_im = nn.Linear(D, D, bias=False)
        self.Wv_re = nn.Linear(D, D, bias=False)
        self.Wv_im = nn.Linear(D, D, bias=False)
        self.Wo_re = nn.Linear(D, D, bias=False)
        self.Wo_im = nn.Linear(D, D, bias=False)
        self.rope = ComplexRoPE(D, max_seq_len, base)
        self.dropout = nn.Dropout(dropout)

    def _c(self, xr, xi, Wr, Wi):
        return Wr(xr) - Wi(xi), Wr(xi) + Wi(xr)

    def forward(self, xr, xi):
        Dd = xr.size(-1)
        Qr, Qi = self._c(xr, xi, self.Wq_re, self.Wq_im)
        Kr, Ki = self._c(xr, xi, self.Wk_re, self.Wk_im)
        Vr, Vi = self._c(xr, xi, self.Wv_re, self.Wv_im)
        Qr, Qi = self.rope(Qr, Qi)
        Kr, Ki = self.rope(Kr, Ki)
        scale = math.sqrt(Dd)
        real_score = (Qr @ Kr.transpose(-2, -1) + Qi @ Ki.transpose(-2, -1)) / scale
        imag_score = (Qi @ Kr.transpose(-2, -1) - Qr @ Ki.transpose(-2, -1)) / scale
        attn = F.softmax(real_score - imag_score, dim=-1)
        attn = self.dropout(attn)
        return self._c(attn @ Vr, attn @ Vi, self.Wo_re, self.Wo_im)


class ComplexBlock(nn.Module):
    def __init__(self, D, num_experts=4, top_k=2, ffn_ratio=4,
                 dropout=0.1, max_seq_len=256, base=10000.0):
        super().__init__()
        self.attn = ComplexAttentionRoPEPP(D, dropout, max_seq_len, base)
        self.moe = ComplexMoE(D, num_experts, top_k, ffn_ratio)
        self.dropout = nn.Dropout(dropout)

    def forward(self, xr, xi):
        xrn, xin = complex_ln(xr, xi)
        ar, ai = self.attn(xrn, xin)
        xr = xr + self.dropout(ar)
        xi = xi + self.dropout(ai)
        xrn, xin = complex_ln(xr, xi)
        fr, fi, aux = self.moe(xrn, xin)
        xr = xr + self.dropout(fr)
        xi = xi + self.dropout(fi)
        return xr, xi, aux


class ComplexMoELM(nn.Module):
    def __init__(self, n_vocab, D, num_layers=4, num_experts=4, top_k=2,
                 ffn_ratio=4, max_K=256, dropout=0.1, base=10000.0):
        super().__init__()
        self.D = D
        self.a = nn.Parameter(torch.randn(n_vocab, D) * 0.5)
        self.b = nn.Parameter(torch.randn(n_vocab, D) * 0.5)
        self.blocks = nn.ModuleList([
            ComplexBlock(D, num_experts, top_k, ffn_ratio, dropout, max_K, base)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def _n(self):
        nm = torch.sqrt(self.a ** 2 + self.b ** 2 + 1e-8)
        return self.a / nm, self.b / nm

    def embed(self, ctx):
        re, im = self._n()
        return re[ctx], im[ctx]

    def forward(self, ctx):
        xr, xi = self.embed(ctx)
        re, im = self._n()
        aux_t = 0.0
        for blk in self.blocks:
            xr, xi, aux = blk(xr, xi)
            aux_t = aux_t + aux
        cr = xr[:, -1, :]
        ci = xi[:, -1, :]
        logits = (cr @ re.t() + ci @ im.t()) / math.sqrt(self.D)
        return logits, aux_t
