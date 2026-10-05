"""标签协议审计核心（plan Task 1）。

三协议的标签构造：
  P0（领域主流）：r <= 5.0 -> 0，r > 5.0 -> 1。tie 规则与 LibEER label_process
      的 `value <= bounds[0]` 逐点一致（5.0 判低类）。
  P1（丢弃模糊区，LibEER `bounds 4 6` 变体）：r <= 4.0 -> 0，r >= 6.0 -> 1，
      4.0 < r < 6.0 -> NaN（该试次被丢弃）。
  P2（逐被试锚定，本文修复层 λ=0 特例）：z = (r - mu_s)/sd_s，z >= 0 -> 1，
      z < 0 -> 0；sd_s < 1e-8 的被试回退全局标准差。

全部为确定性纯统计，零训练。
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8


def apply_protocol(
    ratings_by_subject: dict[int, np.ndarray],
    protocol: str,
) -> dict[int, np.ndarray]:
    """按协议把逐被试评分转换为标签；NaN 表示该试次被丢弃（P1）。"""
    if protocol not in ("P0", "P1", "P2"):
        raise ValueError(f"protocol ∈ {{P0,P1,P2}}，收到 {protocol}")
    if protocol in ("P0", "P1"):
        out = {}
        for sid, r in ratings_by_subject.items():
            r = np.asarray(r, dtype=float)
            if protocol == "P0":
                out[sid] = (r > 5.0).astype(float)          # 5.0 判低类
            else:
                lab = np.full_like(r, np.nan)
                lab[r <= 4.0] = 0.0
                lab[r >= 6.0] = 1.0
                out[sid] = lab
        return out
    # P2：逐被试 z 锚定
    mus = {sid: float(np.asarray(r, float).mean()) for sid, r in ratings_by_subject.items()}
    all_r = np.concatenate([np.asarray(r, float) for r in ratings_by_subject.values()])
    global_sd = float(all_r.std(ddof=0))
    global_sd = global_sd if global_sd > EPS else 1.0
    out = {}
    for sid, r in ratings_by_subject.items():
        r = np.asarray(r, dtype=float)
        sd = float(r.std(ddof=0))
        sd = sd if sd > EPS else global_sd               # σ≈0 回退
        out[sid] = ((r - mus[sid]) / sd >= 0.0).astype(float)
    return out


def _concat(labels: dict[int, np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(v, float) for v in labels.values()])


def flip_rate(a: dict[int, np.ndarray], b: dict[int, np.ndarray], mask: dict[int, np.ndarray] | None = None) -> float:
    """配对改判率：仅在两协议都保留（非 NaN）的试次上计算。"""
    aa, bb = _concat(a), _concat(b)
    valid = ~(np.isnan(aa) | np.isnan(bb))
    if mask is not None:
        valid &= _concat(mask).astype(bool)
    n = int(valid.sum())
    return float((aa[valid] != bb[valid]).sum() / n) if n > 0 else float("nan")


def discard_rate(labels: dict[int, np.ndarray]) -> float:
    """P1 丢弃率：NaN 试次占比。"""
    v = _concat(labels)
    return float(np.isnan(v).mean())


def balance_shift(a: dict[int, np.ndarray], b: dict[int, np.ndarray], mask: dict[int, np.ndarray] | None = None) -> float:
    """类别平衡漂移（a 的高类占比 − b 的高类占比），在共同保留集上。"""
    aa, bb = _concat(a), _concat(b)
    valid = ~(np.isnan(aa) | np.isnan(bb))
    if mask is not None:
        valid &= _concat(mask).astype(bool)
    return float(aa[valid].mean() - bb[valid].mean()) if valid.sum() > 0 else float("nan")
