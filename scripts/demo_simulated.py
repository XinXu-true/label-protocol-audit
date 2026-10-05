"""在模拟评分上跑完整的标签协议审计，不需要 DEAP 数据。

用途有两个：读者在没有 DEAP 授权的情况下也能核查实现；投稿仓库与 CI 可在无数据环境自检。
模拟评议按真实 DEAP 的量表（1–9 整数）与规模（32 名被试 × 40 试次）生成，
并注入跨被试的锚点差异，使三种协议的差异方向与真实数据一致。
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from prlpaper.labels.audit import apply_protocol, balance_shift, discard_rate, flip_rate

N_SUBJECTS = 32
N_TRIALS = 40
DIMS = ["valence", "arousal", "dominance", "liking"]


def simulate_ratings(seed: int = 0) -> dict[int, np.ndarray]:
    """量表 1–9 的整数评分；每名被试有自己的锚点，模拟反应风格差异。"""
    rng = np.random.default_rng(seed)
    ratings = {}
    for sid in range(N_SUBJECTS):
        anchor = rng.normal(0.0, 0.8)                 # 被试间锚点差异
        width = rng.uniform(1.2, 2.2)                 # 评分离散度差异
        raw = rng.normal(5.0 + anchor, width, N_TRIALS)
        ratings[sid] = np.clip(np.rint(raw), 1, 9)
    return ratings


def emt_variant(ratings_by_subject):
    """对照：某已发表工作在另一个数据集上使用的阈值三。"""
    return {sid: (np.asarray(r, float) > 3.0).astype(float)
            for sid, r in ratings_by_subject.items()}


def main() -> int:
    rng = np.random.default_rng(0)
    print("模拟数据审计（32 名被试 × 40 试次 × 4 维度，量表 1–9）")
    print("=" * 66)
    print(f"{'维度':10s} {'P0/P1 改判':>10s} {'P1 丢弃':>9s} {'P0/P2 改判':>11s} "
          f"{'平衡漂移':>9s} {'阈值三':>8s}")
    print("-" * 66)

    failures = []
    for dim in DIMS:
        ratings = simulate_ratings(seed=abs(hash(dim)) % 10000)
        lab = {p: apply_protocol(ratings, p) for p in ("P0", "P1", "P2")}
        f01 = flip_rate(lab["P0"], lab["P1"])
        disc = discard_rate(lab["P1"])
        f02 = flip_rate(lab["P0"], lab["P2"])
        bs = balance_shift(lab["P0"], lab["P2"])
        f3 = flip_rate(lab["P0"], emt_variant(ratings))
        print(f"{dim:10s} {f01*100:9.2f}% {disc*100:8.2f}% {f02*100:10.2f}% "
              f"{bs*100:+8.2f}% {f3*100:7.2f}%")

        # 模拟数据上同样应成立的结构性事实
        if f01 != 0.0:
            failures.append(f"{dim}: P0↔P1 改判应为 0（P1 靠丢弃而非改标签）")
        if not (0.10 <= disc <= 0.40):
            failures.append(f"{dim}: 丢弃率 {disc:.2%} 超出模拟设定范围")
        if f02 <= 0.0:
            failures.append(f"{dim}: P0↔P2 改判应大于 0")

    print("-" * 66)
    # 与真实数据同向的机制检查：改判集中在阈值附近
    ratings = simulate_ratings(seed=0)
    p0, p2 = apply_protocol(ratings, "P0"), apply_protocol(ratings, "P2")
    near, far = [], []
    for sid in sorted(ratings):
        r = ratings[sid]
        flips = (np.asarray(p0[sid]) != np.asarray(p2[sid]))
        near.append(flips[np.abs(r - 5.0) < 0.5].mean() if (np.abs(r - 5.0) < 0.5).any() else np.nan)
        far.append(flips[np.abs(r - 5.0) > 2.0].mean() if (np.abs(r - 5.0) > 2.0).any() else np.nan)
    near_m, far_m = np.nanmean(near), np.nanmean(far)
    print(f"阈值邻域（|r−5|<0.5）改判率: {near_m:.2%}；远离阈值（>2）改判率: {far_m:.2%}")
    if not near_m > far_m:
        failures.append("改判未集中在阈值邻域")
    print()

    if failures:
        print("模拟自检未通过：")
        for f in failures:
            print("  -", f)
        return 1
    print("模拟自检通过：三种协议的方向与机制在合成数据上复现。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
