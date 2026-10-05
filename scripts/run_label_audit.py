"""Task 2：全量标签协议审计（三协议 × 四维度）+ 上界引理数字。

只读 DEAP labels 列，零训练，秒级。
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from prlpaper.data import deap
from prlpaper.labels.audit import apply_protocol, balance_shift, discard_rate, flip_rate

ROOT = Path(os.environ.get("DEAP_ROOT", "data/raw/deap/data_preprocessed_python"))
DIMS = ["valence", "arousal", "dominance", "liking"]  # labels 列序


def emt_protocol(ratings_by_subject):
    """EmT 2025 惯例：阈值 3.0（r > 3 -> 高）。"""
    return {sid: (np.asarray(r, float) > 3.0).astype(float) for sid, r in ratings_by_subject.items()}


def main():
    ratings = {d: {} for d in DIMS}
    for i in range(32):
        d = deap.load_subject(ROOT, f"s{i+1:02d}")
        for j, dim in enumerate(DIMS):
            ratings[dim][i] = d["labels"][:, j].astype(float)

    out = {}
    print(f"{'维度':10s} {'协议对':10s} {'改判率':>8s} {'丢弃率':>8s} {'平衡漂移':>8s}")
    print("-" * 52)
    for dim in DIMS:
        r = ratings[dim]
        lab = {p: apply_protocol(r, p) for p in ("P0", "P1", "P2")}
        pairs = [("P0", "P1"), ("P0", "P2"), ("P1", "P2")]
        out[dim] = {}
        for a, b in pairs:
            flip = flip_rate(lab[a], lab[b])
            dr = discard_rate(lab[b]) if b == "P1" else discard_rate(lab[a])
            bs = balance_shift(lab[a], lab[b])
            out[dim][f"{a}-{b}"] = {"flip": flip, "discard": dr, "balance_shift": bs}
            print(f"{dim:10s} {a+'-'+b:10s} {flip*100:7.2f}% {dr*100:7.2f}% {bs*100:+7.2f}%")
        # 逐被试改判分布（机制分析输入）
        out[dim]["per_subject_flip_P0P1"] = _per_subject_flip(lab["P0"], lab["P1"])
        out[dim]["per_subject_flip_P0P2"] = _per_subject_flip(lab["P0"], lab["P2"])
    # 落地例证：EmT 3.0 vs 领域 5.0（DEAP valence）
    emt = emt_protocol(ratings["valence"])
    flip_emt = flip_rate(apply_protocol(ratings["valence"], "P0"), emt)
    out["worked_example"] = {"EmT_3.0_vs_5.0_flip_valence": flip_emt}

    print("-" * 52)
    print(f"落地例证：EmT(3.0) vs 领域(5.0) 在 DEAP 效价上的改判率 = {flip_emt*100:.2f}%")

    # 锚定的真实作用：逐被试占比的离散度与锚点关联（论文 §3 末段与图 2c）
    print(f"\n{'维度':10s} {'P0 占比 SD':>10s} {'P2 占比 SD':>10s} "
          f"{'r(锚,P0)':>9s} {'r(锚,P2)':>9s} {'P2 占比范围':>14s}")
    print("-" * 62)
    anchor_stats = {}
    for dim in DIMS:
        r = ratings[dim]
        lab = {p: apply_protocol(r, p) for p in ("P0", "P2")}
        sids = sorted(r)
        share = {p: np.array([lab[p][s].mean() for s in sids]) for p in ("P0", "P2")}
        anchors = np.array([r[s].mean() - 5.0 for s in sids])
        sd = {p: float(share[p].std(ddof=1)) for p in ("P0", "P2")}
        rr = {p: float(np.corrcoef(anchors, share[p])[0, 1]) for p in ("P0", "P2")}
        anchor_stats[dim] = {
            "share_sd_P0": sd["P0"], "share_sd_P2": sd["P2"],
            "anchor_corr_P0": rr["P0"], "anchor_corr_P2": rr["P2"],
            "share_min_P2": float(share["P2"].min()), "share_max_P2": float(share["P2"].max()),
        }
        out[dim]["anchoring"] = anchor_stats[dim]
        print(f"{dim:10s} {sd['P0']*100:9.2f}% {sd['P2']*100:9.2f}% "
              f"{rr['P0']:+9.3f} {rr['P2']:+9.3f} "
              f"{share['P2'].min()*100:6.1f}%-{share['P2'].max()*100:5.1f}%")
    print("注：P2 不保证逐被试占比为 1/2；整数量表上的平局使占比偏离一半。")

    # 被试级自助区间（论文 §2.3 所述过程；固定种子，可复现）
    print(f"\n{'维度':10s} {'P0/P2 改判率':>12s} {'95% 自助区间':>20s} "
          f"{'P1 丢弃率':>10s} {'95% 自助区间':>20s}")
    print("-" * 78)
    boot = {}
    for dim in DIMS:
        r = ratings[dim]
        lab = {p: apply_protocol(r, p) for p in ("P0", "P1", "P2")}
        ci_f = _cluster_bootstrap(lab["P0"], lab["P2"], lambda a, b: flip_rate(a, b) * 100)
        ci_d = _cluster_bootstrap(lab["P1"], lab["P1"], lambda a, b: discard_rate(a) * 100)
        boot[dim] = {"flip_P0P2_ci95": [float(x) for x in ci_f],
                     "discard_P1_ci95": [float(x) for x in ci_d]}
        out[dim]["bootstrap"] = boot[dim]
        f = flip_rate(lab["P0"], lab["P2"]) * 100
        d = discard_rate(lab["P1"]) * 100
        print(f"{dim:10s} {f:11.2f}% [{ci_f[0]:7.2f}, {ci_f[1]:6.2f}] "
              f"{d:9.2f}% [{ci_d[0]:7.2f}, {ci_d[1]:6.2f}]")

    ci_e = _cluster_bootstrap(apply_protocol(ratings["valence"], "P0"), emt,
                              lambda a, b: flip_rate(a, b) * 100)
    out["worked_example"]["EmT_3.0_vs_5.0_ci95"] = [float(x) for x in ci_e]
    print(f"EmT(3.0) vs 惯例(5.0) 效价：{flip_emt*100:.2f}% "
          f"[{ci_e[0]:.2f}, {ci_e[1]:.2f}]")

    # GATE 出口
    v = out["valence"]
    h1 = min(v["P0-P1"]["flip"], v["P0-P2"]["flip"], v["P1-P2"]["flip"])
    print(f"\nGATE H1（协议间 flip ≥ 10%）: {'通过' if h1 >= 0.10 else '不通过'}  (最小 {h1*100:.2f}%)")
    print(f"GATE H2（valence 引理数字）: 报告数字在 ≥ {v['P0-P1']['flip']*100:.1f} 个百分点内不可比（P0↔P1）")
    a = out["valence"]["anchoring"]
    print(f"GATE H3（P2 价值）: 占比 SD {a['share_sd_P0']*100:.1f}→{a['share_sd_P2']*100:.1f} pt；"
          f"锚点相关 {a['anchor_corr_P0']:+.2f}→{a['anchor_corr_P2']:+.2f}")

    Path("results").mkdir(exist_ok=True)
    Path("results/label_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n已写出 results/label_audit.json")


def _per_subject_flip(a, b):
    out = {}
    for sid in a:
        aa, bb = np.asarray(a[sid], float), np.asarray(b[sid], float)
        valid = ~(np.isnan(aa) | np.isnan(bb))
        out[sid] = float((aa[valid] != bb[valid]).mean()) if valid.sum() > 0 else float("nan")
    return out


def _cluster_bootstrap(lab_a, lab_b, stat, n=10000, seed=0):
    """被试级自助：重抽被试（不是试次），把 32 名被试当作一个样本。

    审计本身无随机步骤；区间仅反映被试间差异，种子固定以保证可复现。
    """
    sids = sorted(lab_a)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.choice(sids, size=len(sids), replace=True)
        a = {j: lab_a[s] for j, s in enumerate(pick)}
        b = {j: lab_b[s] for j, s in enumerate(pick)}
        vals.append(stat(a, b))
    return np.percentile(vals, [2.5, 97.5])


if __name__ == "__main__":
    main()
