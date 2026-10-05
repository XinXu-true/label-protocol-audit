"""Task 4：机制分析（H4）与失效条件（H5）。

H4：改判集中于阈值邻域与高锚点偏移被试。
H5：P2 校正的失效条件（σ≈0 被试、锚点不漂移的维度）。
"""

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

sys.path.insert(0, "src")

from prlpaper.data import deap
from prlpaper.labels.audit import apply_protocol

ROOT = Path(os.environ.get("DEAP_ROOT", "data/raw/deap/data_preprocessed_python"))
DIMS = ["valence", "arousal", "dominance", "liking"]


def main():
    ratings = {d: {} for d in DIMS}
    for i in range(32):
        d = deap.load_subject(ROOT, f"s{i+1:02d}")
        for j, dim in enumerate(DIMS):
            ratings[dim][i] = d["labels"][:, j].astype(float)

    out = {}
    # ---- H4a：改判 × 距阈值距离（P0 vs P2，valence）----
    r = ratings["valence"]
    p0, p2 = apply_protocol(r, "P0"), apply_protocol(r, "P2")
    dist, flip = [], []
    for sid in r:
        rs = np.asarray(r[sid], float)
        f = (np.asarray(p0[sid], float) != np.asarray(p2[sid], float)).astype(float)
        dist.append(np.abs(rs - 5.0))
        flip.append(f)
    dist, flip = np.concatenate(dist), np.concatenate(flip)
    bins = [0, 0.5, 1.0, 2.0, 3.0, 4.1]
    labels = ["0-0.5", "0.5-1", "1-2", "2-3", "3-4"]
    bin_flip = [float(flip[(dist >= lo) & (dist < hi)].mean()) if ((dist >= lo) & (dist < hi)).any() else float("nan")
                for lo, hi in zip(bins[:-1], bins[1:])]
    out["h4a_bin_flip_by_distance"] = {l: v for l, v in zip(labels, bin_flip)}
    print("H4a 改判率 × 距阈值距离（valence, P0 vs P2）:")
    for l, v in zip(labels, bin_flip):
        print(f"  {l:8s} {v*100:6.2f}%")
    rho_a = pearsonr(dist, flip).statistic if len(dist) > 2 else float("nan")
    print(f"  Pearson r(dist, flip) = {rho_a:.3f}")

    # ---- H4b：逐被试改判 × 锚点偏移 ----
    subj_flip, anchor = [], []
    for sid in r:
        rs = np.asarray(r[sid], float)
        f = (np.asarray(p0[sid], float) != np.asarray(p2[sid], float)).mean()
        subj_flip.append(f)
        anchor.append(abs(rs.mean() - 5.0))
    subj_flip, anchor = np.array(subj_flip), np.array(anchor)
    rho_b = pearsonr(anchor, subj_flip).statistic
    out["h4b_anchor_corr"] = rho_b
    print(f"H4b 逐被试改判率 × 锚点偏移：Pearson r = {rho_b:.3f}")

    # ---- H5：失效条件 ----
    h5 = {}
    for dim in DIMS:
        sds = [np.asarray(v, float).std(ddof=0) for v in ratings[dim].values()]
        anchors = [abs(np.asarray(v, float).mean() - 5.0) for v in ratings[dim].values()]
        h5[dim] = {
            "n_sigma_lt_0.5": int(sum(1 for s in sds if s < 0.5)),
            "anchor_spread_sd": float(np.std(anchors)),
            "max_anchor": float(np.max(anchors)),
        }
    out["h5"] = h5
    print("H5 失效条件:")
    for dim, v in h5.items():
        print(f"  {dim:10s} σ<0.5 被试数={v['n_sigma_lt_0.5']}  锚点散布σ={v['anchor_spread_sd']:.2f}  最大锚点偏移={v['max_anchor']:.2f}")

    # ---- 图 ----
    Path("results/figs").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(labels, [v * 100 for v in bin_flip], color="#4C72B0")
    ax[0].set_xlabel("Distance to threshold |r - 5|")
    ax[0].set_ylabel("Flip rate (%), P0 vs P2")
    ax[0].set_title("Flips concentrate near the threshold")
    ax[1].scatter(anchor, subj_flip * 100, color="#C44E52")
    ax[1].set_xlabel("Subject anchor offset |mean(r) - 5|")
    ax[1].set_ylabel("Per-subject flip rate (%), P0 vs P2")
    ax[1].set_title(f"Anchor shift drives flips (r = {rho_b:.2f})")
    fig.tight_layout()
    fig.savefig("results/figs/fig_mechanism.pdf")
    fig.savefig("results/figs/mechanism.png", dpi=200)
    plt.close(fig)
    print("已写出 results/figs/mechanism.png")

    Path("results/mechanism.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("已写出 results/mechanism.json")


if __name__ == "__main__":
    main()
