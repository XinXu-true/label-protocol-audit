"""Publication figures for the label-protocol audit (Python backend, exclusive).

Figure 1 — what the protocols do to the data
  (a) decision bands: the class each protocol assigns to every rating value
  (b) pairwise flip-rate matrix across protocols and dimensions
  (c) trial-set consequences of P1 (discarded share)
  (d) class balance by protocol and dimension

Figure 2 — mechanism and correction
  (a) flip rate by distance to the threshold
  (b) per-participant flip rate against anchor offset
  (c) per-participant class balance: P0 spreads, P2 pins everyone at 50 percent
  (d) diagnostics: anchor spread and rating dispersion per dimension

All quantities are recomputed here from the DEAP ratings, so the figures are
deterministic. Geometry keeps every text element inside its axes so that
tight_layout succeeds and the alignment gate can measure the panels.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, "src")

try:  # 面板对齐门属于绘图工具链的可选依赖，缺失时不影响出图
    from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
except ImportError:  # pragma: no cover
    def require_matplotlib_panel_alignment(*args, **kwargs):
        return None
from prlpaper.data import deap  # noqa: E402
from prlpaper.labels.audit import apply_protocol  # noqa: E402

ROOT = Path(os.environ.get("DEAP_ROOT", "data/raw/deap/data_preprocessed_python"))
OUT = Path("results/figs")
DIMS = ["valence", "arousal", "dominance", "liking"]
DIM_LABEL = {"valence": "Valence", "arousal": "Arousal",
             "dominance": "Dominance", "liking": "Liking"}

C = {"P0": "#4477AA", "P1": "#EE8866", "P2": "#44AA99", "neutral": "#777777",
     "light": "#DDDDDD", "warn": "#CC6677"}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "figure.dpi": 200,
})


def load_ratings():
    ratings = {d: {} for d in DIMS}
    for i in range(32):
        d = deap.load_subject(ROOT, f"s{i+1:02d}")
        for j, dim in enumerate(DIMS):
            ratings[dim][i] = d["labels"][:, j].astype(float)
    return ratings


def flat(labels):
    return np.concatenate([np.asarray(v, float) for v in labels.values()])


def flip(a, b):
    aa, bb = flat(a), flat(b)
    v = ~(np.isnan(aa) | np.isnan(bb))
    return float((aa[v] != bb[v]).mean())


def discard(labels):
    return float(np.isnan(flat(labels)).mean())


def high_frac(labels):
    return float(np.nanmean(flat(labels)))


def panel_label(ax, s):
    ax.set_title(s, loc="left", fontweight="bold", pad=4)


def save_pub(fig, stem: Path):
    require_matplotlib_panel_alignment(
        fig, json_out=f"{stem}.alignment.json", overlay_svg=f"{stem}.alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
    )
    for ext in ("svg", "pdf"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=400, bbox_inches="tight")
    print(f"wrote {stem}.pdf / .svg / .png")


def fig1(ratings, prot):
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.7))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # (a) decision bands, everything inside the axes
    ra = np.arange(1, 10)
    band_h = 0.26
    y_p0, y_p1, y_key = 0.66, 0.34, 0.03
    for r in ra:
        for y, proto in ((y_p0, "P0"), (y_p1, "P1")):
            if proto == "P0":
                d = 1 if r > 5 else 0
            else:
                d = 0 if r <= 4 else (1 if r >= 6 else -1)
            face = C[proto] if d == 1 else (C["light"] if d == 0 else C["warn"])
            ax_a.add_patch(Rectangle((r - 0.46, y), 0.92, band_h, facecolor=face,
                                     edgecolor="white", lw=0.7))
            ax_a.text(r, y + band_h / 2, "H" if d == 1 else ("L" if d == 0 else "x"),
                      ha="center", va="center", fontsize=6.0,
                      color="white" if d == 1 else "#333333")
    ax_a.text(0.4, y_p0 + band_h / 2, "P0", ha="right", va="center", fontsize=6.8,
              fontweight="bold")
    ax_a.text(0.4, y_p1 + band_h / 2, "P1", ha="right", va="center", fontsize=6.8,
              fontweight="bold")
    ax_a.text(0.4, y_p1 - 0.13, "P2", ha="right", va="center", fontsize=6.8,
              fontweight="bold")
    ax_a.text(1.0, y_p1 - 0.13, "relative to own mean:  below  L     at or above  H",
              ha="left", va="center", fontsize=6.0, color="#2E7D6F")
    ax_a.text(1.0, y_key, "H high     L low     x discarded", ha="left", va="center",
              fontsize=6.0, color=C["neutral"])
    ax_a.set_xlim(-0.15, 9.75)
    ax_a.set_ylim(y_key - 0.05, y_p0 + band_h + 0.07)
    ax_a.set_yticks([])
    ax_a.set_xticks(ra)
    ax_a.set_xlabel("self-report rating")
    ax_a.spines["left"].set_visible(False)
    panel_label(ax_a, "(a)")

    # (b) pairwise flip-rate matrix
    M = np.full((3, 4), np.nan)
    pairs = [("P0", "P1"), ("P0", "P2"), ("P1", "P2")]
    for r, (a, b) in enumerate(pairs):
        for c, dim in enumerate(DIMS):
            M[r, c] = flip(prot[dim][a], prot[dim][b]) * 100
    ax_b.imshow(M, cmap="OrRd", vmin=0, vmax=30, aspect="auto")
    ax_b.set_xticks(range(4), [DIM_LABEL[d] for d in DIMS], rotation=20, ha="right")
    ax_b.set_yticks(range(3), [f"{a} vs {b}" for a, b in pairs])
    for r in range(3):
        for c in range(4):
            ax_b.text(c, r, f"{M[r, c]:.1f}", ha="center", va="center",
                      fontsize=6.2, color="white" if M[r, c] > 18 else "#333333")
    ax_b.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax_b.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax_b.grid(which="minor", color="white", lw=1.0)
    ax_b.tick_params(which="minor", length=0)
    panel_label(ax_b, "(b)")

    # (c) trial-set consequences of P1, no legend (note drawn inside the axes)
    x = np.arange(4)
    kept = [100 * (1 - discard(prot[d]["P1"])) for d in DIMS]
    drop = [100 * discard(prot[d]["P1"]) for d in DIMS]
    ax_c.bar(x, kept, 0.62, color=C["light"], edgecolor=C["neutral"], lw=0.6)
    ax_c.bar(x, drop, 0.62, bottom=kept, color=C["P1"], edgecolor="none")
    for i, dv in enumerate(drop):
        ax_c.text(i, kept[i] + dv + 2.2, f"{dv:.1f}%", ha="center", fontsize=6.2)
    ax_c.text(0.02, 0.97, "orange: discarded by P1", transform=ax_c.transAxes,
              ha="left", va="top", fontsize=6.0, color="#B4562E")
    ax_c.set_xticks(x, [DIM_LABEL[d] for d in DIMS], rotation=20, ha="right")
    ax_c.set_ylim(0, 122)
    ax_c.set_yticks([0, 25, 50, 75, 100])
    ax_c.set_ylabel("trials (%)")
    panel_label(ax_c, "(c)")

    # (d) class balance by protocol
    w = 0.26
    for k, p in enumerate(["P0", "P1", "P2"]):
        vals = [high_frac(prot[d][p]) * 100 for d in DIMS]
        ax_d.bar(x + (k - 1) * w, vals, w, color=C[p], edgecolor="none", label=p)
    ax_d.axhline(50, color=C["neutral"], lw=0.8, ls=(0, (3, 2)))
    ax_d.text(0.02, 0.97, "dashed line: balanced", transform=ax_d.transAxes,
              fontsize=6.0, color=C["neutral"], ha="left", va="top")
    ax_d.set_xticks(x, [DIM_LABEL[d] for d in DIMS], rotation=20, ha="right")
    ax_d.set_ylabel("high-class share (%)")
    ax_d.set_ylim(0, 84)
    ax_d.set_yticks([0, 20, 40, 60])
    ax_d.legend(loc="upper right", ncol=3, handlelength=1.1, handletextpad=0.4,
                columnspacing=0.9, borderaxespad=0.2)
    panel_label(ax_d, "(d)")

    fig.tight_layout(w_pad=1.8, h_pad=1.8)
    return fig


def fig2(ratings, prot):
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.7))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    r = ratings["valence"]
    p0, p2 = prot["valence"]["P0"], prot["valence"]["P2"]

    # (a) flip rate vs distance to threshold
    dist, flipv = [], []
    for sid in r:
        rs = np.asarray(r[sid], float)
        f = (np.asarray(p0[sid], float) != np.asarray(p2[sid], float)).astype(float)
        dist.append(np.abs(rs - 5.0))
        flipv.append(f)
    dist, flipv = np.concatenate(dist), np.concatenate(flipv)
    edges = [0, 0.5, 1, 2, 3, 4.1]
    ticks = ["0-0.5", "0.5-1", "1-2", "2-3", "3-4"]
    vals, ns = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dist >= lo) & (dist < hi)
        vals.append(100 * flipv[m].mean() if m.any() else 0.0)
        ns.append(int(m.sum()))
    ax_a.bar(range(5), vals, 0.62, color=C["P0"], edgecolor="none")
    for i, v in enumerate(vals):
        if v > 1:
            ax_a.text(i, v + 1.6, f"{v:.1f}", ha="center", fontsize=6.2)
    ax_a.set_xticks(range(5), ticks)
    ax_a.set_xlabel("distance from threshold |r - 5|")
    ax_a.set_ylabel("flip rate P0 vs P2 (%)")
    ax_a.set_ylim(-9, 58)
    for i, n in enumerate(ns):
        ax_a.text(i, -6.0, f"n={n}", ha="center", fontsize=5.6, color=C["neutral"])
    panel_label(ax_a, "(a)")

    # (b) per-participant flip rate vs anchor offset
    subj_flip = np.array([(np.asarray(p0[s], float) != np.asarray(p2[s], float)).mean()
                          for s in sorted(r)])
    anchor = np.array([abs(np.asarray(r[s], float).mean() - 5.0) for s in sorted(r)])
    ax_b.scatter(anchor, subj_flip * 100, s=9, color=C["P2"], edgecolor="none", alpha=0.9)
    k = np.polyfit(anchor, subj_flip * 100, 1)
    xs = np.linspace(anchor.min(), anchor.max(), 50)
    ax_b.plot(xs, np.polyval(k, xs), color=C["neutral"], lw=0.9, ls=(0, (3, 2)))
    rr = float(np.corrcoef(anchor, subj_flip)[0, 1])
    ax_b.set_ylim(-2.5, 36)
    ax_b.text(0.02, 0.97, f"Pearson r = {rr:.2f}", transform=ax_b.transAxes,
              fontsize=6.5, va="top")
    ax_b.set_xlabel("participant anchor offset |mean(r) - 5|")
    ax_b.set_ylabel("flip rate P0 vs P2 (%)")
    panel_label(ax_b, "(b)")

    # (c) paired per-participant class balance, P0 against P2
    sids = sorted(r)
    hb_p0 = np.array([np.asarray(p0[s], float).mean() * 100 for s in sids])
    hb_p2 = np.array([np.asarray(p2[s], float).mean() * 100 for s in sids])
    for a_, b_ in zip(hb_p0, hb_p2):
        ax_c.plot([0, 1], [a_, b_], color=C["light"], lw=0.7, zorder=1)
    ax_c.axhline(50, color=C["neutral"], lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax_c.scatter(np.zeros(len(hb_p0)), hb_p0, s=11, color=C["P0"],
                 edgecolor="none", zorder=3, label="P0 (fixed)")
    ax_c.scatter(np.ones(len(hb_p2)), hb_p2, s=11, color=C["P2"],
                 edgecolor="none", zorder=3, label="P2 (anchored)")
    sd0, sd2 = hb_p0.std(ddof=1), hb_p2.std(ddof=1)
    ax_c.set_xlim(-0.46, 1.72)
    ax_c.set_ylim(15, 105)
    ax_c.set_xticks([0, 1], ["P0", "P2"])
    ax_c.set_ylabel("high-class share (%)")
    ax_c.text(-0.42, 50.8, "balance", ha="left", va="bottom", fontsize=5.8,
              color=C["neutral"])
    ax_c.text(1.68, 96, f"SD {sd0:.1f} to {sd2:.1f}", ha="right", va="top",
              fontsize=5.8, color=C["neutral"])
    ax_c.legend(loc="lower center", bbox_to_anchor=(0.42, -0.02), ncol=1,
                handlelength=1.0, handletextpad=0.4, borderaxespad=0.0,
                fontsize=5.8)
    panel_label(ax_c, "(c)")

    # (d) diagnostics per dimension
    spread, disp = [], []
    for d in DIMS:
        anchors = [abs(np.asarray(v, float).mean() - 5.0) for v in ratings[d].values()]
        sds = [np.asarray(v, float).std(ddof=0) for v in ratings[d].values()]
        spread.append(float(np.std(anchors)))
        disp.append(float(np.mean(sds)))
    x = np.arange(4)
    ax_d.bar(x - 0.19, spread, 0.36, color=C["neutral"], edgecolor="none",
             label="anchor spread (SD)")
    ax_d.bar(x + 0.19, disp, 0.36, color=C["P2"], edgecolor="none",
             label="rating dispersion (mean SD)")
    ax_d.set_xticks(x, [DIM_LABEL[d] for d in DIMS], rotation=20, ha="right")
    ax_d.set_ylabel("rating-scale units")
    ax_d.set_ylim(0, 3.15)
    ax_d.set_yticks([0, 1, 2, 3])
    ax_d.legend(loc="upper left", ncol=1, handlelength=1.1, handletextpad=0.4,
                borderaxespad=0.2)
    panel_label(ax_d, "(d)")

    fig.tight_layout(w_pad=1.8, h_pad=1.8)
    return fig


def main():
    ratings = load_ratings()
    prot = {d: {p: apply_protocol(ratings[d], p) for p in ("P0", "P1", "P2")} for d in DIMS}
    OUT.mkdir(parents=True, exist_ok=True)

    save_pub(fig1(ratings, prot), OUT / "fig1_protocols")
    plt.close("all")
    save_pub(fig2(ratings, prot), OUT / "fig2_mechanism")
    plt.close("all")

    summary = {d: {"discard_P1": discard(prot[d]["P1"]),
                   "flip_P0P2": flip(prot[d]["P0"], prot[d]["P2"]),
                   "high_P0": high_frac(prot[d]["P0"]),
                   "high_P2": high_frac(prot[d]["P2"])} for d in DIMS}
    (OUT / "fig_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("figures done")


if __name__ == "__main__":
    main()
