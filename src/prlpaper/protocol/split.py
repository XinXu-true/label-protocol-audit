"""双情境双轴协议划分器（spec §4.2 第三轮修订版）。

两条泄漏轴各自活在**不同的评估情境**里，因为试次轴在 LOSO 下不可识别
（测试被试的试次不可能出现在训练集）。

情境 A：跨被试（cross-subject）——被试轴在此激活
  A0 = "loso"           + 试次轴固定块级   → 干净协议
  A1 = "pooled"         + trial_phi=0      → 仅被试轴泄漏
  A2 = "pooled"         + trial_phi=1      → 领域默认（双轴泄漏）

情境 B：被试内（subject-dependent）——试次轴在此激活
  B0 = "within"         + trial_phi=0      → 干净被试内协议
  B1 = "within"         + trial_phi=1      → 仅试次轴泄漏

结构性事实（本模块断言并测试之）：LOSO 下 trial_phi 无效应。
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

VALID_SUBJECT_MODES = ("loso", "pooled", "within")


@dataclass(frozen=True)
class Window:
    subject: int
    trial: int
    start: int  # 采样点起
    end: int    # 采样点止（不含）
    label: int


def cell_name(subject_mode: str, trial_phi: float) -> str:
    if subject_mode == "loso":
        return "A0"
    if subject_mode == "pooled":
        return "A2" if trial_phi > 0.0 else "A1"
    # within
    return "B1" if trial_phi > 0.0 else "B0"


def split_windows(
    windows: list[Window],
    subject_mode: str,
    trial_phi: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """双情境划分，返回 (train_idx, test_idx)（windows 列表下标）。"""
    if subject_mode not in VALID_SUBJECT_MODES:
        raise ValueError(f"subject_mode ∈ {VALID_SUBJECT_MODES}，收到 {subject_mode}")
    if not 0.0 <= trial_phi <= 1.0:
        raise ValueError(f"trial_phi 必须在 [0,1]，收到 {trial_phi}")
    if not windows:
        return np.array([], dtype=int), np.array([], dtype=int)

    subjects = sorted({w.subject for w in windows})

    if subject_mode == "loso":
        if len(subjects) < 2:
            raise ValueError("loso 模式至少需要 2 个被试")
        test_subject = subjects[int(rng.integers(len(subjects)))]
        train = [i for i, w in enumerate(windows) if w.subject != test_subject]
        test = [i for i, w in enumerate(windows) if w.subject == test_subject]
        return np.asarray(train, dtype=int), np.asarray(test, dtype=int)

    if subject_mode == "within":
        focus = subjects[int(rng.integers(len(subjects)))]
        indices = [i for i, w in enumerate(windows) if w.subject == focus]
    else:  # pooled
        indices = list(range(len(windows)))

    # 试次轴在候选池内生效
    groups: dict[tuple[int, int], list[int]] = {}
    for i in indices:
        w = windows[i]
        groups.setdefault((w.subject, w.trial), []).append(i)

    train: list[int] = []
    test: list[int] = []
    for gidx in groups.values():
        g = np.asarray(gidx, dtype=int)
        if rng.random() < trial_phi:
            side = rng.random(len(g)) < 0.5  # 窗口级随机
            t, e = g[side], g[~side]
        else:
            t, e = (g, np.array([], dtype=int)) if rng.random() < 0.5 else (np.array([], dtype=int), g)
        if len(t) == 0 or len(e) == 0:
            if len(train) <= len(test):
                t, e = g, np.array([], dtype=int)
            else:
                t, e = np.array([], dtype=int), g
        train.extend(np.asarray(t).tolist())
        test.extend(np.asarray(e).tolist())

    return np.asarray(train, dtype=int), np.asarray(test, dtype=int)


def protocol_card(subject_mode: str, trial_phi: float, window_length: int, step: int) -> dict:
    """协议卡片：受控实验的匹配变量审计。"""
    situation = {"loso": "A-cross-subject", "pooled": "A-cross-subject", "within": "B-within-subject"}[subject_mode]
    return {
        "cell": cell_name(subject_mode, trial_phi),
        "situation": situation,
        "subject_mode": subject_mode,
        "trial_phi": trial_phi,
        "trial_axis_identifiable": subject_mode != "loso",
        "window_length": window_length,
        "step": step,
        "overlap": max(0, window_length - step),
        "split_mode": "window-level-random" if trial_phi >= 1.0 else ("trial-block" if trial_phi <= 0.0 else "mixed"),
    }
