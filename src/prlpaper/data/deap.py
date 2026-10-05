"""DEAP 装载层（自研，纯 numpy/pickle）。

torcheeg 数据集模块因依赖链问题（torch_scatter/spectrum）不作为必需依赖；
本装载层直接读取官方预处理 .dat（Python 2 pickle，需 encoding='latin1'），
窗口化交给本模块，划分交给 prlpaper.protocol.split。
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from prlpaper.protocol.split import Window

# DEAP 预处理版结构：每被试 data (40 trials, 40 channels, 8064 samples@128Hz)、
# labels (40, 4)：valence, arousal, dominance, liking（1-9 自评）。
# 8064 样本 = 前 384 样本（3 s）基线 + 后 7680 样本（60 s）视频观测。
N_TRIALS = 40
N_SAMPLES = 8064
BASELINE_SAMPLES = 384      # 3 s @ 128 Hz，情绪诱发前静息
VIDEO_START = BASELINE_SAMPLES
VIDEO_SAMPLES = N_SAMPLES - BASELINE_SAMPLES  # 7680（60 s）


def load_subject(root: Path, sid: str) -> dict:
    path = root / f"{sid}.dat"
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def make_windows(
    subj_data: dict,
    subject: int,
    window_length: int = 128,
    step: int = 128,
    label_col: int = 0,
    label_thresh: float = 5.0,
    n_channels: int = 32,
) -> list[Window]:
    """把一个被试的数据切成窗口列表（不复制信号，只存索引元数据）。

    只在视频观测段（跳过前 3 s 基线）内切窗。
    """
    labels = subj_data["labels"]
    windows: list[Window] = []
    for trial in range(N_TRIALS):
        label = 1 if labels[trial, label_col] > label_thresh else 0
        for start in range(VIDEO_START, N_SAMPLES - window_length + 1, step):
            windows.append(
                Window(subject=subject, trial=trial, start=start,
                       end=start + window_length, label=label)
            )
    return windows


def window_array(subj_data: dict, w: Window, n_channels: int = 32,
                 baseline_correct: bool = True) -> np.ndarray:
    """取窗口信号：形状 (n_channels, window_length)，float32。

    baseline_correct=True 时按 DEAP 惯例减去该试次 3 s 基线的通道均值
    （基线取全基线段的均值，逐通道），再截取窗口。
    """
    trial = np.asarray(subj_data["data"][w.trial, :n_channels, :], dtype=np.float64)
    if baseline_correct:
        base = trial[:, :BASELINE_SAMPLES].mean(axis=1, keepdims=True)
        trial = trial - base
    return np.asarray(trial[:, w.start : w.end], dtype=np.float32)
