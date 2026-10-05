"""Task 1 单元测试：协议三态、tie 规则、配对语义、丢弃与改判分离。"""

import numpy as np
import pytest

from prlpaper.labels.audit import apply_protocol, balance_shift, discard_rate, flip_rate


def make_ratings() -> dict[int, np.ndarray]:
    # 两个被试，含边界值、模糊区、锚点偏移
    return {
        0: np.array([1.0, 4.0, 4.5, 5.0, 5.1, 6.0, 9.0, 2.0]),
        1: np.array([3.0, 4.0, 5.0, 5.5, 6.0, 8.0, 8.5, 4.0]),
    }


def test_p0_tie_rule():
    lab = apply_protocol(make_ratings(), "P0")
    assert lab[0][3] == 0.0   # 5.0 判低类（LibEER <= 语义）
    assert lab[0][4] == 1.0   # 5.1 判高类
    assert lab[0][0] == 0.0   # 1.0 判低类


def test_p1_three_state():
    lab = apply_protocol(make_ratings(), "P1")
    assert lab[0][1] == 0.0       # 4.0 -> 低
    assert lab[0][5] == 1.0       # 6.0 -> 高
    assert np.isnan(lab[0][2])    # 4.5 丢弃
    assert np.isnan(lab[0][3])    # 5.0 丢弃
    assert np.isnan(lab[1][3])    # 5.5 丢弃


def test_flip_paired_semantics():
    r = make_ratings()
    a, b = apply_protocol(r, "P0"), apply_protocol(r, "P1")
    # 共同保留集：被试0 的 1,4,5.1,6,9,2 与被试1 的 3,4,6,8,8.5
    # 手算：4.0 同低(0=0)；5.1/6.0/9.0/8.0/8.5 同高；1.0/2.0/3.0 同低 → flip = 0
    assert flip_rate(a, b) == 0.0


def test_discard_rate():
    r = make_ratings()
    lab = apply_protocol(r, "P1")
    assert discard_rate(lab) == 5.0 / 16.0   # 4.5,5.0,5.0,5.5 与被试0的4.5 → 共5个NaN


def test_p2_anchor():
    r = make_ratings()
    lab = apply_protocol(r, "P2")
    # 被试0 均值 4.575：低于均值的判 0，不低于均值的判 1
    mu0 = r[0].mean()
    assert lab[0][0] == 0.0          # 1.0 < mu0
    assert lab[0][6] == 1.0          # 9.0 > mu0
    # 本夹具恰好 8 个值中 4 个不低于均值，占比为 0.5；这是夹具的巧合，不是协议保证
    assert lab[0].mean() == 0.5
    assert lab[1].mean() == 0.5


def test_p2_share_is_not_generally_half():
    """z >= 0 判高时，恰好一半只在个别评分构成下成立。

    整数量表上的平局会把占比推离一半，图 2c 与摘要据此表述。
    """
    high_anchor = {0: np.array([1.0] + [9.0] * 7)}   # 均值 8.0
    lab = apply_protocol(high_anchor, "P2")
    assert lab[0].mean() == 7.0 / 8.0                # 不低于均值的 7 个 9.0

    ties_at_mean = {0: np.array([5.0, 5.0, 5.0, 5.0, 6.0, 7.0])}   # 均值 5.5
    lab2 = apply_protocol(ties_at_mean, "P2")
    assert lab2[0].mean() == 2.0 / 6.0               # 只有 6.0 与 7.0 达标


def test_p2_decouples_balance_from_anchor():
    """只用高段的被试用尽高类；P2 仍按自身分布切分，但并非对半。"""
    r = {0: np.array([7.0, 7.0, 8.0, 8.0, 9.0, 9.0])}   # 均值 8.0
    assert apply_protocol(r, "P0")[0].mean() == 1.0
    assert apply_protocol(r, "P2")[0].mean() == 4.0 / 6.0


def test_balance_shift_sign():
    r = make_ratings()
    a, b = apply_protocol(r, "P0"), apply_protocol(r, "P2")
    s = balance_shift(a, b)
    assert isinstance(s, float)


def test_invalid_protocol():
    with pytest.raises(ValueError):
        apply_protocol(make_ratings(), "PX")
