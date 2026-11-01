# Copyright (C) 2026 Cyril Voisard
# SPDX-License-Identifier: AGPL-3.0-or-later

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Window:
    label: str
    start: int        # first sample
    end: int          # last sample + 1

    def slice(self):
        return slice(self.start, self.end)

    def duration(self, freq):
        return (self.end - self.start) / freq


@dataclass
class Stillness:
    t_before: float                # standing duration before the walk (s)
    t_after: float                 # standing duration after the walk (s)
    before: Optional[Window]       # still window before the walk
    after: Optional[Window]        # still window after the walk
    rest: Window                   # window used by the rest indicators
    active: Window                 # the walk
    activity_acc: np.ndarray
    activity_gyr: np.ndarray
    threshold_acc: float
    threshold_gyr: float


def _moving_std(x, win):
    return pd.Series(x).rolling(win, center=True, min_periods=1).std(ddof=0).to_numpy()


def runs(mask):
    """(start, end) of the runs of True values of a boolean array (end excluded)."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=int), [0])))
    return list(zip(np.where(edges == 1)[0].tolist(), np.where(edges == -1)[0].tolist()))


def activity(aligned, sensors, win_s):
    """Acceleration and angular velocity activities, maximum over ``sensors``."""
    win = max(3, int(round(win_s * aligned.freq)))
    acc = np.max([_moving_std(np.linalg.norm(aligned.acc[s], axis=1), win) for s in sensors], axis=0)
    gyr = np.max([np.sqrt(np.sum([_moving_std(aligned.gyr[s][:, k], win) ** 2 for k in range(3)], axis=0))
                  for s in sensors], axis=0)
    return acc, gyr


def quietest(gyr_activity, length, lo=0, hi=None):
    """Window of ``length`` samples of minimal mean activity within [lo, hi)."""
    hi = len(gyr_activity) if hi is None else hi
    length = max(1, min(length, hi - lo))
    c = np.concatenate(([0.0], np.cumsum(gyr_activity[lo:hi])))
    k = int(np.argmin(c[length:] - c[:-length]))
    return lo + k, lo + k + length


def detect_stillness(aligned, settings):
    freq, n = aligned.freq, aligned.n
    act_acc, act_gyr = activity(aligned, settings["still_sensors"], settings["still_win"])
    centre = slice(n // 3, 2 * n // 3)
    thr_acc = max(settings["still_floor_acc"], settings["still_rel"] * np.median(act_acc[centre]))
    thr_gyr = max(settings["still_floor_gyr"], settings["still_rel"] * np.median(act_gyr[centre]))
    moving = (act_acc >= thr_acc) | (act_gyr >= thr_gyr)

    # the walk: longest movement period once short pauses are merged
    gap = int(round(settings["still_merge_gap"] * freq))
    merged = []
    for b, e in runs(moving):
        if merged and b - merged[-1][1] < gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((b, e))
    onset, offset = max(merged, key=lambda r: r[1] - r[0]) if merged else (n, n)

    margin = int(round(settings["still_margin"] * freq))
    min_len = int(round(settings["still_min_duration"] * freq))

    def longest_still(lo, hi, label):
        cands = [(b, e) for b, e in runs(~moving[lo:hi]) if e - b >= min_len] if hi > lo else []
        if not cands:
            return None
        b, e = max(cands, key=lambda r: r[1] - r[0])
        return Window(label, lo + b, lo + e)

    before = longest_still(0, max(onset - margin, 0), "before")
    after = longest_still(min(offset + margin, n), n, "after")
    rest = before or after
    if rest is None:
        rest = Window("quietest", *quietest(act_gyr, int(round(settings["rest_sub"] * freq))))
    return Stillness(t_before=onset / freq, t_after=(n - offset) / freq, before=before, after=after,
                     rest=rest, active=Window("walk", onset, max(offset, onset)),
                     activity_acc=act_acc, activity_gyr=act_gyr,
                     threshold_acc=float(thr_acc), threshold_gyr=float(thr_gyr))
