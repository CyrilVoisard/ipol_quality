# Copyright (C) 2026 Cyril Voisard
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Common structure of the indicators.

Every indicator is always given a quality, with one exception:

* computed: Q(delta; delta50) in [0, 100];
* not computable because of the recording (no standing period after the walk, U-turn not
  segmented, no gait events...): quality 0, the reason being given in ``note``;
* sensor not provided (the head sensor is optional): quality None, left out of the scores.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..params import quality

GROUPS = ("raw", "sensor", "protocol", "uturn", "segmentation")


@dataclass
class Indicator:
    name: str
    group: str
    sensor: Optional[str]        # None for a trial-level indicator
    value: Optional[float]
    unit: str
    delta: Optional[float]
    quality: Optional[float]
    note: str = ""


def make(name, group, sensor, value, unit, delta, params, gate=1.0, note=""):
    """Indicator with quality gate * Q(delta; delta50[name]).

    A deviation that cannot be evaluated (None, NaN, infinite) gives quality 0 with the reason,
    so that the indicator is never silently left out."""
    d = _num(delta)
    if d is None:
        reason = "not computable: invalid value ({})".format(delta)
        return Indicator(name, group, sensor, _num(value), unit, None, 0.0, (note + "; " if note else "") + reason)
    q = float(gate) * quality(d, params["delta50"][name])
    return Indicator(name, group, sensor, _num(value), unit, d, q, note)


def impossible(name, group, sensor, unit, reason):
    """Indicator that cannot be computed because of the recording: quality 0, with the reason."""
    return Indicator(name, group, sensor, None, unit, None, 0.0, "not computable: " + reason)


def absent(name, group, sensor, unit):
    """Indicator of a sensor that was not provided (left out of the scores)."""
    return Indicator(name, group, sensor, None, unit, None, None, "sensor not provided")


def _num(x):
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None
