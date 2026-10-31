# Copyright (C) 2026 Cyril Voisard
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Group "segmentation": physiological consistency of the gait events.

* alternation -- the heel strikes of both feet, merged in chronological order, must alternate
                 between left and right: alpha = 100 * (number of foot changes) / (L - 1) for L
                 events, delta = 100 - alpha (%). Quality 0 if fewer than 2 events are given.
"""

import numpy as np

from .base import impossible, make

GROUP = "segmentation"


def alternation(gait_events):
    """Percentage of right/left alternation of the heel strikes (None if fewer than 2 events)."""
    feet = gait_events.sort_values(["HS", "TO"])["Foot"].to_numpy(dtype=int)
    if len(feet) < 2:
        return None
    return 100.0 * np.sum(np.abs(np.diff(feet))) / (len(feet) - 1)


def compute(segmentation, params):
    events = segmentation.gait_events
    if events is None:
        return [impossible("alternation", GROUP, None, "%", "no gait events ({})".format(segmentation.gait_events_error))]
    alpha = alternation(events)
    if alpha is None:
        return [impossible("alternation", GROUP, None, "%", "fewer than 2 gait events")]
    return [make("alternation", GROUP, None, alpha, "%", 100.0 - alpha, params,
                 note="{} gait events".format(len(events)))]
