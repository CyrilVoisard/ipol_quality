# Copyright (C) 2026 Cyril Voisard
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np

from .base import impossible, make

GROUP = "segmentation"


def alternation(gait_events):
    """Percentage of right/left alternation of the heel strikes (None if fewer than 2 events)."""
    feet = gait_events.sort_values(["HS", "TO"])["Foot"].to_numpy(dtype=int)
    if len(feet) < 2:
        return None
    return 100.0 * np.sum(np.abs(np.diff(feet))) / (len(feet) - 1)


def legs(segmentation):
    """Gait events of the walk before and after the U-turn (the alternation of the feet is not
    expected during the turn); the whole walk when the U-turn is not segmented."""
    events, u = segmentation.gait_events, segmentation.uturn
    if u is None:
        return {"walk": events}
    return {"before the U-turn": events[events.HS < u.start], "after the U-turn": events[events.HS > u.end]}


def compute(segmentation, params):
    events = segmentation.gait_events
    if events is None:
        return [impossible("alternation", GROUP, None, "%", "no gait events ({})".format(segmentation.gait_events_error))]
    deltas, notes = [], []
    for name, part in legs(segmentation).items():
        alpha = alternation(part)
        # a straight walk of 10 m has several steps: fewer than 2 events there means missing events
        deltas.append(100.0 if alpha is None else 100.0 - alpha)
        notes.append("{}: {} events{}".format(name, len(part), "" if alpha is None else ", {:.0f} % alternation".format(alpha)))
    delta = float(np.mean(deltas))
    return [make("alternation", GROUP, None, 100.0 - delta, "%", delta, params,
                 note="; ".join(notes) + ("" if segmentation.uturn is not None else " (U-turn not segmented)"))]
