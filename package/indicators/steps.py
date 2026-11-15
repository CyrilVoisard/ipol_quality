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


def flight(part):
    """Samples with both feet in the air in a straight walk, and its length (samples), from its
    first to its last gait event. A foot is in the air from the toe-off to the heel strike of each
    of its steps; a step whose heel strike comes before its toe-off is impossible and its whole
    span counts as well (the gait parameters read each step from its toe-off to its heel strike)."""
    start = int(min(part.TO.min(), part.HS.min()))
    end = int(max(part.TO.max(), part.HS.max())) + 1
    air = {0: np.zeros(end - start, bool), 1: np.zeros(end - start, bool)}
    wrong = np.zeros(end - start, bool)
    for foot, to, hs in part[["Foot", "TO", "HS"]].itertuples(index=False):
        to, hs = int(to) - start, int(hs) - start
        if to < hs:
            air[int(foot)][to:hs] = True
        else:
            wrong[hs:to + 1] = True
    return int(np.sum((air[0] & air[1]) | wrong)), end - start


def compute(segmentation, params):
    events = segmentation.gait_events
    if events is None:
        reason = "no gait events ({})".format(segmentation.gait_events_error)
        return [impossible("alternation", GROUP, None, "%", reason), impossible("flight", GROUP, None, "%", reason)]
    out = []
    deltas, notes = [], []
    for name, part in legs(segmentation).items():
        alpha = alternation(part)
        # a straight walk of 10 m has several steps: fewer than 2 events there means missing events
        deltas.append(100.0 if alpha is None else 100.0 - alpha)
        notes.append("{}: {} events{}".format(name, len(part), "" if alpha is None else ", {:.0f} % alternation".format(alpha)))
    delta = float(np.mean(deltas))
    out.append(make("alternation", GROUP, None, 100.0 - delta, "%", delta, params,
                    note="; ".join(notes) + ("" if segmentation.uturn is not None else " (U-turn not segmented)")))

    # both feet in the air: never in a walk; any time costs 25 points, beyond the second threshold 50
    in_air = length = 0
    for part in legs(segmentation).values():
        if len(part) >= 2:
            n, total = flight(part)
            in_air, length = in_air + n, length + total
    if length == 0:
        out.append(impossible("flight", GROUP, None, "%", "fewer than 2 gait events on each straight walk"))
        return out
    share = 100.0 * in_air / length
    q = 100.0
    for threshold, quality_above in params["settings"]["flight_quality"]:
        if share > threshold:
            q = quality_above
    out.append(make("flight", GROUP, None, share, "%", share, params, q=q,
                    note="{:.1f} % of the straight walks ({} samples) with both feet in the air".format(share, in_air)))
    return out
