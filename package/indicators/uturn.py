# Copyright (C) 2026 Cyril Voisard
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np

from ..segmentation import vertical_yaw_rate
from .base import absent, impossible, make
from .sensor import rest_window

GROUP = "uturn"


def heading_change(aligned, stillness, sensor, start, end, settings):
    """Range of the heading of ``sensor`` (deg) over [start, end)."""
    yaw = vertical_yaw_rate(aligned, sensor, rest_window(aligned, stillness, sensor, settings))
    theta = np.cumsum(yaw) / aligned.freq
    seg = theta[max(start, 0):min(end, aligned.n)]
    return float(np.max(seg) - np.min(seg)) if len(seg) else 0.0


def compute(aligned, stillness, segmentation, params):
    u = segmentation.uturn
    if u is None:
        reason = "U-turn not segmented ({})".format(segmentation.uturn_error)
        out = [impossible("uturn_fit", GROUP, None, "-", reason),
               impossible("uturn_amplitude", GROUP, None, "deg", reason),
               impossible("straight_dev", GROUP, None, "-", reason)]
        out.append(absent("head_trunk", GROUP, None, "-") if "HE" not in aligned.acc
                   else impossible("head_trunk", GROUP, None, "-", reason))
        return out

    out = [make("uturn_fit", GROUP, None, u.r2, "-", max(0.0, 1.0 - u.r2), params),
           make("uturn_amplitude", GROUP, None, u.amplitude, "deg", abs(180.0 - u.amplitude), params)]
    if u.a == 0:
        out.append(impossible("straight_dev", GROUP, None, "-", "U-turn model of zero amplitude"))
    else:
        dev = abs(u.C) * abs(u.c) / abs(u.a)
        out.append(make("straight_dev", GROUP, None, dev, "-", dev, params))
    if "HE" not in aligned.acc:
        out.append(absent("head_trunk", GROUP, None, "-"))
        return out
    st = params["settings"]
    m = int(round(st["head_trunk_margin"] * aligned.freq))
    amp_lb = heading_change(aligned, stillness, "LB", u.start - m, u.end + m + 1, st)
    amp_he = heading_change(aligned, stillness, "HE", u.start - m, u.end + m + 1, st)
    ratio = min(amp_lb, amp_he) / max(amp_lb, amp_he) if max(amp_lb, amp_he) > 0 else 0.0
    out.append(make("head_trunk", GROUP, None, ratio, "-", 1.0 - ratio, params,
                    note="head {:.0f} deg, trunk {:.0f} deg".format(amp_he, amp_lb)))
    return out
