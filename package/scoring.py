# Copyright (C) 2026 Cyril Voisard
# SPDX-License-Identifier: AGPL-3.0-or-later

from dataclasses import dataclass, field

import numpy as np

from .indicators.base import GROUPS
from .params import SENSORS

UPSTREAM = ("raw", "sensor", "orientation")


@dataclass
class Scores:
    groups: dict                    # group -> score or None
    per_sensor: dict                # sensor -> {upstream group -> score or None}
    global_score: float = None
    critical: list = field(default_factory=list)   # (group, name, sensor, quality)


def combine(values, rule="critical_min", critical=50.0):
    """Aggregate qualities (None values are ignored)."""
    values = [v for v in values if v is not None]
    if not values:
        return None
    if rule == "critical_min" and min(values) < critical:
        return float(min(values))
    if rule not in ("critical_min", "mean"):
        raise ValueError("Unknown aggregation rule: {}".format(rule))
    return float(np.mean(values))


def aggregate(indicators, params):
    st = params["settings"]
    rule, critical = st.get("aggregation", "critical_min"), st["critical"]

    per_sensor = {s: {g: combine([i.quality for i in indicators if i.sensor == s and i.group == g],
                                 rule, critical) for g in UPSTREAM} for s in SENSORS}
    groups = {}
    for g in GROUPS:
        elements = [per_sensor[s][g] for s in SENSORS] if g in UPSTREAM else []
        elements += [i.quality for i in indicators if i.group == g and i.sensor is None]
        groups[g] = combine(elements, rule, critical)
    available = [v for v in groups.values() if v is not None]
    critical_list = [(i.group, i.name, i.sensor, i.quality) for i in indicators
                     if i.quality is not None and i.quality < critical]
    return Scores(groups=groups, per_sensor=per_sensor,
                  global_score=float(np.mean(available)) if available else None, critical=critical_list)
