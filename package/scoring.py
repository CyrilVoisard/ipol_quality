# Copyright (C) 2026 Cyril Voisard
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Aggregation of the indicators into scores.

Aggregation rule ``agg`` (``settings["aggregation"]``):

* "critical_min" (default): mean of the elements if all of them reach the critical quality
  (``settings["critical"]``, 50), minimum otherwise -- a single failing element (a sensor
  upside down, a long gap on one sensor) cannot be hidden by the others;
* "mean": plain mean.

Levels:

* per sensor, for each upstream group (raw, sensor, protocol): agg of the qualities of the
  indicators of that sensor -- the four quarters of the upstream gauges;
* per group: agg of the per-sensor scores of the group and of its trial-level indicators;
* global: mean of the available group scores.

Indicators whose quality is None (sensor not provided) are left out; a group without any available
element has no score. ``Scores.critical`` lists the indicators below the critical quality.
"""

from dataclasses import dataclass, field

import numpy as np

from .indicators.base import GROUPS
from .params import SENSORS

UPSTREAM = ("raw", "sensor", "protocol")


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
