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


def combine(values, rule="min", critical=50.0):
    """Aggregate qualities (None values are ignored): "min" keeps the weakest point,
    "critical_min" the mean unless an element is critical, "mean" the mean."""
    values = [v for v in values if v is not None]
    if not values:
        return None
    if rule not in ("min", "critical_min", "mean"):
        raise ValueError("Unknown aggregation rule: {}".format(rule))
    if rule == "min" or (rule == "critical_min" and min(values) <= critical):
        return float(min(values))
    return float(np.mean(values))


def quality_class(q, settings):
    """"good", "acceptable", "critical" (None if q is None); a quality on a boundary falls in the
    more severe class."""
    if q is None:
        return None
    if q <= settings["critical"]:
        return "critical"
    return "acceptable" if q <= settings["good"] else "good"


def aggregate(indicators, params):
    st = params["settings"]
    rule, critical = st.get("aggregation", "min"), st["critical"]

    per_sensor = {s: {g: combine([i.quality for i in indicators if i.sensor == s and i.group == g],
                                 rule, critical) for g in UPSTREAM} for s in SENSORS}
    groups = {}
    for g in GROUPS:
        elements = [per_sensor[s][g] for s in SENSORS] if g in UPSTREAM else []
        elements += [i.quality for i in indicators if i.group == g and i.sensor is None]
        groups[g] = combine(elements, rule, critical)
    # global score: same rule over the six groups (by default their minimum: the recording is as
    # good as its weakest point)
    global_score = combine(list(groups.values()), rule, critical)
    critical_list = [(i.group, i.name, i.sensor, i.quality) for i in indicators
                     if i.quality is not None and i.quality <= critical]
    return Scores(groups=groups, per_sensor=per_sensor, global_score=global_score, critical=critical_list)
