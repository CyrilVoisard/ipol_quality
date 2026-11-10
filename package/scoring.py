# Copyright (C) 2026 Cyril Voisard
# SPDX-License-Identifier: AGPL-3.0-or-later

from dataclasses import dataclass, field

from .indicators.base import GROUPS
from .params import SENSORS

UPSTREAM = ("raw", "sensor", "orientation")


@dataclass
class Scores:
    groups: dict                    # group -> score or None
    per_sensor: dict                # sensor -> {upstream group -> score or None}
    global_score: float = None
    critical: list = field(default_factory=list)   # (group, name, sensor, quality)


def combine(values):
    """Score of a set of qualities: its weakest point (None values are ignored)."""
    values = [v for v in values if v is not None]
    return float(min(values)) if values else None


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
    critical = st["critical"]

    per_sensor = {s: {g: combine([i.quality for i in indicators if i.sensor == s and i.group == g])
                      for g in UPSTREAM} for s in SENSORS}
    groups = {}
    for g in GROUPS:
        elements = [per_sensor[s][g] for s in SENSORS] if g in UPSTREAM else []
        elements += [i.quality for i in indicators if i.group == g and i.sensor is None]
        groups[g] = combine(elements)
    # global score: the lowest of the six groups (the recording is as good as its weakest point)
    global_score = combine(list(groups.values()))
    critical_list = [(i.group, i.name, i.sensor, i.quality) for i in indicators
                     if i.quality is not None and i.quality <= critical]
    return Scores(groups=groups, per_sensor=per_sensor, global_score=global_score, critical=critical_list)
